"""Small pose boundary, including an explicit adapter for saved COLMAP poses."""
from abc import ABC,abstractmethod
from pathlib import Path
import json,numpy as np
from .runtime_types import PoseEstimate


class PoseBackend(ABC):
    @abstractmethod
    def track(self,frame,imu_samples=None):pass
    def get_keyframes(self):return []
    def get_pose_updates(self,since_version=0):return []
    def save_checkpoint(self,path):
        raise NotImplementedError('Native optimizer resume unavailable; replay the source prefix')
    def close(self):pass


class ColmapPoseBackend(PoseBackend):
    """A saved accepted model is reusable input, explicitly not fresh tracking."""
    def __init__(self,model_path):
        import pycolmap
        from .hybrid_dense import pose4
        self.path=Path(model_path);self.model=pycolmap.Reconstruction(self.path)
        self.poses={im.name:np.linalg.inv(pose4(im)) for im in self.model.images.values() if im.has_pose}

    def track(self,frame,imu_samples=None):
        valid=frame.frame_id in self.poses
        return PoseEstimate(frame.frame_id,frame.timestamp,self.poses.get(frame.frame_id,np.eye(4)),
            'colmap_saved',state='tracking' if valid else 'lost',diagnostics={'cached_model':str(self.path),'independent_ground_truth':False})


def build_colmap_model(frames,directory,*,use_gpu=False,threads=4,max_seconds=90):
    """CPU-compatible offline baseline; its full initialization cost is timed."""
    import pycolmap
    from .runtime_metrics import save_json
    directory=Path(directory);images=directory/'images';images.mkdir(parents=True,exist_ok=True)
    if len(frames)<5:raise ValueError('COLMAP requires at least five selected frames')
    first=next(iter(frames.values()));k=np.asarray(first['K']);db=directory/'database.db'
    reader=pycolmap.ImageReaderOptions();reader.camera_model='PINHOLE'
    reader.camera_params=','.join(map(str,[k[0,0],k[1,1],k[0,2],k[1,2]]))
    for name,record in frames.items():
        target=images/name
        if not target.exists():target.symlink_to(Path(record['path']).resolve())
    extraction=pycolmap.FeatureExtractionOptions();extraction.use_gpu=use_gpu;extraction.num_threads=threads
    extraction.sift.max_num_features=4096
    device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    pycolmap.extract_features(db,images,camera_mode=pycolmap.CameraMode.SINGLE,reader_options=reader,extraction_options=extraction,device=device)
    names=list(frames);pair_file=directory/'pairs.txt'
    pair_file.write_text(''.join(f'{a} {b}\n' for i,a in enumerate(names) for b in names[i+1:i+9]))
    matching=pycolmap.FeatureMatchingOptions();matching.use_gpu=use_gpu;matching.num_threads=threads
    pairs=pycolmap.ImportedPairingOptions();pairs.match_list_path=pair_file
    pycolmap.match_image_pairs(db,matching_options=matching,pairing_options=pairs,device=device)
    options=pycolmap.IncrementalPipelineOptions();options.num_threads=threads;options.mapper.num_threads=threads
    from .gpu_sfm import configure_mapping_ba
    configure_mapping_ba(options,{'sfm_use_gpu':use_gpu,'sfm_ba_backend':'auto'},pycolmap)
    options.mapper.init_min_tri_angle=4. # Match the existing mapper's baseline, not the native 16-degree default.
    options.max_runtime_seconds=max_seconds;options.min_model_size=5;options.ba_refine_focal_length=False;options.ba_refine_extra_params=False
    output=directory/'mapping';output.mkdir(exist_ok=True)
    models=pycolmap.incremental_mapping(db,images,output,options=options)
    if not models:raise RuntimeError('COLMAP bootstrap did not produce a model')
    model=max(models.values(),key=lambda m:m.num_reg_images());(directory/'sparse').mkdir(exist_ok=True)
    errors=[point.error for point in model.points3D.values() if np.isfinite(point.error)]
    if not errors or np.median(errors)>2:raise RuntimeError('COLMAP pose quality gate failed: reprojection exceeds 2 pixels')
    model.write(directory/'sparse')
    save_json(directory/'backend.json',{'backend':'colmap','offline_bootstrap':True,'use_gpu':use_gpu,'registered_cameras':model.num_reg_images()})
    return ColmapPoseBackend(directory/'sparse')
