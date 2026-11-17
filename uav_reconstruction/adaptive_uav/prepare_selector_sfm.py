"""CPU-only geometry recovery for RGB-only training recordings.

Fits actual admitted images, avoiding an unverified JB3D video/pose frame-number
association. This yields an SfM proxy, never a new LiDAR ground truth. Each source
gets a durable report of registered and excluded images and failed reconstruction.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .prepare_multisource_selector import emit, write


class GeometryNotAdmitted(ValueError):
    """An inspected source cannot supply qualified reconstruction supervision."""


def reconstruct(job):
    base,output,key,records=job;base,output=Path(base),Path(output)
    import pycolmap as pc
    threads=int(os.environ.get('GSF_CPU_THREADS','4'))
    cv2.setNumThreads(1)
    name=hashlib.sha256(key.encode()).hexdigest()[:16];folder=output/'sfm'/name;folder.mkdir(exist_ok=True,parents=True)
    marker=folder/'status.json'
    if marker.exists() and json.loads(marker.read_text()).get('state')=='complete':return json.loads(marker.read_text())
    images=folder/'images';images.mkdir(exist_ok=True)
    # Source symlinks stay on the data server; never copy datasets to the GPU host.
    for i,r in enumerate(records):
        p=images/f'{i:09d}.jpg'
        if not p.exists():p.symlink_to(base/r['path'])
    if len(records)<8:raise GeometryNotAdmitted(f'Too few temporal images for {key}: {len(records)}')
    db=folder/'database.db';sparse=folder/'sparse';sparse.mkdir(exist_ok=True)
    write(marker,dict(state='recovering_geometry',source=key,images=len(records)))
    extraction=pc.FeatureExtractionOptions();extraction.num_threads=threads;extraction.use_gpu=False;extraction.sift.max_num_features=4096
    reader=pc.ImageReaderOptions();reader.camera_model='OPENCV';reader.camera_params='268.8,268.8,112,112,0,0,0,0'
    pc.extract_features(db,images,camera_mode=pc.CameraMode.SINGLE,reader_options=reader,extraction_options=extraction,device=pc.Device.cpu)
    matching=pc.FeatureMatchingOptions();matching.num_threads=threads;matching.use_gpu=False
    pairing=pc.SequentialPairingOptions();pairing.overlap=20;pairing.quadratic_overlap=True;pairing.num_threads=threads
    pc.match_sequential(db,matching_options=matching,pairing_options=pairing,device=pc.Device.cpu)
    opts=pc.IncrementalPipelineOptions();opts.num_threads=threads;opts.mapper.num_threads=threads;opts.ba_use_gpu=False;opts.random_seed=20261115
    opts.min_model_size=8;opts.mapper.filter_max_reproj_error=3.;opts.max_runtime_seconds=6*3600
    maps=pc.incremental_mapping(db,images,sparse,options=opts)
    if not maps:raise GeometryNotAdmitted(f'No geometrically registered model for {key}')
    all_paths=[];registered=set();reports=[]
    for mid,model in maps.items():
        if model.num_reg_images()<8:continue
        undistorted=folder/f'undistorted_{mid}'
        options=pc.UndistortCameraOptions();options.max_image_size=256
        pc.undistort_images(undistorted,sparse/str(mid),images,undistort_options=options,num_threads=threads)
        model=pc.Reconstruction(undistorted/'sparse')
        points={pid:p.xyz for pid,p in model.points3D.items() if p.error<=3 and p.track.length()>=3}
        rows=[]
        for image in sorted(model.images.values(),key=lambda im:im.name):
            index=int(Path(image.name).stem);camera=model.cameras[image.camera_id]
            P=np.eye(4);P[:3]=image.cam_from_world().matrix()
            image_points=[points[p.point3D_id] for p in image.points2D if p.has_point3D() and p.point3D_id in points]
            if len(image_points)<60:continue
            rgb=cv2.imread(str(undistorted/'images'/image.name))
            if rgb is None:raise ValueError(f'Undistorted image missing: {image.name}')
            h,w=rgb.shape[:2];K=camera.calibration_matrix();K[0]*=256/w;K[1]*=192/h
            rgb=cv2.cvtColor(cv2.resize(rgb,(256,192),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            rows.append(dict(rgb=rgb,K=K,P=P,points=np.asarray(image_points),index=index));registered.add(index)
        dataset=records[0]['dataset']
        paths=emit(output,name+f'_model{mid}',dataset,'train',rows,'recovered_SfM_sparse_proxy_not_independent_LiDAR')
        all_paths.extend(paths);reports.append(dict(model=mid,registered=len(rows),points=len(points),episodes=len(paths)))
    if not all_paths:raise GeometryNotAdmitted(f'No model passed tracked-point/anchor gates for {key}')
    result=dict(state='complete',source=key,images=len(records),registered=len(registered),excluded_indices=[i for i in range(len(records)) if i not in registered],models=reports,episodes=all_paths,
        trust='SfM fitted to admitted training RGB only; no metric geometry or held-out test claim')
    write(marker,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    groups={}
    for line in (a.base/'visual_train.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['dataset'] not in ('jb3d','tanks_temples','dronescapes'):continue
        if row['role']!='train':raise ValueError('Held-out RGB cannot enter SfM training')
        key=row['source']
        if row['dataset']=='dronescapes':key+=':'+Path(row['key']).stem.rsplit('_',1)[0]
        groups.setdefault(key,[]).append(row)
    jobs=[(str(a.base),str(a.output),key,rows) for key,rows in groups.items()]
    # Add a producer marker before launching workers: the gain-label consumer
    # must not finish while geometry recovery can still add new episodes.
    results=[];failures=[];excluded=[]
    write(a.output/'geometry_status.json',dict(state='running',stage='recovering_RGB_only_recordings',sources=len(jobs)))
    with ProcessPoolExecutor(3) as pool:
        fs={pool.submit(reconstruct,j):j[2] for j in jobs}
        for f in as_completed(fs):
            try:results.append(f.result())
            except GeometryNotAdmitted as exc:
                excluded.append(dict(source=fs[f],reason=str(exc)))
                print(json.dumps(excluded[-1]),flush=True)
            except Exception as exc:
                failure=dict(source=fs[f],error=repr(exc));failures.append(failure)
                print(json.dumps(failure),flush=True)
            state=dict(state='running',completed=len(results)+len(failures)+len(excluded),total=len(jobs),failures=failures,excluded_sources=excluded)
            write(a.output/'sfm_status.json',state)
    state=dict(state='failed' if failures else 'complete',completed=len(results)+len(failures)+len(excluded),total=len(jobs),failures=failures,excluded_sources=excluded,results=results)
    write(a.output/'sfm_status.json',state)
    write(a.output/'geometry_status.json',dict(state=state['state'],stage='RGB_only_geometry_recovery_finished',failures=failures,excluded_sources=excluded))


if __name__=='__main__':main()
