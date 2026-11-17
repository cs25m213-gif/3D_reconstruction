"""Export real poses without inventing COLMAP feature observations."""
import json,shutil
from pathlib import Path
import numpy as np
from .runtime_metrics import save_json


def write_workspace(directory,frames,poses):
    import pycolmap
    directory=Path(directory);(directory/'sparse').mkdir(parents=True,exist_ok=True)
    # The native reader prefers binary over text. An explicit export must replace
    # old binaries before reading the newly written camera/image text records.
    for old in (directory/'sparse').glob('*.bin'):old.unlink()
    (directory/'images').mkdir(exist_ok=True)
    for name in ('depth_maps','normal_maps','consistency_graphs'):
        (directory/'stereo'/name).mkdir(parents=True,exist_ok=True)
    rec=pycolmap.Reconstruction()
    camera_ids={}
    for index,(name,frame) in enumerate(frames.items(),1):
        if name not in poses:continue
        k=np.asarray(frame['K']);signature=(frame['width'],frame['height'],*k.ravel())
        if signature not in camera_ids:
            cid=len(camera_ids)+1;camera_ids[signature]=cid
            rec.add_camera(pycolmap.Camera(camera_id=cid,model='PINHOLE',width=frame['width'],height=frame['height'],
                params=[k[0,0],k[1,1],k[0,2],k[1,2]]))
        cid=camera_ids[signature]
        p=np.linalg.inv(poses[name]);image=pycolmap.Image(image_id=index,name=name,camera_id=cid)
        # PyCOLMAP 4.x pose storage uses a trivial rig/frame; use the text
        # reader to construct these consistently across supported bindings.
        target=directory/'images'/name;source=Path(frame['path'])
        if not target.exists():target.symlink_to(source.resolve())
    rec.write_text(directory/'sparse')
    lines=[];count=0
    for name,frame in frames.items():
        if name not in poses:continue
        count+=1;k=np.asarray(frame['K']);signature=(frame['width'],frame['height'],*k.ravel());cid=camera_ids[signature]
        p=np.linalg.inv(poses[name]);rotation=pycolmap.Rotation3d(p[:3,:3]).quat
        # Native quaternion is xyzw, COLMAP text is qw qx qy qz.
        q=rotation[[3,0,1,2]]
        lines.append(' '.join(map(str,[count,*q,*p[:3,3],cid,name]))+'\n\n')
    (directory/'sparse/images.txt').write_text(''.join(lines))
    # Do not retain empty rig/frame text files from the cameras-only model:
    # the native reader reconstructs trivial frames from images.txt.
    for name in ('frames.txt','rigs.txt'):
        path=directory/'sparse'/name
        if path.exists():path.unlink()
    model=pycolmap.Reconstruction(directory/'sparse');model.write(directory/'sparse')
    save_json(directory/'pose_provenance.json',{'real_feature_tracks_exported':False,'poses':len(poses),
        'note':'Pose-only export; no point3D records or feature observations fabricated','metric_scale_validated':False})
    return model
