"""Deterministic metric simulation fixture; no learned or estimated ground truth."""
import argparse,json
from pathlib import Path
import numpy as np
from .runtime_types import DepthFrame
from .submaps import SubmapStore,save_depth
from .runtime_metrics import save_json


def procedural_frames(count=12,height=48,width=64):
    k=np.array([[.9*width,0,width/2],[0,.9*width,height/2],[0,0,1.]])
    yy,xx=np.indices((height,width));rays=np.stack(((xx-width/2)/k[0,0],(yy-height/2)/k[1,1],np.ones_like(xx)),axis=-1)
    for i in range(count):
        pose=np.eye(4);pose[0,3]=i*.025
        depth=np.full((height,width),4.,np.float32)
        near=rays*2+pose[:3,3];inside=(np.abs(near[:,:,0])<.4)&(np.abs(near[:,:,1])<.35)
        depth[inside]=2.
        world=rays*depth[:,:,None]+pose[:3,3]
        pattern=((np.floor(world[:,:,0]*10)+np.floor(world[:,:,1]*10))%2)*100+80
        rgb=np.stack((pattern,np.where(inside,210,130),255-pattern),axis=-1).astype(np.uint8)
        yield DepthFrame(f'frame_{i:06d}.jpg',i/10,depth,rgb,k,pose,f'synthetic:{i}',
            source='simulated_ground_truth',scale_status='synthetic_metric_ground_truth',validation={'ground_truth_source':'analytic ray intersections'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--frames',type=int,default=12)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    store=SubmapStore(args.output/'submaps',voxel_fraction=.02,max_active=1,max_frames=8)
    for frame in procedural_frames(args.frames):
        save_depth(args.output/'dataset'/(frame.frame_id+'.npz'),frame);store.put(frame)
    result=args.output/'multiview';result.mkdir(exist_ok=True);mesh=store.export(result/'scene_mesh.ply')
    if not len(mesh.triangles):raise RuntimeError('Synthetic TSDF produced no surface')
    from .hybrid_dense import export_glb
    from .mesh_viewer import write_mesh_viewer
    export_glb(mesh,result/'scene_colored.glb')
    save_json(result/'dense_metrics.json',{'mesh_vertices':len(mesh.vertices),'mesh_faces':len(mesh.triangles),
        'quality_warning':'Synthetic analytic RGB/depth/pose fixture in metres; not a real-world accuracy benchmark'})
    write_mesh_viewer(result,Path(__file__).resolve().parents[2]/'tools/three_0.169.0')
    save_json(args.output/'dataset/metadata.json',{'source':'procedural analytic geometry','units':'metres','depth_convention':'camera_z','pose_convention':'world_from_camera',
        'frames':args.frames,'training_dataset':False})
    print(json.dumps({'vertices':len(mesh.vertices),'triangles':len(mesh.triangles),'output':str(result)},indent=2))


if __name__=='__main__':main()
