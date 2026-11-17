"""CPU ray-cast coverage check. Coverage is not geometric accuracy validation."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .open3d_cpu import cpu_bindings
from .preview_mesh import rotation
from .session_reconstruct import save_json


def audit_surface(root, threads=8, views=9, min_view_coverage=.35, min_supported_fraction=.6):
    o3d=cpu_bindings()
    mesh=o3d.io.read_triangle_mesh(str(root/'dense'/'scene_mesh.ply'))
    scene=o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices).astype(np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.triangles).astype(np.uint32)))
    data=json.loads((root/'camera_poses.json').read_text())
    poses=sorted(data['poses'],key=lambda p:p['image'])
    results=[]
    for i in np.unique(np.linspace(0,len(poses)-1,min(views,len(poses)),dtype=int)):
        pose=poses[i]
        camera=next(c for c in data['cameras'] if c['id']==pose['camera_id'])
        if camera['model']!='SIMPLE_RADIAL':
            raise ValueError('Surface audit currently requires SIMPLE_RADIAL cameras')
        f,cx,cy,k=camera['parameters']
        intrinsics=np.array([[f,0,cx],[0,f,cy],[0,0,1]])
        x,y=np.meshgrid((np.arange(160)+.5)*camera['width']/160,
                        (np.arange(90)+.5)*camera['height']/90)
        uv=np.column_stack([x.ravel(),y.ravel()])
        normalized=cv2.undistortPoints(uv[:,None,:],intrinsics,np.array([k,0.,0.,0.])).reshape(-1,2)
        r=rotation(pose['quaternion_wxyz'])
        origin=-r.T@np.asarray(pose['translation_world_to_camera'])
        directions=np.column_stack([normalized,np.ones(len(normalized))])@r
        rays=np.column_stack([np.broadcast_to(origin,directions.shape),directions]).astype(np.float32)
        depth=scene.cast_rays(o3d.core.Tensor(rays),nthreads=threads)['t_hit'].numpy()
        coverage=float(np.mean(np.isfinite(depth)&(depth>1e-6)))
        results.append({'image':pose['image'],'ray_hit_fraction':coverage,
                        'sufficient_coverage':coverage>=min_view_coverage})
    fraction=float(np.mean([p['sufficient_coverage'] for p in results])) if results else 0.
    report={'coverage_gate_passed':fraction>=min_supported_fraction,'sampled_views':results,
            'supported_view_fraction':fraction,'min_view_coverage':min_view_coverage,
            'min_supported_fraction':min_supported_fraction,'geometric_accuracy_validated':False,
            'note':'Ray hits detect missing surface coverage, not correct texture, depth or metric accuracy.'}
    save_json(root/'surface_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reconstruction',type=Path)
    args=p.parse_args()
    print(json.dumps(audit_surface(args.reconstruction),indent=2))
