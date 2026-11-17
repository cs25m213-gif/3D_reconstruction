"""Render a reconstructed mesh using CPU polygon rasterization and its poses."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh


def rotation(q):
    w,x,y,z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])


def preview(root):
    poses = json.loads((root/'camera_poses.json').read_text())
    pose = sorted(poses['poses'], key=lambda p:p['image'])[len(poses['poses'])//2]
    cam = next(c for c in poses['cameras'] if c['id']==pose['camera_id'])
    w,h = cam['width'], cam['height']
    f,cx,cy = cam['parameters'][:3]
    r,t = rotation(pose['quaternion_wxyz']), np.asarray(pose['translation_world_to_camera'])
    scene = trimesh.load(root/'dense/scene_textured_clean.obj',force='scene',process=False)
    batches=[]
    for mesh in scene.geometry.values():
        vertices = mesh.vertices @ r.T + t
        projected = np.stack((vertices[:,0]*f/vertices[:,2]+cx,vertices[:,1]*f/vertices[:,2]+cy),axis=1)
        faces = mesh.faces
        valid = np.all(vertices[faces,2]>.001,axis=1)
        faces = faces[valid]
        triangles = projected[faces]
        visible = np.all(np.isfinite(triangles),axis=(1,2)) & np.any(
            (triangles[:,:,0]>=0)&(triangles[:,:,0]<w)&(triangles[:,:,1]>=0)&(triangles[:,:,1]<h),axis=1)
        faces, triangles = faces[visible], triangles[visible]
        depth = vertices[faces,2].mean(axis=1)
        uv = mesh.visual.uv[faces].mean(axis=1)
        material = mesh.visual.material
        texture = np.asarray(material.image.convert('RGB'))
        tx=np.clip((uv[:,0]*texture.shape[1]).astype(int),0,texture.shape[1]-1)
        ty=np.clip(((1-uv[:,1])*texture.shape[0]).astype(int),0,texture.shape[0]-1)
        colors=texture[ty,tx]
        batches.extend(zip(depth,triangles,colors))
    batches.sort(key=lambda b:b[0], reverse=True)
    image=Image.new('RGB',(w,h),(30,34,39))
    draw=ImageDraw.Draw(image)
    for _,triangle,color in batches:
        draw.polygon([tuple(p) for p in triangle],fill=tuple(map(int,color)))
    image.save(root/'preview.jpg',quality=92)
    print(f"Rendered {len(batches)} triangles; view {pose['image']}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reconstruction',type=Path)
    preview(p.parse_args().reconstruction)
