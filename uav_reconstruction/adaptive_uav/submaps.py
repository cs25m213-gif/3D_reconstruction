"""Bounded TSDF working set, persisted observations and correction-safe rebuilds."""
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .runtime_types import DepthFrame,MapUpdate
from .runtime_metrics import save_json
from .hybrid_dense import make_volume,integrate


def save_depth(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta={k:v for k,v in frame.__dict__.items() if k not in ('depth','rgb','K','T_world_from_camera','excluded','confidence','evidence')}
    meta['context_poses']={k:np.asarray(v).tolist() for k,v in meta['context_poses'].items()}
    temp=path.with_suffix('.partial.npz')
    np.savez_compressed(temp,depth=frame.depth,rgb=frame.rgb,K=frame.K,T_world_from_camera=frame.T_world_from_camera,
        excluded=frame.excluded,confidence=frame.confidence,evidence=frame.evidence,metadata=json.dumps(meta,allow_nan=False))
    temp.replace(path)


def load_depth(path):
    with np.load(path) as data:
        meta=json.loads(str(data['metadata']))
        return DepthFrame(**meta,evidence=data['evidence'] if 'evidence' in data else None,
            **{k:data[k] for k in ('depth','rgb','K','T_world_from_camera','excluded','confidence')})


class SubmapStore:
    def __init__(self,directory,voxel_fraction=.01,max_active=2,max_frames=12):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.voxel_fraction=voxel_fraction;self.max_active=max_active;self.max_frames=max_frames
        if max_active<1 or max_frames<1 or voxel_fraction<=0:raise ValueError('Positive map budgets required')
        self.meta={};self.index={};self.volumes=OrderedDict();self.sequence=0;self.observations=set();self.current=None
        if (self.directory/'index.json').exists():
            state=json.loads((self.directory/'index.json').read_text())
            self.meta=state['submaps'];self.index=state['frames'];self.sequence=state['sequence']
            self.observations=set(state['observations']);self.current=state['current']

    def _frame_path(self,name):return self.directory/'frames'/(name+'.npz')

    def _integrate(self,volume,frame,scale):
        # Excluded and invalid pixels never create either a surface or free ray.
        depth=np.where(frame.valid,frame.depth,0).astype(np.float32)
        limit=3*float(np.percentile(depth[depth>0],95)) if depth.any() else 3*scale
        integrate(volume,[{'depth':depth,'tier':(depth>0).astype(np.uint8),'rgb':frame.rgb,
            'K':frame.K,'P':np.linalg.inv(frame.T_world_from_camera)}],[1],limit)

    def _volume(self,key,rebuild=False):
        if rebuild:self.volumes.pop(key,None)
        if key in self.volumes:
            self.volumes.move_to_end(key);return self.volumes[key]
        while len(self.volumes)>=self.max_active:
            old,volume=self.volumes.popitem(last=False);self._export_chunk(old,volume)
        meta=self.meta[key];volume=make_volume(self.voxel_fraction*meta['depth_scale'])
        for name in meta['frames']:
            if name in meta['stale_frames']:continue
            self._integrate(volume,load_depth(self._frame_path(name)),meta['depth_scale'])
        self.volumes[key]=volume;return volume

    def _export_chunk(self,key,volume):
        import open3d as o3d
        mesh=volume.extract_triangle_mesh();mesh.compute_vertex_normals()
        if len(mesh.triangles):o3d.io.write_triangle_mesh(str(self.directory/(key+'.ply')),mesh)
        elif (self.directory/(key+'.ply')).exists():(self.directory/(key+'.ply')).unlink()

    def put(self,frame,replace=False):
        if frame.source=='warped_previous_estimate' and frame.observation_id in self.observations:
            return MapUpdate(self.sequence) # Re-observing a warp is not independent evidence.
        if frame.frame_id in self.index and not replace:return MapUpdate(self.sequence)
        if not frame.valid.any():return MapUpdate(self.sequence)
        key=self.index.get(frame.frame_id)
        scale=float(np.median(frame.depth[frame.valid]))
        if key is None:
            key=self.current
            if key is not None:
                meta=self.meta[key]
                gap=np.linalg.norm(frame.T_world_from_camera[:3,3]-np.asarray(meta['center']))
                new=(len(meta['frames'])>=self.max_frames or meta['tracking_submap']!=frame.submap_id
                    or gap>2*meta['depth_scale'] or not .25<meta['depth_scale']/scale<4)
                if new:key=None
            if key is None:
                key=f'{len(self.meta):06d}';self.current=key
                self.meta[key]={'frames':[],'stale_frames':[],'depth_scale':scale,
                    'center':frame.T_world_from_camera[:3,3].tolist(),'tracking_submap':frame.submap_id}
            # Initialize before appending the new frame, so it is integrated once.
            volume=self._volume(key)
            self.meta[key]['frames'].append(frame.frame_id);self.index[frame.frame_id]=key
        else:volume=None
        save_depth(self._frame_path(frame.frame_id),frame);self.observations.add(frame.observation_id)
        if replace and not .5<self.meta[key]['depth_scale']/scale<2:
            # A corrected pose context can change learned depth scale. Keeping
            # the old tiny voxel would allocate an enormous, misleading surface.
            values=[]
            for name in self.meta[key]['frames']:
                item=load_depth(self._frame_path(name))
                if item.valid.any():values.append(float(np.median(item.depth[item.valid])))
            if values:self.meta[key]['depth_scale']=float(np.median(values))
        stale=self.meta[key]['stale_frames']
        if frame.frame_id in stale:stale.remove(frame.frame_id)
        if volume is None:self._volume(key,rebuild=True)
        else:self._integrate(volume,frame,self.meta[key]['depth_scale'])
        self.sequence+=1;self.meta[key]['revision']=self.sequence;self.checkpoint()
        return MapUpdate(self.sequence,added=(key,),pose_version=frame.pose_version)

    def correct(self,poses,version):
        changed=set()
        for name,key in self.index.items():
            frame=load_depth(self._frame_path(name))
            updated=poses.get(name,frame.T_world_from_camera)
            old_cam=np.linalg.inv(frame.T_world_from_camera)
            new_cam=np.linalg.inv(updated)
            stale=False
            for context,old in frame.context_poses.items():
                new=poses.get(context,np.asarray(old))
                if not np.allclose(old_cam@np.asarray(old),new_cam@new,atol=1e-5,rtol=0):stale=True
            moved=not np.allclose(updated,frame.T_world_from_camera,atol=1e-5,rtol=0)
            if not moved and not stale:continue
            if stale:
                if name not in self.meta[key]['stale_frames']:self.meta[key]['stale_frames'].append(name)
            else:
                frame.T_world_from_camera=np.asarray(updated);frame.pose_version=version
                frame.context_poses={n:np.asarray(poses.get(n,p)).tolist() for n,p in frame.context_poses.items()}
                save_depth(self._frame_path(name),frame)
            changed.add(key)
        for key in changed:self._volume(key,rebuild=True)
        if changed:
            self.sequence+=1
            for key in changed:self.meta[key]['revision']=self.sequence
            self.checkpoint()
        return MapUpdate(self.sequence,added=tuple(sorted(changed)),rebuilding=tuple(k for k in changed if self.meta[k]['stale_frames']),pose_version=version)

    def stale_frames(self):return [name for meta in self.meta.values() for name in meta['stale_frames']]

    def discard(self,names):
        changed=set()
        for name in names:
            key=self.index.pop(name,None)
            if key is None:continue
            changed.add(key);self.meta[key]['frames'].remove(name)
            if name in self.meta[key]['stale_frames']:self.meta[key]['stale_frames'].remove(name)
        for key in changed:
            self.sequence+=1;self.meta[key]['revision']=self.sequence;self._volume(key,rebuild=True)
        if changed:self.checkpoint()

    def checkpoint(self):
        save_json(self.directory/'index.json',{'submaps':self.meta,'frames':self.index,'sequence':self.sequence,
            'observations':sorted(self.observations),'current':self.current})

    def export(self,path):
        import open3d as o3d
        for key,volume in self.volumes.items():self._export_chunk(key,volume)
        result=o3d.geometry.TriangleMesh()
        for key in self.meta:
            part=self.directory/(key+'.ply')
            if part.exists():result+=o3d.io.read_triangle_mesh(str(part))
        if len(result.triangles):
            result.remove_duplicated_vertices();result.remove_duplicated_triangles();result.remove_degenerate_triangles()
            result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(path),result)
        return result
