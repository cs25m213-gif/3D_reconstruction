"""Prepare auditable reconstruction episodes from existing remote geometry.

UAVLight uses a sparse SfM proxy, not independent metric ground truth. M3ED
uses released distorted-event-camera z-depth, rig extrinsics and interpolated
world-from-camera poses to project LiDAR into undistorted RGB. No downloads.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def write(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def partition(points):
    # Stable spatial partition: repeated observations of one cell cannot provide
    # both scale anchors and scored reference geometry.
    cell=np.floor(points*100).astype(np.int64)
    return ((cell[:,0]*73856093 ^ cell[:,1]*19349663 ^ cell[:,2]*83492791)&1)==0


def cloud(depth, K, world, stride=1):
    yy,xx=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[yy,xx];good=np.isfinite(z)&(z>0)&(z<250)
    rays=np.linalg.solve(K,np.stack((xx[good],yy[good],np.ones(good.sum()))))
    p=(rays*z[good]).T
    return p@world[:3,:3].T+world[:3,3]


def project(points, P, K, shape=(192,256)):
    cam=points@P[:3,:3].T+P[:3,3]
    uv=cam@K.T; uv=uv[:,:2]/np.maximum(cam[:,2:],1e-9)
    good=(cam[:,2]>.05)&(cam[:,2]<250)&np.isfinite(uv).all(1)
    good&=(uv[:,0]>=0)&(uv[:,0]<shape[1])&(uv[:,1]>=0)&(uv[:,1]<shape[0])
    q=uv[good].astype(int);d=np.full(shape,np.inf,np.float32)
    np.minimum.at(d,(q[:,1],q[:,0]),cam[good,2]);d[~np.isfinite(d)]=0
    return d


def emit(output, name, dataset, role, rows, trust):
    paths=[]
    # Every admitted view belongs to a window; retain a short tail by merging it.
    groups=[rows[i:i+32] for i in range(0,len(rows),32)]
    if len(groups)>1 and len(groups[-1])<3:groups[-2].extend(groups.pop())
    for j,group in enumerate(groups):
        if len(group)<3:continue
        stem=f'{dataset}_{name}_{j:05d}'
        path=output/'geometry'/f'{stem}.npz';meta=path.with_suffix('.json')
        if not path.exists():
            refs=[];good=[]
            for row in group:
                pts=row['points'];anchor=partition(pts)
                a=project(pts[anchor],row['P'],row['K'])
                if np.count_nonzero(a)<30:continue
                row=dict(row,anchors=a);good.append(row);refs.append(pts[~anchor])
            if len(good)<3 or sum(map(len,refs))<100:continue
            reference=np.concatenate(refs)
            cells=np.floor(reference*100).astype(np.int64)
            reference=reference[np.unique(cells,axis=0,return_index=True)[1]]
            temp=path.with_suffix('.partial.npz')
            np.savez_compressed(temp,rgb=np.stack([r['rgb'] for r in good]),
                K=np.stack([r['K'] for r in good]),P=np.stack([r['P'] for r in good]),
                anchors=np.stack([r['anchors'] for r in good]),reference_points=reference,
                source_index=np.asarray([r['index'] for r in good]))
            temp.replace(path)
            write(meta,dict(clip=stem,dataset=dataset,flight=name,split=role,trust=trust,
                frames=len(good),source_indices=[r['index'] for r in good],
                reference_partition='disjoint spatial-cell hash; even anchors, odd reference',
                scope='reference poses; fixed-context student; explicit source geometry trust'))
        paths.append(str(path))
    return paths


def uavlight(job):
    root,out,row=job;root,out=Path(root),Path(out)
    cache=root/row['cache'];rows=[]
    for i,p in enumerate(sorted(cache.glob('frame_*.jpg.npz'))):
        with np.load(p) as z:
            P=z['P'].copy();K=z['K'].copy();anchors=z['anchors']
            points=cloud(anchors,K,np.linalg.inv(P))
            rows.append(dict(rgb=z['rgb'].copy(),K=K,P=P,points=points,index=i))
    return emit(out,row['name'],'uavlight',row['role'],rows,'publisher_SfM_sparse_proxy_not_independent_LiDAR')


def m3ed(job):
    data,out=map(Path,job)
    import h5py
    name=data.stem.removesuffix('_data')
    role='validation' if 'fast_flight_2' in name else ('test' if 'forest_road' in name else 'train')
    gt=data.with_name(name+'_depth_gt.h5')
    if not gt.exists():raise ValueError(f'Missing released depth: {gt}')
    rows=[]
    with h5py.File(data) as f,h5py.File(gt) as g:
        def calibration(prefix):
            fx,fy,cx,cy=f[prefix+'/intrinsics'][()]
            return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),f[prefix+'/distortion_coeffs'][()]
        Kc,Dc=calibration('prophesee/left/calib');Kr,Dr=calibration('ovc/rgb/calib')
        C_T_R=f['ovc/rgb/calib/T_to_prophesee_left'][()];R_T_C=np.linalg.inv(C_T_R)
        ts=g['ts'][:].astype(float);rgbts=f['ovc/ts'][:].astype(float)
        if np.any(np.diff(ts)<=0) or np.any(np.diff(rgbts)<=0):raise ValueError('Nonmonotonic timestamps')
        world=np.linalg.inv(g['Cn_T_C0'][:])
        rotation=Slerp(ts,Rotation.from_matrix(world[:,:3,:3]))
        h,w=f['ovc/rgb/data'].shape[1:3]
        K=Kr.copy();K[0]*=256/w;K[1]*=192/h
        mx,my=cv2.initUndistortRectifyMap(Kr,Dr,None,K,(256,192),cv2.CV_32FC1)
        # Released depth is projected with distortion; undistort rays, not z.
        dh,dw=g['depth/prophesee/left'].shape[1:]
        yy,xx=np.mgrid[0:dh:4,0:dw:4]
        pixels=np.stack((xx,yy),-1).astype(np.float64).reshape(-1,1,2)
        rays=cv2.undistortPoints(pixels,Kc,Dc).reshape(-1,2)
        rays=np.column_stack((rays,np.ones(len(rays))))
        for i,t in enumerate(rgbts):
            if t<ts[0] or t>ts[-1]:continue
            j=int(np.abs(ts-t).argmin())
            if abs(ts[j]-t)>25000:continue
            d=g['depth/prophesee/left'][j][yy,xx].reshape(-1)
            valid=np.isfinite(d)&(d>.05)&(d<250)
            xyz=rays[valid]*d[valid,None]
            pts=xyz@world[j,:3,:3].T+world[j,:3,3]
            W_T_C=np.eye(4);W_T_C[:3,:3]=rotation([t]).as_matrix()[0]
            W_T_C[:3,3]=[np.interp(t,ts,world[:,axis,3]) for axis in range(3)]
            W_T_R=W_T_C@C_T_R
            image=cv2.remap(f['ovc/rgb/data'][i],mx,my,cv2.INTER_LINEAR)
            # Reference population is restricted to this RGB camera's support.
            P=np.linalg.inv(W_T_R);depth=project(pts,P,K)
            pts=cloud(depth,K,W_T_R)
            rows.append(dict(rgb=image,K=K,P=P,points=pts,index=i))
            if len(rows)==32:
                # Flush incrementally: this function must not retain full videos in RAM.
                paths=emit(Path(out),name+f'_at{i:06d}','m3ed',role,rows,'released_LiDAR_camera_depth_registered_rig_static_scene')
                with (Path(out)/f'm3ed_{name}.episodes.jsonl').open('a') as stream:
                    for p in paths:stream.write(json.dumps(p)+'\n')
                rows=[]
        if rows:
            paths=emit(Path(out),name+'_tail','m3ed',role,rows,'released_LiDAR_camera_depth_registered_rig_static_scene')
            with (Path(out)/f'm3ed_{name}.episodes.jsonl').open('a') as stream:
                for p in paths:stream.write(json.dumps(p)+'\n')
    return list(map(str,(Path(out)/'geometry').glob(f'm3ed_{name}*.npz')))


def kitti(job):
    archive,out=map(Path,job);date=archive.name[:10];name=archive.stem
    role='validation' if '2011_09_30_drive_0016' in name else 'train'
    # Coordinate conventions follow utiasSTARS/pykitti utils.py and raw.py.
    with zipfile.ZipFile(archive.parent/f'{date}_calib.zip') as c:
        def read(suffix):
            text=c.read(next(n for n in c.namelist() if n.endswith(suffix))).decode();r={}
            for line in text.splitlines():
                key,values=line.split(':',1)
                try:r[key]=np.array(list(map(float,values.split())))
                except ValueError:pass
            return r
        def rigid(data):
            T=np.eye(4);T[:3,:3]=data['R'].reshape(3,3);T[:3,3]=data['T'];return T
        V_T_I=rigid(read('calib_imu_to_velo.txt'));C0_T_V=rigid(read('calib_velo_to_cam.txt'))
        cam=read('calib_cam_to_cam.txt');projection=cam['P_rect_02'].reshape(3,4);K0=projection[:,:3]
        rect=np.eye(4);rect[:3,:3]=cam['R_rect_00'].reshape(3,3)
        offset=np.eye(4);offset[:3,3]=np.linalg.solve(K0,projection[:,3])
        C_T_V=offset@rect@C0_T_V;C_T_I=C_T_V@V_T_I
    rows=[];origin=None;latitude_scale=None
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist());images=sorted(n for n in names if '/image_02/data/' in n and n.endswith('.png'))
        for i,n in enumerate(images):
            prefix=n.split('/image_02/')[0];stem=Path(n).stem
            gps=f'{prefix}/oxts/data/{stem}.txt';scan=f'{prefix}/velodyne_points/data/{stem}.bin'
            if gps not in names or scan not in names:raise ValueError(f'Missing synchronized sensor sample: {n}')
            values=np.fromstring(z.read(gps).decode(),sep=' ')
            lat,lon,alt,roll,pitch,yaw=values[:6]
            if latitude_scale is None:latitude_scale=np.cos(np.deg2rad(lat))
            translation=np.array([latitude_scale*lon*np.pi*6378137/180,
                latitude_scale*6378137*np.log(np.tan((90+lat)*np.pi/360)),alt])
            if origin is None:origin=translation.copy()
            W_T_I=np.eye(4);W_T_I[:3,:3]=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix();W_T_I[:3,3]=translation-origin
            W_T_C=W_T_I@np.linalg.inv(C_T_I);W_T_V=W_T_I@np.linalg.inv(V_T_I)
            lidar=np.frombuffer(z.read(scan),np.float32).reshape(-1,4)[:,:3]
            points=lidar@W_T_V[:3,:3].T+W_T_V[:3,3]
            image=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(f'Invalid KITTI image {n}')
            h,w=image.shape[:2];K=K0.copy();K[0]*=256/w;K[1]*=192/h
            rgb=cv2.cvtColor(cv2.resize(image,(256,192),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            P=np.linalg.inv(W_T_C);points=cloud(project(points,P,K),K,W_T_C)
            rows.append(dict(rgb=rgb,K=K,P=P,points=points,index=int(stem)))
    return emit(out,name,'kitti',role,rows,'released_Velodyne_OXTS_rectified_rig_static_scene')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);(a.output/'geometry').mkdir(exist_ok=True)
    root=a.root/'geostreamfusion_prepared_20261107/uavlight_v1'
    manifest=json.loads((root/'manifest.json').read_text())
    jobs=[('uavlight',(str(root),str(a.output),r)) for r in manifest['clips'] if r['role'] in ('train','validation')]
    jobs += [('m3ed',(str(f),str(a.output))) for f in sorted((a.root/'geostreamfusion_direct_20261107/m3ed').rglob('*_data.h5'))]
    jobs += [('kitti',(str(f),str(a.output))) for f in sorted((a.root/'student_training_v1/archives/kitti').glob('*_sync.zip'))]
    failures=[];done=0
    with ProcessPoolExecutor(4) as pool:
        fs={pool.submit({'uavlight':uavlight,'m3ed':m3ed,'kitti':kitti}[kind],job):(kind,str(job)[-200:]) for kind,job in jobs}
        for future in as_completed(fs):
            try:result=future.result();print(json.dumps({'source':fs[future],'episodes':len(result)}),flush=True)
            except Exception as exc:failures.append({'source':fs[future],'error':repr(exc)});print(failures[-1],flush=True)
            done+=1;write(a.output/'geometry_status.json',dict(state='running',completed=done,total=len(jobs),failures=failures))
    write(a.output/'geometry_status.json',dict(state='failed' if failures else 'complete',completed=done,total=len(jobs),failures=failures))


if __name__=='__main__':main()
