"""Bounded CPU-only bundle adjustment for shared SIMPLE_RADIAL video cameras.

Uses SciPy sparse TRF/LSMR, not CHOLMOD. Fixes one pose and one translation
coordinate to remove the monocular similarity gauge. No metric scale is inferred.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .session_reconstruct import read_model, save_json


def load_observations(folder):
    images=[]
    lines=[line for line in (folder/'images.txt').read_text().splitlines()
           if not line.startswith('#')]
    cursor=0
    while cursor<len(lines):
        if not lines[cursor].strip():
            cursor+=1
            continue
        values=lines[cursor].split()
        cursor+=1
        tokens=lines[cursor].split() if cursor<len(lines) else []
        cursor+=1
        q=np.array(list(map(float,values[1:5])))
        observations=np.asarray(list(map(float,tokens))).reshape(-1,3)
        images.append({'id':int(values[0]),'camera_id':int(values[8]),
                       'name':' '.join(values[9:]),'q':q,
                       't':np.array(list(map(float,values[5:8]))),
                       'observations':observations})
    landmarks=[]
    for line in (folder/'points3D.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        values=line.split()
        landmarks.append({'id':int(values[0]),'xyz':np.array(list(map(float,values[1:4]))),
                          'rgb':list(map(int,values[4:7])),
                          'track':np.array(list(map(int,values[8:]))).reshape(-1,2)})
    return sorted(images,key=lambda p:p['id']),landmarks


def project(camera_parameters, points, camera_indices, point_indices, intrinsics):
    matrices=Rotation.from_rotvec(camera_parameters[:,:3]).as_matrix()
    xyz=np.einsum('nij,nj->ni',matrices[camera_indices],points[point_indices])
    xyz+=camera_parameters[camera_indices,3:]
    # A bounded denominator keeps trial steps finite; cheirality is checked on export.
    depth=np.maximum(xyz[:,2],1e-6)
    uv=xyz[:,:2]/depth[:,None]
    focal,cx,cy,radial=intrinsics
    return focal*uv*(1+radial*np.sum(uv*uv,axis=1)[:,None])+[cx,cy],xyz[:,2]


def refine(source, target, max_points=3000, max_nfev=40, timeout=180,
           min_angle=3., max_error=2.5, refine_intrinsics=True):
    began=time.monotonic()
    if max_points<50 or max_nfev<1 or timeout<=0:
        raise ValueError('Invalid optimization budget')
    if source.resolve()==target.resolve():
        raise ValueError('Refinement must use a separate output directory')
    if (target/'images.txt').exists():
        raise FileExistsError('Refinement output already exists; choose a fresh directory')
    target.mkdir(parents=True,exist_ok=True)
    cameras,_,_=read_model(source)
    if len(cameras)!=1 or cameras[0]['model']!='SIMPLE_RADIAL':
        raise ValueError('Robust backend currently supports one shared SIMPLE_RADIAL camera only')
    camera=cameras[0]
    intrinsics=np.asarray(camera['parameters'],dtype=float)
    images,landmarks=load_observations(source)
    if any(p['camera_id']!=camera['id'] for p in images):
        raise ValueError('All registered views must share the same camera')
    image_lookup={p['id']:i for i,p in enumerate(images)}
    eligible=[p for p in landmarks if sum(int(v) in image_lookup for v,_ in p['track'])>=3]
    if len(eligible)<50 or len(images)<3:
        raise ValueError('Insufficient multi-view observations')
    # Deterministic spread, rather than choosing only long tracks from one scene region.
    chosen=set()
    for image in images:
        visible=[i for i,p in enumerate(eligible) if image['id'] in p['track'][:,0]]
        chosen.update(sorted(visible,key=lambda i:-len(eligible[i]['track']))[:40])
    if len(chosen)>max_points:
        raise ValueError('Point budget cannot provide camera coverage; increase --max-points')
    remaining=[i for i in range(len(eligible)) if i not in chosen]
    needed=min(max_points,len(eligible))-len(chosen)
    if needed:
        chosen.update(remaining[i] for i in np.linspace(0,len(remaining)-1,needed,dtype=int))
    landmarks=[eligible[i] for i in sorted(chosen)]
    cp=np.array([np.r_[Rotation.from_quat(p['q'][[1,2,3,0]]).as_rotvec(),p['t']] for p in images])
    xyz=np.array([p['xyz'] for p in landmarks])
    ci,pi,observed=[],[],[]
    for j,p in enumerate(landmarks):
        track=[(int(v),int(k)) for v,k in p['track'] if int(v) in image_lookup]
        for view,key in track:
            image=images[image_lookup[view]]
            if key>=len(image['observations']) or int(image['observations'][key,2])!=p['id']:
                raise ValueError('Inconsistent COLMAP observation/track references')
            ci.append(image_lookup[view]); pi.append(j)
            observed.append(image['observations'][key,:2])
    ci,pi,observed=np.array(ci),np.array(pi),np.array(observed)
    support=np.bincount(ci,minlength=len(images))
    if support.min()<20:
        raise ValueError('Point budget leaves a camera with fewer than 20 observations; increase --max-points')
    ncam=len(cp)*6
    npoint=xyz.size
    packed=np.r_[cp.ravel(),xyz.ravel(),intrinsics[[0,3]]]
    free=np.ones(len(packed),dtype=bool)
    free[:6]=False
    # Fix a translation component with the largest separation from the fixed pose.
    rotations=Rotation.from_rotvec(cp[:,:3]).as_matrix()
    centers=-np.einsum('nji,nj->ni',rotations,cp[:,3:])
    second=int(np.argmax(np.linalg.norm(centers-centers[0],axis=1)))
    offset=cp[second,3:]-rotations[second]@rotations[0].T@cp[0,3:]
    axis=int(np.argmax(np.abs(offset)))
    if np.linalg.norm(centers[second]-centers[0])<1e-8 or abs(offset[axis])<1e-8:
        raise ValueError('No usable camera baseline to anchor scale')
    free[second*6+3+axis]=False
    if not refine_intrinsics:
        free[-2:]=False
    rows=np.arange(2*len(ci))
    sparsity=lil_matrix((len(rows),len(packed)),dtype=np.int8)
    for k in range(6):
        sparsity[rows,np.repeat(ci*6+k,2)]=1
    for k in range(3):
        sparsity[rows,np.repeat(ncam+pi*3+k,2)]=1
    sparsity[:,-2:]=1
    initial=packed.copy()

    def unpack(values):
        full=initial.copy(); full[free]=values
        calibrated=intrinsics.copy(); calibrated[[0,3]]=full[-2:]
        return full[:ncam].reshape(-1,6),full[ncam:ncam+npoint].reshape(-1,3),calibrated

    def residual(values):
        if time.monotonic()-began>timeout:
            raise TimeoutError('CPU bundle adjustment exceeded its wall-time budget')
        camera_params,points,calibrated=unpack(values)
        uv,_=project(camera_params,points,ci,pi,calibrated)
        return (uv-observed).ravel()

    lower=np.full(len(packed),-np.inf); upper=-lower
    lower[-2:]=[camera['width']*.35,-.3]
    upper[-2:]=[camera['width']*2.,.3]
    start_residual=residual(initial[free]).reshape(-1,2)
    result=least_squares(residual,initial[free],jac_sparsity=sparsity.tocsr()[:,free],
                         bounds=(lower[free],upper[free]),method='trf',tr_solver='lsmr',
                         tr_options={'regularize':True,'maxiter':80},loss='soft_l1',f_scale=1.,
                         x_scale='jac',max_nfev=max_nfev,ftol=1e-4,xtol=1e-5,gtol=1e-5)
    cp,optimized_xyz,intrinsics=unpack(result.x)
    matrices=Rotation.from_rotvec(cp[:,:3]).as_matrix()
    centers=-np.einsum('nji,nj->ni',matrices,cp[:,3:])
    # The optimization budget must not become a spatial-coverage budget. Keep
    # all eligible tracks, triangulating the non-optimized ones with the new poses.
    optimized={p['id']:optimized_xyz[i] for i,p in enumerate(landmarks)}
    optimized_count=len(landmarks)
    landmarks=eligible
    points=[]
    intrinsic_matrix=np.array([[intrinsics[0],0,intrinsics[1]],
                               [0,intrinsics[0],intrinsics[2]],[0,0,1]])
    projection_matrices=np.concatenate([matrices,cp[:,3:,None]],axis=2)
    for p in landmarks:
        if p['id'] in optimized:
            points.append(optimized[p['id']]); continue
        track=[(int(v),int(k)) for v,k in p['track'] if int(v) in image_lookup]
        views=np.array([image_lookup[v] for v,_ in track])
        obs=np.array([images[i]['observations'][k,:2] for i,(_,k) in zip(views,track)])
        normalized=cv2.undistortPoints(obs[:,None,:],intrinsic_matrix,
                                      np.array([intrinsics[3],0.,0.,0.])).reshape(-1,2)
        proj=projection_matrices[views]
        design=np.stack([normalized[:,0,None]*proj[:,2]-proj[:,0],
                         normalized[:,1,None]*proj[:,2]-proj[:,1]],axis=1).reshape(-1,4)
        _,_,vectors=np.linalg.svd(design,full_matrices=False)
        candidate=vectors[-1,:3]/vectors[-1,3] if abs(vectors[-1,3])>1e-12 else p['xyz']
        candidates=np.array([p['xyz'],candidate])
        costs=[]
        for index in range(2):
            uv,depth=project(cp,candidates,views,np.full(len(views),index),intrinsics)
            cost=np.sum(2*(np.sqrt(1+(uv-obs)**2)-1))
            costs.append(cost if np.all(depth>1e-6) and np.isfinite(cost) else np.inf)
        points.append(candidates[int(np.argmin(costs))])
    xyz=np.array(points)
    kept=[]
    output_support=np.zeros(len(images),dtype=int)
    for j,p in enumerate(landmarks):
        track=np.array([(v,k) for v,k in p['track'] if int(v) in image_lookup],dtype=int)
        views=np.array([image_lookup[int(v)] for v,_ in track])
        uv,depth=project(cp,xyz,views,np.full(len(views),j),intrinsics)
        obs=np.array([images[i]['observations'][k,:2] for i,(_,k) in zip(views,track)])
        error=np.linalg.norm(uv-obs,axis=1)
        good=np.isfinite(error)&(error<=max_error)&(depth>1e-6)
        track,views,error=track[good],views[good],error[good]
        if len(track)<3:
            continue
        rays=xyz[j]-centers[views]
        rays/=np.maximum(np.linalg.norm(rays,axis=1)[:,None],1e-12)
        angle=np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1)))
        if angle<min_angle:
            continue
        kept.append(dict(p,xyz=xyz[j],track=track,error=float(error.mean())))
        output_support[views]+=1
    # Remove unsupported cameras, then recursively remove tracks that depend on
    # them. Do not export registered poses supported by only a handful of points.
    active=np.ones(len(images),dtype=bool)
    while True:
        support=np.zeros(len(images),dtype=int)
        supported=[]
        for p in kept:
            track=np.array([(v,k) for v,k in p['track'] if active[image_lookup[int(v)]]],dtype=int).reshape(-1,2)
            if len(track)<3:
                continue
            views=np.array([image_lookup[int(v)] for v,_ in track])
            rays=p['xyz']-centers[views]
            rays/=np.maximum(np.linalg.norm(rays,axis=1)[:,None],1e-12)
            angle=np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1)))
            if angle<min_angle:
                continue
            uv,_=project(cp,np.array([p['xyz']]),views,np.zeros(len(views),dtype=int),intrinsics)
            observed_track=np.array([images[i]['observations'][k,:2] for i,(_,k) in zip(views,track)])
            supported.append(dict(p,track=track,error=float(np.linalg.norm(uv-observed_track,axis=1).mean())))
            support[views]+=1
        new_active=active&(support>=20)
        kept=supported
        if np.array_equal(active,new_active):
            break
        active=new_active
    retained={p['id']:set(map(tuple,p['track'])) for p in kept}
    camera_line=f"{camera['id']} SIMPLE_RADIAL {camera['width']} {camera['height']} "+' '.join(map(str,intrinsics))
    (target/'cameras.txt').write_text(camera_line+'\n')
    lines=[]
    for i,p in enumerate(images):
        if not active[i]:
            continue
        q=Rotation.from_rotvec(cp[i,:3]).as_quat()[[3,0,1,2]]
        lines.append(' '.join(map(str,[p['id'],*q,*cp[i,3:],p['camera_id'],p['name']])))
        tokens=[]
        for key,(x,y,point_id) in enumerate(p['observations']):
            point_id=int(point_id)
            if point_id not in retained or (p['id'],key) not in retained[point_id]:
                point_id=-1
            tokens.extend([str(x),str(y),str(point_id)])
        lines.append(' '.join(tokens))
    (target/'images.txt').write_text('\n'.join(lines)+'\n')
    lines=[' '.join(map(str,[p['id'],*p['xyz'],*p['rgb'],p['error'],*p['track'].ravel()])) for p in kept]
    (target/'points3D.txt').write_text('\n'.join(lines)+'\n')
    end_residual=residual(result.x).reshape(-1,2)
    initial_cost=float(np.sum(2*(np.sqrt(1+start_residual**2)-1)))
    final_cost=float(np.sum(2*(np.sqrt(1+end_residual**2)-1)))
    accepted=bool(result.success and final_cost<initial_cost and len(kept)>=150
                  and active.sum()>=max(5,.7*len(images)))
    report={'backend':'scipy_sparse_trf_lsmr_cpu','accepted':accepted,'solver_success':bool(result.success),
            'solver_message':result.message,'evaluations':result.nfev,'seconds':time.monotonic()-began,
            'input_points':len(eligible),'optimized_points':optimized_count,
            'retriangulated_or_revalidated_points':len(eligible)-optimized_count,'retained_points':len(kept),
            'input_cameras':len(images),'retained_cameras':int(active.sum()),
            'rejected_camera_names':[p['name'] for i,p in enumerate(images) if not active[i]],
            'min_retained_observations_per_camera':int(support[active].min()) if active.any() else 0,
            'initial_robust_cost':initial_cost,'final_robust_cost':final_cost,
            'initial_median_observation_error_px':float(np.median(np.linalg.norm(start_residual,axis=1))),
            'final_median_observation_error_px':float(np.median(np.linalg.norm(end_residual,axis=1))),
            'intrinsics':intrinsics.tolist(),'metric_accuracy_validated':False,'surface_quality_validated':False}
    save_json(target/'optimization_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--max-points',type=int,default=3000)
    parser.add_argument('--max-nfev',type=int,default=40)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--fixed-intrinsics',action='store_true')
    args=parser.parse_args()
    report=refine(args.source,args.output,args.max_points,args.max_nfev,args.timeout,
                  refine_intrinsics=not args.fixed_intrinsics)
    print(report,flush=True)
    if not report['accepted']:
        raise SystemExit('Candidate failed validation; do not use it for dense export')


if __name__=='__main__':
    main()
