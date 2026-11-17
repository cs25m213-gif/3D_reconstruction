"""Recover geometric evidence for sampled video frames, including skipped frames.

This writes evidence, not action labels. A PnP failure alone is not a verified
relocalization event and must not be treated as one during training.
"""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree


def index(name):
    match=re.search(r'frame_(\d+)',name)
    if match is None:raise ValueError(f'Unknown frame name: {name}')
    return int(match.group(1))


def reference(image,model,image_dir,sift):
    gray=cv2.imread(str(image_dir/image.name),cv2.IMREAD_GRAYSCALE)
    if gray is None:raise FileNotFoundError(image_dir/image.name)
    keypoints,descriptors=sift.detectAndCompute(gray,None)
    observations=[p for p in image.points2D if p.has_point3D() and p.point3D_id in model.points3D]
    if not observations or descriptors is None:
        return None
    distances,indices=cKDTree([p.xy for p in observations]).query([k.pt for k in keypoints])
    lookup={i:observations[j].point3D_id for i,(distance,j) in enumerate(zip(distances,indices)) if distance<=3}
    return {'image':image,'keypoints':keypoints,'descriptors':descriptors,'lookup':lookup}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--sparse',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=15)
    parser.add_argument('--max-candidates',type=int,default=0)
    args=parser.parse_args()
    if args.stride<1:parser.error('--stride must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    model=pycolmap.Reconstruction(args.sparse)
    images=sorted((i for i in model.images.values() if i.has_pose),key=lambda i:index(i.name))
    if not images:raise ValueError('No registered camera poses')
    camera=model.cameras[images[0].camera_id]
    if camera.model_name not in ('PINHOLE','SIMPLE_PINHOLE','SIMPLE_RADIAL'):
        raise ValueError(f'Unsupported camera: {camera.model_name}')
    if any(model.cameras[i.camera_id].camera_id!=camera.camera_id for i in images):
        raise ValueError('This bounded extractor requires a common calibrated camera')
    cap=cv2.VideoCapture(str(args.video))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if total<=0 or fps<=0:raise ValueError('Invalid video metadata')
    indices=list(range(0,total,args.stride))
    if args.max_candidates:indices=indices[:args.max_candidates]
    sift=cv2.SIFT_create(nfeatures=5000)
    matcher=cv2.BFMatcher()
    references={}
    registered_indices=np.array([index(i.name) for i in images])
    k=camera.calibration_matrix().astype(np.float64)
    distortion=np.array([camera.params[-1],0,0,0,0],np.float64) if camera.model_name=='SIMPLE_RADIAL' else None
    frames=out/'frames';frames.mkdir(exist_ok=True)
    rows=[];wanted=set(indices);frame_id=0
    while frame_id<total:
        ok,bgr=cap.read()
        if not ok:break
        if frame_id not in wanted:
            frame_id+=1;continue
        resized=cv2.resize(bgr,(camera.width,camera.height),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        keypoints,descriptors=sift.detectAndCompute(gray,None)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        nearest=np.argsort(abs(registered_indices-frame_id))[:5]
        correspondences={};reference_hits={}
        if descriptors is not None:
            for j in nearest:
                image=images[int(j)]
                if image.name not in references:
                    references[image.name]=reference(image,model,args.images,sift)
                ref=references[image.name]
                if ref is None:continue
                matches=matcher.knnMatch(ref['descriptors'],descriptors,k=2)
                accepted=0
                for pair in matches:
                    if len(pair)<2:continue
                    match,next_match=pair
                    if match.distance>=.75*next_match.distance:continue
                    point_id=ref['lookup'].get(match.queryIdx)
                    if point_id is None:continue
                    accepted+=1
                    current=correspondences.get(point_id)
                    if current is None or match.distance<current[0]:
                        correspondences[point_id]=(match.distance,keypoints[match.trainIdx].pt)
                reference_hits[image.name]=accepted
        pnp_ok=False;inlier_count=0;reprojection=None;pose=None;inlier_ids=[]
        if len(correspondences)>=8:
            ids=list(correspondences)
            xyz=np.asarray([model.points3D[i].xyz for i in ids],np.float32)
            uv=np.asarray([correspondences[i][1] for i in ids],np.float32)
            success,rotation,translation,inliers=cv2.solvePnPRansac(xyz,uv,k,distortion,
                iterationsCount=500,reprojectionError=4.,confidence=.999)
            if success and inliers is not None and len(inliers)>=12:
                rotation_matrix,_=cv2.Rodrigues(rotation)
                projected,_=cv2.projectPoints(xyz[inliers[:,0]],rotation,translation,k,distortion)
                reprojection=float(np.median(np.linalg.norm(projected[:,0]-uv[inliers[:,0]],axis=1)))
                pnp_ok=bool(reprojection<=3.)
                inlier_count=len(inliers)
                if pnp_ok:
                    pose=np.eye(4);pose[:3,:3]=rotation_matrix;pose[:3,3]=translation[:,0]
                    inlier_ids=[int(ids[j]) for j in inliers[:,0]]
        image_path=frames/f'frame_{frame_id:06d}.jpg'
        cv2.imwrite(str(image_path),cv2.resize(bgr,(320,180),interpolation=cv2.INTER_AREA))
        rows.append({'frame_index':frame_id,'time_seconds':frame_id/fps,'image':str(image_path),
            'registered_in_existing_map':bool(frame_id in registered_indices),
            'sharpness':sharpness,'matched_3d_points':len(correspondences),
            'reference_matches':reference_hits,'pnp_verified':pnp_ok,
            'pnp_inliers':inlier_count,'median_reprojection_px':reprojection,
            'world_to_camera':pose.tolist() if pose is not None else None,
            'pnp_inlier_point_ids':inlier_ids})
        if len(rows)%20==0:print(f'candidate evidence {len(rows)}/{len(indices)}',flush=True)
        frame_id+=1
    cap.release()
    report={'source_video':str(args.video.resolve()),'sparse':str(args.sparse.resolve()),
        'camera_model':camera.model_name,'video_frames':total,'candidate_frames':len(rows),
        'existing_registered_frames':len(images),'verified_pnp':sum(r['pnp_verified'] for r in rows),
        'previously_skipped_verified':sum(r['pnp_verified'] and not r['registered_in_existing_map'] for r in rows),
        'note':'Evidence only; no bridge or relocalization action labels inferred from PnP failure.'}
    (out/'candidates.json').write_text(json.dumps({'report':report,'frames':rows},indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
