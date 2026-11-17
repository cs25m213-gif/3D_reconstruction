"""Isolated CPU remapping from cached matches with a parallax-ranked start pair."""
import json,sqlite3,shutil,time,os
from pathlib import Path
import numpy as np
import cv2,pycolmap

run=Path(Path('/tmp/geostreamfusion_current_test_run').read_text())
source=run/'reconstruction/sparse_run/online'
root=run/os.environ.get('GEOSTREAM_TEST_CAMERA_TRIAL','camera_initialization_trial');folder=root/'online_parallax'
folder.mkdir(parents=True,exist_ok=False)
shutil.copy2(source/'database.db',folder/'database.db');(folder/'images').symlink_to((source/'images').resolve(),target_is_directory=True)
shutil.copy2(source.parent/'pipeline_manifest.json',root/'pipeline_manifest.json')
audit=json.load(open(run/'focal_audit/report.json'));focal=audit['median_curve_best_focal_px']
K=np.array([[focal,0,640],[0,focal,360],[0,0,1.]])
db=sqlite3.connect(f'file:{folder}/database.db?mode=ro',uri=True)
names=dict(db.execute('select image_id,name from images'))
keys={i:np.frombuffer(data,np.float32).reshape(n,c)[:,:2].copy() for i,n,c,data in db.execute('select image_id,rows,cols,data from keypoints')}
rank=[]
for item in audit['pairs_detail']:
    i=next(i for i,n in names.items() if n==item['first']);j=next(i for i,n in names.items() if n==item['second'])
    pair=2147483647*min(i,j)+max(i,j)
    n,data=db.execute('select rows,data from two_view_geometries where pair_id=?',(pair,)).fetchone()
    matches=np.frombuffer(data,np.uint32).reshape(n,2);p1=keys[i][matches[:,0]];p2=keys[j][matches[:,1]]
    E,mask=cv2.findEssentialMat(p1,p2,K,method=cv2.RANSAC,prob=.999,threshold=1.)
    if E is None or E.shape!=(3,3):continue
    count,R,t,mask=cv2.recoverPose(E,p1,p2,K,mask=mask)
    if count<200:continue
    rays1=np.column_stack([(p1[:,0]-640)/focal,(p1[:,1]-360)/focal,np.ones(n)])
    rays2=np.column_stack([(p2[:,0]-640)/focal,(p2[:,1]-360)/focal,np.ones(n)])@R
    rays1/=np.linalg.norm(rays1,axis=1)[:,None];rays2/=np.linalg.norm(rays2,axis=1)[:,None]
    angle=np.degrees(np.arccos(np.clip((rays1*rays2).sum(1),-1,1)))[mask.ravel()>0]
    median=float(np.median(angle))
    if median<2:continue
    rank.append({'i':i,'j':j,'images':[names[i],names[j]],'verified_points':count,'median_bearing_parallax_degrees':median,'score':count*min(median,10)})
db.close()
if not rank:raise RuntimeError('No verified wide-baseline initialization pair')
chosen=max(rank,key=lambda x:x['score']);print('Chosen',chosen,flush=True)
with pycolmap.Database.open(folder/'database.db') as db:
    for camera in db.read_all_cameras():
        db.update_camera(pycolmap.Camera(camera_id=camera.camera_id,model='SIMPLE_PINHOLE',width=camera.width,height=camera.height,params=[focal,640,360]))
options=pycolmap.IncrementalPipelineOptions()
options.num_threads=options.mapper.num_threads=8;options.ba_use_gpu=False
options.multiple_models=False;options.min_model_size=12;options.max_runtime_seconds=180
options.init_num_trials=1
options.mapper.init_min_tri_angle=1.5
options.init_image_id1=chosen['i'];options.init_image_id2=chosen['j']
options.ba_refine_focal_length=True;options.ba_refine_extra_params=False
options.mapper.abs_pose_refine_focal_length=True;options.mapper.abs_pose_refine_extra_params=False
options.ba_local_max_num_iterations=12;options.ba_global_max_num_iterations=25;options.ba_global_max_refinements=2
options.ba_global_frames_ratio=1.6;options.ba_global_points_ratio=1.6
settings={'initial_focal_px':focal,'initialization':chosen,'ranked_candidates':rank,'CPU_only':True,'focal_refinement_enabled':True,'calibration_exact':False,'quality_thresholds_changed':False,'note':'Uses existing matched observations; focal seed derived from a same-video two-view sensitivity audit. No external calibration or metric accuracy is claimed.'}
(root/'settings.json').write_text(json.dumps(settings,indent=2))
start=time.monotonic();out=folder/'blocks/0000/sparse';out.mkdir(parents=True)
models=pycolmap.incremental_mapping(folder/'database.db',folder/'images',out,options=options)
if not models:raise RuntimeError('No camera model recovered')
key,model=max(models.items(),key=lambda t:t[1].num_reg_images());modelpath=out/str(key);modelpath.mkdir(exist_ok=True);model.write(modelpath)
(folder/'current_map.json').write_text(json.dumps({'binary_model':str(modelpath.resolve()),'update_index':0},indent=2))
(folder/'compute_backend.json').write_text(json.dumps({'features':'cached','matching':'cached','bundle_adjustment':{'backend':'CERES','gpu_bundle_adjustment':False}},indent=2))
settings.update(elapsed_seconds=time.monotonic()-start,registered_frames=model.num_reg_images(),points=model.num_points3D(),cameras=[{'model':c.model_name,'params':c.params.tolist()} for c in model.cameras.values()],mean_reprojection_error=model.compute_mean_reprojection_error())
(root/'report.json').write_text(json.dumps(settings,indent=2));print('Result',settings['registered_frames'],settings['cameras'],flush=True)
