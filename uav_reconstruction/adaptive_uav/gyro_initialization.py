"""Use coarse gyro rotation agreement to rank bounded SfM starting pairs.

This is initialization guidance, not calibrated visual-inertial pose estimation.
Unknown video/sensor offset is searched over the same diagnostic 0--2.5s range
as the existing audit. Final gyro and rendered-depth checks remain independent
stages (though gyro has also influenced initialization).
"""
import csv
import json
import math
from pathlib import Path
import sqlite3
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


def gyro_history(session):
    session=Path(session)
    metadata=session/'metadata.json';sensors=session/'sensors.csv'
    return _gyro_history(str(session.resolve()),metadata.stat().st_mtime_ns,metadata.stat().st_size,
                         sensors.stat().st_mtime_ns,sensors.stat().st_size)


@lru_cache(maxsize=8)
def _gyro_history(session,metadata_mtime,metadata_size,sensors_mtime,sensors_size):
    session=Path(session)
    metadata=json.loads((session/'metadata.json').read_text())
    origin=metadata['session_start_elapsed_ns']*1e-9
    rows=[]
    with (session/'sensors.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['type'])==4:
                rows.append([float(row['elapsed_ns'])*1e-9-origin,*[float(row[k]) for k in ('x','y','z')]])
    if len(rows)<100:raise ValueError('Insufficient gyro samples')
    samples=np.asarray(rows);samples=samples[np.argsort(samples[:,0])]
    samples=samples[np.isfinite(samples).all(axis=1)]
    samples=samples[np.r_[True,np.diff(samples[:,0])>0]]
    if len(samples)<100:raise ValueError('Insufficient finite, distinct gyro samples')
    rot=Rotation.identity();quats=[rot.as_quat()]
    increments=Rotation.from_rotvec(.5*np.diff(samples[:,0])[:,None]*(samples[1:,1:]+samples[:-1,1:]))
    for delta in increments:
        rot=rot*delta;quats.append(rot.as_quat())
    return samples[:,0],Rotation.from_quat(quats)


def rotation_difference(times,rotations,t1,t2,visual_degrees):
    offsets=np.arange(0,2.51,.1)
    a=np.searchsorted(times,t1+offsets).clip(0,len(times)-1)
    b=np.searchsorted(times,t2+offsets).clip(0,len(times)-1)
    valid=(t1+offsets>=times[0])&(t2+offsets<=times[-1])
    if not valid.any():return None
    angles=(rotations[a].inv()*rotations[b]).magnitude()*180/np.pi
    errors=np.abs(angles-visual_degrees);errors[~valid]=np.inf
    i=int(np.argmin(errors))
    return float(errors[i]),float(offsets[i])


def choose_initial_pair(database,session,fps,pycolmap,max_candidates=64,min_angle_deg=4.):
    try:times,rotations=gyro_history(session)
    except (OSError,ValueError,KeyError):return {'used':False,'reason':'No usable timestamped gyro'}
    with sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro',uri=True) as db:
        names=dict(db.execute('SELECT image_id,name FROM images'))
        rows=list(db.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>=100 AND config IN (2,3) ORDER BY rows DESC'))
    candidates=[];evaluated=0
    with pycolmap.Database.open(database) as db:
        images={i.image_id:i for i in db.read_all_images()}
        cameras={c.camera_id:c for c in db.read_all_cameras()}
        keys={}
        for pair,count in rows:
            a,b=divmod(pair,2147483647)
            if a not in names or b not in names:continue
            t1,t2=sorted(int(Path(names[i]).stem.split('_')[-1])/fps for i in (a,b))
            if not .4<=t2-t1<=4.:continue
            evaluated+=1
            g=db.read_two_view_geometry(a,b)
            for i in (a,b):
                if i not in keys:keys[i]=db.read_keypoints(i)[:,:2]
            ok=pycolmap.estimate_two_view_geometry_pose(cameras[images[a].camera_id],keys[a],cameras[images[b].camera_id],keys[b],g)
            if ok and g.cam2_from_cam1 is not None:
                angle=math.degrees(float(g.tri_angle))
                visual=float(Rotation.from_matrix(g.cam2_from_cam1.rotation.matrix()).magnitude()*180/np.pi)
                diagnostic=rotation_difference(times,rotations,t1,t2,visual)
                translation=g.cam2_from_cam1.translation
                forward=abs(float(translation[2]))/max(float(np.linalg.norm(translation)),1e-12)
                if diagnostic and angle>=min_angle_deg and diagnostic[0]<=5 and forward<.95:
                    score=count*min(angle/5.,2.)/(1+diagnostic[0]**2)
                    candidates.append({'image_id1':a,'image_id2':b,'images':[names[a],names[b]],
                        'inliers':count,'triangulation_angle_deg':angle,'visual_rotation_deg':visual,
                        'gyro_rotation_difference_deg':diagnostic[0],'diagnostic_offset_seconds':diagnostic[1],'score':score})
            if evaluated>=max_candidates:break
    return {'used':bool(candidates),'evaluated_pairs':evaluated,'selected':max(candidates,key=lambda r:r['score']) if candidates else None,
            'note':'Coarse initialization guidance; sensor synchronization and camera/IMU extrinsics are not calibrated.'}
