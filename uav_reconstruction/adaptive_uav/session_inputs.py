"""Interpret saved recorder files without promoting capability flags to calibration."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SESSION_FILES = ('metadata.json', 'sensors.csv', 'gps.csv',
                 'camera_intrinsics.json', 'stabilization_profile.json')


def input_hashes(session):
    session=Path(session)
    return {name:hashlib.sha256((session/name).read_bytes()).hexdigest()
            for name in SESSION_FILES if (session/name).is_file()}


def _json(path):
    if not path.is_file():return None
    value=json.loads(path.read_text())
    if not isinstance(value,dict):raise ValueError(f'{path.name} must contain a JSON object')
    return value


def _time_summary(stamps):
    stamps=np.asarray(stamps,dtype=np.int64)
    if not len(stamps):return {'samples':0}
    differences=np.diff(stamps)*1e-9
    positive=differences[differences>0]
    return {'samples':len(stamps),'first_elapsed_ns':int(stamps[0]),'last_elapsed_ns':int(stamps[-1]),
            'span_seconds':float((int(stamps[-1])-int(stamps[0]))*1e-9),
            'timestamps_monotonic':bool(np.all(differences>=0)),
            'duplicate_timestamps':int(np.sum(differences==0)),
            'median_rate_hz':float(1/np.median(positive)) if len(positive) else None,
            'max_gap_seconds':float(positive.max()) if len(positive) else None}


def sensor_summary(path):
    if not path.is_file():return {'available':False,'reason':'No sensors.csv'}
    groups={};invalid=0
    with path.open() as stream:
        reader=csv.DictReader(stream)
        required={'elapsed_ns','type','x','y','z'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('sensors.csv needs elapsed_ns,type,x,y,z columns')
        for row in reader:
            try:
                stamp=int(row['elapsed_ns']);kind=int(row['type'])
                values=np.array([float(row[k]) for k in ('x','y','z')])
                if stamp<0 or not np.isfinite(values).all():raise ValueError('Invalid sensor sample')
            except (ValueError,TypeError,KeyError):invalid+=1;continue
            group=groups.setdefault(kind,{'timestamps':[],'name':row.get('sensor',str(kind))})
            group['timestamps'].append(stamp)
    types={str(kind):dict(_time_summary(group['timestamps']),name=group['name']) for kind,group in groups.items()}
    return {'available':True,'valid_samples':sum(x['samples'] for x in types.values()),
            'invalid_rows':invalid,'types':types,
            'rotation_vector_camera_poses_used':False,
            'note':'Logged IMU vectors are not calibrated camera poses; camera/IMU extrinsics and video timing are separate inputs.'}


def gps_summary(path):
    if not path.is_file():return {'available':False,'metric_scale_used':False,'reason':'No gps.csv'}
    stamps=[];coords=[];accuracies=[];altitudes=[];invalid=0
    with path.open() as stream:
        reader=csv.DictReader(stream)
        if not {'elapsed_ns','latitude','longitude','accuracy_m'}.issubset(reader.fieldnames or []):
            raise ValueError('gps.csv needs elapsed_ns,latitude,longitude,accuracy_m columns')
        for row in reader:
            try:
                stamp=int(row['elapsed_ns']);lat=float(row['latitude']);lon=float(row['longitude']);accuracy=float(row['accuracy_m'])
                if not np.isfinite([lat,lon,accuracy]).all() or not -90<=lat<=90 or not -180<=lon<=180 or accuracy<=0:
                    raise ValueError('Invalid GPS fix')
                altitude=float(row.get('altitude_m') or 'nan')
            except (ValueError,TypeError,KeyError):invalid+=1;continue
            stamps.append(stamp);coords.append([lat,lon]);accuracies.append(accuracy)
            if np.isfinite(altitude):altitudes.append(altitude)
    report={'available':True,'invalid_rows':invalid,'metric_scale_used':False,
            'camera_pose_constraints_used':False,**_time_summary(stamps)}
    if coords:
        coordinates=np.asarray(coords);reference=coordinates.mean(0)
        # A short-track diagnostic in metres; it is not GPS-to-SfM alignment.
        xy=np.column_stack(((coordinates[:,1]-reference[1])*111320*np.cos(np.radians(reference[0])),
                            (coordinates[:,0]-reference[0])*111320))
        report.update(unique_coordinates=len(np.unique(coordinates,axis=0)),
            median_horizontal_accuracy_m=float(np.median(accuracies)),
            p90_horizontal_accuracy_m=float(np.quantile(accuracies,.9)),
            horizontal_extent_m=np.ptp(xy,axis=0).tolist(),
            altitude_range_m=float(np.ptp(altitudes)) if altitudes else None)
    report['note']='GPS is reported as recorded, with its accuracy. No RTK/PPK, synchronized camera positions, or metric mesh scale is assumed.'
    return report


def _video_intrinsics(raw, video_width, video_height):
    """Accept explicit video-pixel intrinsics, never an unavailable sensor array."""
    if not raw or raw.get('is_available') is False:return None
    try:
        if all(k in raw for k in ('width','height','fx','fy','cx','cy')):
            values=[float(raw[k]) for k in ('fx','fy','cx','cy')]
            width,height=int(raw['width']),int(raw['height'])
            distortion=list(raw.get('distortion') or [])
        elif raw.get('intrinsics_reference')=='video' and raw.get('intrinsic_calibration'):
            width,height=int(raw['video_width']),int(raw['video_height'])
            values=list(map(float,raw['intrinsic_calibration'][:4]));distortion=[]
            # Android lens_distortion is not silently interpreted as OpenCV.
            if any(raw.get('lens_distortion') or []):return None
        else:return None
        if (width,height)!=(video_width,video_height) or len(values)!=4:return None
        fx,fy,cx,cy=values
        if not np.isfinite(values+distortion).all() or min(fx,fy)<=0 or not 0<cx<width or not 0<cy<height:return None
        if len(distortion) not in (0,4,5,8):return None
        return {'width':width,'height':height,'fx':fx,'fy':fy,'cx':cx,'cy':cy,'distortion':distortion}
    except (TypeError,ValueError,KeyError):return None


def inspect_session(session, reconstruction_width=1280):
    """Return a reproducible input report and minimal SfM configuration changes."""
    from .fast_session import camera_initialization
    session=Path(session);metadata=_json(session/'metadata.json')
    if metadata is None:raise ValueError('metadata.json is required')
    if metadata.get('complete') is False:
        raise ValueError('The session is still recording; finish the recording before saved-session reconstruction')
    before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in session.iterdir() if p.is_file()}
    cap=cv2.VideoCapture(str(session/'video.mp4'))
    width,height=int(cap.get(3)),int(cap.get(4));fps,count=cap.get(5),int(cap.get(7))
    if count>0:
        cap.set(cv2.CAP_PROP_POS_FRAMES,count-1)
        tail_decoded,_=cap.read()
    else:tail_decoded=False
    cap.release()
    if min(width,height,fps,count)<=0:raise ValueError('Invalid or unfinished video.mp4')
    if not tail_decoded:raise ValueError('Cannot decode the final video frame; finish copying/recording this session')
    camera_file=_json(session/'camera_intrinsics.json')
    stabilization_file=_json(session/'stabilization_profile.json')
    stabilization=dict(metadata.get('drone_stabilization') or {})
    if stabilization_file:stabilization.update(stabilization_file)
    eis=stabilization.get('electronic_eis')
    if not isinstance(eis,bool):eis=None
    target_width=round(reconstruction_width*width/max(width,height))
    target_height=round(target_width*height/width)
    seed=camera_initialization(metadata,target_width,target_height)
    calibration=_video_intrinsics(camera_file,width,height)
    exact=bool(calibration and camera_file.get('calibration_exact') is True and eis is not True)
    extended=camera_file is not None or stabilization_file is not None
    refine=not exact
    # Preserve the legacy profile for legacy captures. Extended recorder data
    # must not freeze an unavailable/cropped/EIS focal estimate as exact.
    freeze=exact if extended else True
    overrides={'camera_intrinsics':calibration,'focal_ratio':seed['focal_ratio'],
               'refine_intrinsics':refine,'freeze_guided_intrinsics':freeze,
               'calibration_exact':exact}
    origin=metadata.get('session_start_elapsed_ns')
    explicit_start=metadata.get('video_start_elapsed_ns')
    offset=None
    if origin is not None and explicit_start is not None:
        offset=(int(explicit_start)-int(origin))*1e-9
        if not np.isfinite(offset) or offset<0:raise ValueError('Invalid video_start_elapsed_ns')
    sensors=sensor_summary(session/'sensors.csv');gps=gps_summary(session/'gps.csv')
    notes=[]
    if camera_file and camera_file.get('is_available') is False:
        notes.append('camera_intrinsics.json reports unavailable calibration; metadata focal length is only an initialization estimate.')
    if eis is True:notes.append('EIS is reported. Per-frame crop/warp calibration is absent; shared intrinsics can be refined but remain approximate.')
    if offset is None:notes.append('Video-to-IMU monotonic start offset is missing; gyro comparison uses a diagnostic offset search, not exact synchronization.')
    gyro=sensors.get('types',{}).get('4')
    declared_rate=stabilization.get('imu_rate_hz')
    if gyro and declared_rate==0:notes.append('Declared IMU rate is zero, but timestamped gyroscope samples are present; use the measured rate.')
    video_duration=count/fps
    declared_duration=metadata.get('duration_seconds')
    report={'session':str(session.resolve()),'recording_complete':metadata.get('complete'),
        'input_hashes':input_hashes(session),'video':{'width':width,'height':height,'fps':fps,'frames':count,
            'duration_seconds':video_duration,'declared_session_duration_seconds':declared_duration},
        'camera':{'sidecar_available':camera_file.get('is_available') if camera_file else None,
            'video_intrinsics_used':calibration,'calibration_exact':exact,'initialization':seed,
            'intrinsics_refinement_enabled':refine,'guided_intrinsics_frozen':freeze},
        'stabilization':{'recorded':stabilization,'electronic_eis_reported':eis,
            'per_frame_crop_available':False,'actual_warp_calibration_used':False},
        'timing':{'video_to_session_offset_seconds':offset,'exact_start_timestamp_recorded':offset is not None,
            'camera_imu_extrinsics_available':False,'gyroflow_flag_is_timing_calibration':False},
        'imu':sensors,'gps':gps,'notes':notes}
    after={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in session.iterdir() if p.is_file()}
    if before!=after:raise ValueError('Session files changed during inspection; finish copying/recording and retry')
    report['files_stable_during_inspection']=True
    return report,overrides
