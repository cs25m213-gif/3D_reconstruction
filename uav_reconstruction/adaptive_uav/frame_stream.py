"""Recorded/live frame ingestion using actual decoder timestamps."""
from pathlib import Path
import cv2,numpy as np
from .runtime_types import FramePacket


def video_frames(source,K,max_size=640,tracking_fps=10.,offset=None,live=False,calibration_size=None):
    cap=cv2.VideoCapture(str(source));previous=-float('inf');selected=-float('inf');index=0
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():raise ValueError('Cannot open video: '+str(source))
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if live:
                import time
                stamp=time.monotonic()
            else:
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
                if not np.isfinite(stamp) or (index>0 and stamp<=previous):
                    raise ValueError('Decoder did not expose increasing video PTS; transcode timestamps or use a supported decoder')
            previous=stamp;current=index;index+=1
            if tracking_fps>0 and stamp-selected<1/tracking_fps-1e-6:continue
            selected=stamp;h,w=bgr.shape[:2];scale=min(1.,max_size/max(h,w))
            width=max(16,int(w*scale)//16*16);height=max(16,int(h*scale)//16*16)
            reference_w,reference_h=calibration_size or (w,h)
            pixel=np.diag([width/reference_w,height/reference_h,1.]);k=pixel@K
            rgb=cv2.cvtColor(cv2.resize(bgr,(width,height),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            yield FramePacket(f'frame_{current:06d}.jpg',current,stamp,rgb,k,str(source),
                clock='host_arrival' if live else 'video_pts',pixel_transform=pixel,
                sensor_timestamp=stamp+offset if offset is not None else None)
    finally:cap.release()


def imu_interval(times,values,start,end):
    times=np.asarray(times,float);values=np.asarray(values,float)
    if len(times)<2 or start<times[0] or end>times[-1] or end<=start:return None
    if np.any(np.diff(times)<=0):raise ValueError('IMU timestamps must increase')
    ids=(times>start)&(times<end)
    stamps=np.r_[start,times[ids],end]
    data=np.column_stack([np.interp(stamps,times,values[:,i]) for i in range(values.shape[1])])
    return stamps,data
