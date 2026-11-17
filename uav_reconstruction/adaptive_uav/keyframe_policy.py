"""Mapping frames are selected independently of the tracking frame rate."""
import cv2,numpy as np


class KeyframePolicy:
    def __init__(self,min_seconds=.25,max_seconds=1.,min_parallax_pixels=8.,minimum_sharpness=15.):
        self.minimum=min_seconds;self.maximum=max_seconds;self.parallax=min_parallax_pixels
        self.sharpness=minimum_sharpness;self.last=None

    def decide(self,frame,pose,queue_load=0.,gyro_motion=None):
        gray=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
        sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        if pose.state!='tracking':return False,{'reason':pose.state,'sharpness':sharp}
        if sharp<self.sharpness:return False,{'reason':'blur','sharpness':sharp}
        if self.last is None:
            self.last=(frame,pose);return True,{'reason':'first_tracked_frame','sharpness':sharp}
        old,oldpose=self.last;dt=frame.timestamp-old.timestamp
        rapid_rotation=bool(gyro_motion and gyro_motion['angular_speed_rad_s']>=.7)
        minimum=self.minimum*(.5 if rapid_rotation else 1.)
        if dt<minimum:return False,{'reason':'minimum_interval','seconds':dt,
                                    'imu_rapid_rotation':rapid_rotation}
        points=cv2.goodFeaturesToTrack(cv2.cvtColor(old.rgb,cv2.COLOR_RGB2GRAY),maxCorners=200,qualityLevel=.01,minDistance=8)
        parallax=0.;tracked=0
        if points is not None:
            moved,status,_=cv2.calcOpticalFlowPyrLK(cv2.cvtColor(old.rgb,cv2.COLOR_RGB2GRAY),gray,points,None)
            if moved is not None:
                good=status.ravel().astype(bool);a=points[good,0];b=moved[good,0];tracked=len(a)
                relative=pose.cam_from_world@oldpose.T_world_from_camera
                rays=np.column_stack((a,np.ones(len(a))))@np.linalg.inv(old.K).T
                rotated=rays@relative[:3,:3].T@frame.K.T
                predicted=rotated[:,:2]/rotated[:,2:3]
                if tracked:parallax=float(np.median(np.linalg.norm(b-predicted,axis=1)))
        translation=float(np.linalg.norm(pose.T_world_from_camera[:3,3]-oldpose.T_world_from_camera[:3,3]))
        stats={'sharpness':sharp,'rotation_compensated_flow_pixels':parallax,'tracked_features':tracked,'translation_visual_units':translation,'seconds':dt,
               'imu_rapid_rotation':rapid_rotation,'gyro_motion':gyro_motion}
        # A heartbeat alone must not insert pure-rotation/stationary observations.
        threshold=self.parallax*(.5 if rapid_rotation else 1.)*(1+queue_load)
        accepted=translation>1e-5 and tracked>=20 and parallax>=threshold
        if accepted:self.last=(frame,pose)
        stats['parallax_threshold_pixels']=threshold
        stats['reason']=('rotation_bridge_with_geometric_content' if rapid_rotation else 'new_geometric_content') if accepted else 'insufficient_translation_or_parallax'
        return accepted,stats

    def propose(self,frame,pose,queue_load=0.,gyro_motion=None):
        """The rule decision with no side effect; pair with commit() for the action taken."""
        last=self.last
        try:return self.decide(frame,pose,queue_load,gyro_motion)
        finally:self.last=last

    def commit(self,frame,pose):
        self.last=(frame,pose)
