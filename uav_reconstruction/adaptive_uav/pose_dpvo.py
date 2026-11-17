"""Pinned pretrained DPVO adapter. Imports CUDA dependencies only on selection."""
from pathlib import Path
import hashlib,json
import numpy as np
from .pose_backend import PoseBackend
from .runtime_types import PoseEstimate,PoseCorrection


class DPVOBackend(PoseBackend):
    def __init__(self,root,config):
        import sys
        # Upstream's wheel omits a namespace subdirectory; native extensions are
        # installed, while Python sources are loaded from the pinned checkout.
        sys.path.insert(0,str(Path(root)/'tools/DPVO'))
        import torch
        from dpvo.config import cfg
        if not torch.cuda.is_available():raise RuntimeError('DPVO requires CUDA; use the COLMAP CPU backend')
        self.root=Path(root);self.config=config;self.slam=None;self.frames={};self.known={};self.version=0;self.pending=[]
        self.cfg=cfg.clone();self.cfg.BUFFER_SIZE=int(config.get('pose_buffer',1024))
        self.cfg.PATCHES_PER_FRAME=int(config.get('patches_per_frame',80))
        self.cfg.OPTIMIZATION_WINDOW=int(config.get('optimization_window',12))
        self.cfg.MIXED_PRECISION=bool(config.get('mixed_precision',True))
        self.cfg.LOOP_CLOSURE=bool(config.get('loop_closure',False));self.cfg.CLASSIC_LOOP_CLOSURE=False
        self.checkpoint=self.root/'tools/DPVO/dpvo.pth'
        self.last_ba_error=None
        if not self.checkpoint.is_file():raise RuntimeError('DPVO checkpoint missing; run tools/setup_streaming.py')
        self.provenance={'backend':'dpvo','checkpoint_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            'repository':json.loads((self.root/'tools/DPVO/geostream_provenance.json').read_text()),
            'native_resume':False,'loop_closure':self.cfg.LOOP_CLOSURE,'metric_scale_validated':False}

    def track(self,frame,imu_samples=None):
        import torch
        from dpvo.dpvo import DPVO
        if self.slam is None:
            h,w=frame.rgb.shape[:2];self.slam=DPVO(self.cfg,str(self.checkpoint),ht=h,wd=w,viz=False)
        index=self.slam.counter;self.frames[index]=(frame.frame_id,frame.timestamp)
        # Upstream consumes BGR and reverses it for colour export.
        image=torch.from_numpy(frame.rgb[:,:,::-1].copy()).permute(2,0,1).cuda()
        k=torch.tensor([frame.K[0,0],frame.K[1,1],frame.K[0,2],frame.K[1,2]],dtype=torch.float32,device='cuda')
        # The upstream update catches native BA errors. Record them before that
        # catch so a finite extrapolated pose is not reported as a solved pose.
        from dpvo import fastba
        original_ba=fastba.BA;self.last_ba_error=None
        def checked_ba(*args,**kwargs):
            try:return original_ba(*args,**kwargs)
            except Exception as error:self.last_ba_error=str(error);raise
        fastba.BA=checked_ba
        try:
            with torch.inference_mode():self.slam(frame.timestamp,image,k)
        finally:fastba.BA=original_ba
        self.version+=1;updates=self._snapshot()
        if updates:self.pending.append(PoseCorrection(self.version,updates))
        self.pending=self.pending[-2:]
        if frame.frame_id not in self.known:
            # Upstream can discard stationary initialization frames. Do not call
            # its interpolated trajectory a measured successful pose interval.
            return PoseEstimate(frame.frame_id,frame.timestamp,np.eye(4),'dpvo','initializing',version=self.version)
        pose=self.known[frame.frame_id]
        diagnostics=self._diagnostics()
        good=self.slam.is_initialized and diagnostics['finite_pose'] and self.last_ba_error is None and diagnostics['active_patch_constraints']>=20
        if diagnostics['median_patch_residual_pixels'] is not None:
            good=good and diagnostics['median_patch_residual_pixels']<=8.
        return PoseEstimate(frame.frame_id,frame.timestamp,pose,'dpvo',
            'tracking' if good else 'initializing' if not self.slam.is_initialized else 'degraded',
            version=self.version,diagnostics=diagnostics)

    def _snapshot(self):
        from dpvo.lietorch import SE3
        n=self.slam.n;pg=self.slam.pg
        if not n:return {}
        matrices=SE3(pg.poses_[:n]).inv().matrix().detach().cpu().numpy()
        changes={}
        for i,matrix in enumerate(matrices):
            record=self.frames.get(int(pg.tstamps_[i]))
            if record is None or not np.isfinite(matrix).all():continue
            name=record[0]
            if name in self.known and not np.allclose(matrix,self.known[name],atol=1e-5,rtol=0):changes[name]=matrix.copy()
            self.known[name]=matrix.copy()
        return changes

    def _diagnostics(self):
        import torch
        pg=self.slam.pg;residual=None;raw_residual=None;count=0
        if pg.jj.numel() and self.slam.is_initialized:
            with torch.inference_mode():
                coords=self.slam.reproject()[0,:,:,1,1]
                delta=(coords-pg.target[0]).norm(dim=-1)*self.slam.RES
                latest=torch.isfinite(delta)&(pg.jj>=self.slam.n-2)
                if latest.any():raw_residual=float(delta[latest].median())
                # Out-of-image patches and near-zero-weight correspondences
                # are not geometric constraints in the optimized camera view.
                height,width=self.slam.ht/self.slam.RES,self.slam.wd/self.slam.RES
                target=pg.target[0]
                visible=(coords[:,0]>=0)&(coords[:,0]<width)&(coords[:,1]>=0)&(coords[:,1]<height)
                visible&=(target[:,0]>=0)&(target[:,0]<width)&(target[:,1]>=0)&(target[:,1]<height)
                valid=latest&visible&(pg.weight[0].mean(-1)>.2)
                count=int(valid.sum())
                if count:residual=float(delta[valid].median())
        return {'finite_pose':True,'active_frames':self.slam.n,'active_patch_constraints':count,
            'median_patch_residual_pixels':residual,'residual_is_independent_accuracy':False,
            'raw_unfiltered_patch_residual_pixels':raw_residual,'patch_weight_gate':.2,
            'native_ba_error':self.last_ba_error,
            'camera_imu_fusion_used':False,'calibration_exact':False}

    def get_pose_updates(self,since_version=0):return [c for c in self.pending if c.version>since_version]

    def get_keyframes(self):return [{'frame_id':name,'T_world_from_camera':pose.tolist()} for name,pose in self.known.items()]

    def close(self):
        import torch
        self.slam=None;torch.cuda.empty_cache()
