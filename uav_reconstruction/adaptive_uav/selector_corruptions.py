"""Deterministic image faults. Pixel geometry stays fixed; no fake camera poses."""
import hashlib
import cv2
import numpy as np

CASES = ('clean','motion_blur','defocus','dark_noise','overexposure','compression','occlusion','low_texture','frame_gap','all_bad')

def frame_seed(seed, clip, index):
    return int.from_bytes(hashlib.sha256(f'{seed}/{clip}/{index}'.encode()).digest()[:8], 'little')

def corrupt(rgb, case, severity, seed):
    if case not in CASES: raise ValueError(case)
    if not 0 <= severity <= 1: raise ValueError('Severity must be in [0, 1]')
    image=np.asarray(rgb).copy(); h,w=image.shape[:2]; mask=np.ones((h,w),bool); rng=np.random.default_rng(seed)
    if case in ('clean','frame_gap') or severity==0:return image,mask
    if case=='motion_blur':
        size=max(3,int(round((.008+.045*severity)*w))|1);kernel=np.zeros((size,size),np.float32)
        angle=rng.uniform(0,np.pi);r=(size-1)/2;dx,dy=np.cos(angle)*r,np.sin(angle)*r
        cv2.line(kernel,(round(r-dx),round(r-dy)),(round(r+dx),round(r+dy)),1.,1)
        kernel/=kernel.sum(); image=cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REFLECT101)
    elif case=='defocus':
        radius=max(1,round((.002+.012*severity)*w));k=np.zeros((2*radius+1,2*radius+1),np.float32)
        cv2.circle(k,(radius,radius),radius,1.,-1);k/=k.sum();image=cv2.filter2D(image,-1,k,borderType=cv2.BORDER_REFLECT101)
    elif case=='dark_noise':
        image=np.clip(image.astype(np.float32)*(1-.85*severity)+rng.normal(0,2+12*severity,image.shape),0,255).astype(np.uint8)
    elif case=='overexposure':image=np.clip(image.astype(np.float32)*(1+2*severity)+40*severity,0,255).astype(np.uint8)
    elif case=='compression':
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,max(5,round(60-55*severity))])
        if not ok:raise ValueError('JPEG encoding failed')
        image=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    elif case=='occlusion':
        side=np.sqrt(.1+.45*severity);rh,rw=max(1,round(h*side)),max(1,round(w*side));y=int(rng.integers(h-rh+1));x=int(rng.integers(w-rw+1))
        image[y:y+rh,x:x+rw]=rng.integers(0,256,3,dtype=np.uint8);mask[y:y+rh,x:x+rw]=False
    elif case=='low_texture':
        image=np.clip((1-.95*severity)*image.astype(np.float32)+.95*severity*image.mean(axis=(0,1)),0,255).astype(np.uint8)
    elif case=='all_bad':image.fill(0);mask.fill(False)
    return image,mask

def quality_vector(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray=cv2.resize(gray,(256,max(8,round(gray.shape[0]*256/gray.shape[1]))),interpolation=cv2.INTER_AREA)
    hist=np.bincount((gray//8).ravel(),minlength=32).astype(float);hist/=hist.sum();hist=hist[hist>0]
    texture=float(gray.var());lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    return np.float32([np.log1p(lap)/12,np.log1p(texture)/10,np.mean(gray<12),np.mean(gray>243),-np.sum(hist*np.log(hist))/np.log(32),np.log1p(lap/max(texture,1))/6])

def usable_rgb(rgb):
    # Exact blank camera failures are unobservable; texture/blur alone is never a hard veto.
    return bool(rgb.size and np.isfinite(rgb).all() and not (np.all(rgb==0) or np.all(rgb==255)))

def frame_fault(case, seed, clip, index, frame_count, severity):
    if case=='all_bad':return case,severity
    if case in ('clean','frame_gap'):return case,severity
    # A contiguous burst plus scattered faults leaves alternative clean frames.
    rng=np.random.default_rng(frame_seed(seed,clip,index)); burst=(frame_count//3<=index<frame_count//3+max(2,frame_count//5))
    return (case,severity) if burst or rng.random()<.35 else ('clean',0.)
