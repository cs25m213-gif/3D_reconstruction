"""Local feature overlap proposals from images, independent of camera poses."""
from pathlib import Path

import cv2
import numpy as np


def image_descriptors(paths,max_features=600):
    sift=cv2.SIFT_create(nfeatures=max_features)
    output=[]
    for path in paths:
        gray=cv2.imread(str(Path(path)),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise FileNotFoundError(path)
        keypoints,descriptors=sift.detectAndCompute(gray,None)
        output.append((keypoints,descriptors))
    return output


def match_overlap(first,second,ratio=.75):
    """Return RANSAC inlier fraction and counts for a pair of cached images."""
    key_a,desc_a=first;key_b,desc_b=second
    if desc_a is None or desc_b is None or min(len(key_a),len(key_b))<8:
        return {'score':0.,'matches':0,'inliers':0}
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a,desc_b,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<ratio*pair[1].distance]
    inliers=0
    if len(good)>=8:
        source=np.float32([key_a[match.queryIdx].pt for match in good])
        target=np.float32([key_b[match.trainIdx].pt for match in good])
        _,mask=cv2.findFundamentalMat(source,target,cv2.FM_RANSAC,1.5,.99)
        if mask is not None:inliers=int(mask.sum())
    return {'score':inliers/max(1,min(len(key_a),len(key_b))),
            'matches':len(good),'inliers':inliers}


def score_history(paths,history=12):
    descriptors=image_descriptors(paths)
    scores={};details={}
    for i in range(1,len(paths)):
        for j in range(max(0,i-history),i):
            result=match_overlap(descriptors[i],descriptors[j])
            scores[(i,j)]=result['score'];details[(i,j)]=result
    return scores,details
