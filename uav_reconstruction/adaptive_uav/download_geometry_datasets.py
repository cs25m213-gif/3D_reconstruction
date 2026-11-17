"""Resumable research dataset downloads with a small status file.

Run only for datasets whose licenses permit the intended use. The two archives
from UAVScenes contain posed camera/LiDAR data and 3D maps; Dronescapes supplies
appearance and depth, but no per-frame 3D reconstruction ground truth.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import hf_hub_download, snapshot_download


def status(path,**fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'updated_unix':time.time(),**fields},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('datasets'))
    parser.add_argument('--dataset',choices=('dronescapes','uavscenes','both'),default='both')
    args=parser.parse_args();root=args.root.resolve();state=root/f'{args.dataset}_download_status.json'
    jobs=[]
    if args.dataset in ('dronescapes','both'):
        jobs.append(('Meehai/dronescapes',None,root/'dronescapes_full'))
    if args.dataset in ('uavscenes','both'):
        for name in ('interval5_CAM_LIDAR.zip','terra_3dmap_pointcloud_mesh.zip'):
            jobs.append(('sijieaaa/UAVScenes',name,root/'uavscenes'))
    completed=[]
    for repo,name,destination in jobs:
        destination.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(root).free<70*1024**3:
            status(state,state='stopped_low_disk',completed=completed,remaining_gb=round(shutil.disk_usage(root).free/1024**3,1))
            raise RuntimeError('Less than 70 GiB remains; stopping dataset download')
        for attempt in range(1,9):
            try:
                status(state,state='downloading',repo=repo,file=name,attempt=attempt,completed=completed)
                if name is None:
                    snapshot_download(repo,repo_type='dataset',local_dir=destination,
                        max_workers=6)
                else:
                    hf_hub_download(repo,name,repo_type='dataset',local_dir=destination)
                completed.append({'repo':repo,'file':name,'path':str(destination)})
                break
            except Exception as exc:
                status(state,state='retrying',repo=repo,file=name,attempt=attempt,
                    error=f'{type(exc).__name__}: {str(exc)[:300]}',completed=completed)
                if attempt==8:raise
                time.sleep(min(60*attempt,300))
    status(state,state='complete',completed=completed)


if __name__=='__main__':main()
