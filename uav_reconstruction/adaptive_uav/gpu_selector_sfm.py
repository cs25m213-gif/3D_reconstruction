"""GPU SIFT over remote images; RAM-only SQLite; remote CPU mapping handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

REMOTE_BASE='/home/vishnu/datasets/selector_large_20261114'
REMOTE_PROJECT='/home/vishnu/geostreamfusion_attribution_20261108'
REMOTE_PY='/home/vishnu/miniconda3/envs/drone3d/bin/python'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def gpu(plan):
    import pycolmap as pc
    job=json.loads(plan.read_text());base=Path(job['base']).resolve();out=Path(job['output'])
    name=hashlib.sha256(job['key'].encode()).hexdigest()[:16]
    if (out/'sfm'/name).exists():return False
    folder=out/'cpu_gpu_boost'/'work'/name/'sfm'/name
    images=folder/'images';images.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(job['records']):
        if row['role']!='train':raise ValueError('Protected image in training GPU plan')
        path=images/f'{i:09d}.jpg'
        if not path.exists():path.symlink_to(os.path.relpath(base/row['path'],images.resolve()))
    with tempfile.TemporaryDirectory(prefix='gsf_gpu_sift_',dir='/dev/shm') as ram:
        db=Path(ram)/'database.db'
        extraction=pc.FeatureExtractionOptions();extraction.num_threads=4
        extraction.use_gpu=True;extraction.sift.max_num_features=4096
        reader=pc.ImageReaderOptions();reader.camera_model='OPENCV'
        reader.camera_params='268.8,268.8,112,112,0,0,0,0'
        pc.extract_features(db,images,camera_mode=pc.CameraMode.SINGLE,reader_options=reader,
                            extraction_options=extraction,device=pc.Device.cuda)
        matching=pc.FeatureMatchingOptions();matching.num_threads=4;matching.use_gpu=True
        pairing=pc.SequentialPairingOptions();pairing.overlap=20
        pairing.quadratic_overlap=True;pairing.num_threads=4
        pc.match_sequential(db,matching_options=matching,pairing_options=pairing,device=pc.Device.cuda)
        # Back up a closed, complete SQLite database, including any WAL content.
        backup=Path(ram)/'complete.db'
        with sqlite3.connect(db) as source,sqlite3.connect(backup) as target:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('SQLite integrity')
            count=target.execute('SELECT count(*) FROM images').fetchone()[0]
            if count!=len(job['records']):raise RuntimeError('Missing extracted images')
        temporary=folder/'database.uploading';shutil.copyfile(backup,temporary)
        temporary.replace(folder/'database.db')
    write(folder/'gpu_features_ready.json',dict(source=job['key'],images=count,device='CUDA',
        pycolmap=pc.__version__,matching='sequential_overlap20_quadratic',time=time.time()))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpu-child',action='store_true')
    a=p.parse_args()
    if a.gpu_child:gpu(a.plan);return
    job=json.loads(a.plan.read_text());root=Path(job['output'])/'cpu_gpu_boost'
    try:
        write(root/'status.json',dict(state='running',stage='GPU_feature_extraction_and_matching',source=job['key']))
        # Separate child releases its CUDA context before remote CPU mapping.
        subprocess.run([sys.executable,'-u','-m',__spec__.name,'--plan',str(a.plan),'--gpu-child'],check=True)
        name=hashlib.sha256(job['key'].encode()).hexdigest()[:16]
        folder=root/'work'/name/'sfm'/name
        if not (folder/'gpu_features_ready.json').exists():
            write(root/'status.json',dict(state='complete',reason='Original worker already started; preserved'));return
        remote_job=dict(job,base=REMOTE_BASE,output=REMOTE_BASE+'/complete_20261115')
        remote_plan=root/'remote_plan.json';write(remote_plan,remote_job)
        write(root/'status.json',dict(state='running',stage='remote_CPU_mapping',source=job['key'],gpu_stage_complete=True))
        command=f'cd {REMOTE_PROJECT} && GSF_CPU_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH={REMOTE_BASE}/complete_20261115/sfm_deps:{REMOTE_BASE}/deps {REMOTE_PY} -u -m uav_reconstruction.adaptive_uav.boost_selector_gpu_publish --job {REMOTE_BASE}/complete_20261115/cpu_gpu_boost/remote_plan.json'
        subprocess.run(['ssh','-i','/home/iit-t/.ssh/geostreamfusion_transfer_ed25519','-o','BatchMode=yes','vishnu@10.23.20.60',command],check=True)
        result=json.loads(remote_plan.with_suffix('.result.json').read_text())
        write(root/'status.json',dict(state='complete',result=result,gpu_stage_complete=True))
    except Exception as exc:
        write(root/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':main()
