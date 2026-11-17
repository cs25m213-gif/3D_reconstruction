"""Wait for active jobs, preserve them, and verify final label coverage.

The GPU job snapshots labels per epoch. A producer may finish during its final
epoch; in that case completion requires an additional pass including late labels.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

SSH=['ssh','-i','/home/iit-t/.ssh/geostreamfusion_transfer_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=10','vishnu@10.23.20.60']


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def alive(pid):
    try:
        text=Path(f'/proc/{int(pid)}/stat').read_text()
        return text.split(') ',1)[1].split()[0]!='Z'
    except (FileNotFoundError,ValueError):return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.output
    while True:
        state=read(r/'utility_status.json');sfm=read(r/'sfm_status.json')
        # Expected failures of geometry recovery are explicit exclusions, not
        # a reason to invent targets or fail every usable source. Unexpected
        # exceptions remain errors requiring diagnosis.
        if sfm.get('state')=='failed' and not (r/'sfm_admission_review.json').exists():
            expected=('Too few temporal images','No geometrically registered model','No model passed tracked-point/anchor gates')
            failures=sfm.get('failures',[])
            if failures and all(any(s in f['error'] for s in expected) for f in failures):
                sfm.update(state='complete',excluded_sources=failures,failures=[])
                (r/'sfm_admission_review.json').write_text(json.dumps(sfm,indent=2))
                (r/'geometry_status.json').write_text(json.dumps(dict(state='complete',excluded_sources=failures,failures=[])))
                if state.get('state')=='failed' and not (r/'utility_resume_started.json').exists():
                    # The old consumer may have observed the earlier producer
                    # failure. Reuse every finished shard on a normal restart.
                    script='cd /home/vishnu/geostreamfusion_attribution_20261108 && nohup /home/vishnu/miniconda3/envs/drone3d/bin/python -u -m uav_reconstruction.adaptive_uav.label_multisource_selector --output /home/vishnu/datasets/selector_large_20261114/complete_20261115 --student /home/vishnu/geostreamfusion_more_data_20261104/data/student.pt --workers 8 >> /home/vishnu/datasets/selector_large_20261114/complete_20261115/utility_preparation_resume.log 2>&1 < /dev/null &'
                    subprocess.run(SSH+[script],timeout=30,check=True)
                    (r/'utility_resume_started.json').write_text(json.dumps({'started':True}))
                    time.sleep(30);continue
            else:
                (r/'supervisor_status.json').write_text(json.dumps(dict(state='failed',reason='Unexpected SfM exception',failures=failures),indent=2));return
        pidfile=r/'training.pid'
        pid=pidfile.read_text().strip() if pidfile.exists() else '0'
        if alive(pid):
            (r/'supervisor_status.json').write_text(json.dumps(dict(state='watching_active_training',pid=pid,label_state=state.get('state'))))
            time.sleep(30);continue
        if state.get('state')!='complete':
            if state.get('state')=='failed':
                (r/'supervisor_status.json').write_text(json.dumps(dict(state='failed',reason='Gain preparation failed; inspect utility_status.json')));return
            (r/'supervisor_status.json').write_text(json.dumps(dict(state='waiting_for_CPU_preparation',label_state=state.get('state'),geometry_state=read(r/'geometry_status.json').get('state'))))
            time.sleep(30);continue
        extra={name:read(r/name/'status.json') for name in ('cpu_remote_boost','cpu_local_boost','cpu_gpu_boost')}
        if any(v.get('state')=='failed' for v in extra.values()):
            (r/'supervisor_status.json').write_text(json.dumps(dict(state='failed',reason='Additional CPU producer failed',producers=extra),indent=2));return
        waiting={name:v.get('state') for name,v in extra.items() if v and v.get('state')!='complete'}
        if waiting:
            (r/'supervisor_status.json').write_text(json.dumps(dict(state='waiting_for_additional_CPU_producers',producers=waiting)))
            time.sleep(30);continue
        from .train_complete_selector import paths
        expected=set(map(str,paths(a.base,r)['train']))
        manifests=sorted(r.glob('epoch_*_manifest.json'))
        last=read(manifests[-1]) if manifests else {}
        missing=expected-set(last.get('training_files',[]))
        complete=read(r/'complete.json')
        if complete.get('state')=='complete' and not missing:
            result=dict(state='complete',training_shards=len(expected),all_final_training_shards_in_last_epoch=True,
                scope='Admitted reconstruction proxies plus full eligible auxiliary RGB; not full tracked-video mesh validation')
            (r/'coverage_verified.json').write_text(json.dumps(result,indent=2));(r/'supervisor_status.json').write_text(json.dumps(result,indent=2));return
        # Do not restart a failed training loop automatically.
        tail=(r/'training.log').read_text()[-6000:]
        if 'Traceback (most recent call last)' in tail:
            (r/'supervisor_status.json').write_text(json.dumps(dict(state='failed',reason='Training exception; checkpoints preserved')));return
        next_limit=int(last.get('epoch',0))+5
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        with (r/'training.log').open('a') as log:
            proc=subprocess.Popen(['.venv/bin/python','-u','-m','uav_reconstruction.adaptive_uav.train_complete_selector','--base',str(a.base),'--output',str(r),'--epochs',str(next_limit),'--batch-size','24','--workers','4'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(proc.pid)+'\n');time.sleep(30)


if __name__=='__main__':main()
