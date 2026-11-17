"""Use local CPU for extra measured robustness trajectories; write only remotely."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import label_multisource_selector as labels


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2
    raise RuntimeError('No memory availability measurement')


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def publish(source,target):
    if target.exists() and target.with_suffix('.json').exists():return
    # Only this producer owns the local_explore2 namespace. No original
    # remote-label worker sees the private alias geometry or writes these names.
    tmp=target.with_suffix('.local.partial.npz');shutil.copyfile(source,tmp);tmp.replace(target)
    tmp=target.with_suffix('.local.partial.json');shutil.copyfile(source.with_suffix('.json'),tmp);tmp.replace(target.with_suffix('.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--max-episodes',type=int,default=12);a=p.parse_args()
    stage=a.output/'cpu_local_boost';(stage/'geometry').mkdir(parents=True,exist_ok=True);(stage/'utility').mkdir(exist_ok=True)
    labels.init();torch.set_num_threads(a.threads);labels.old.OUT=a.base
    # Read the existing versioned model through the mounted corpus model bank.
    student=a.base/'cpu_label_student.pt'
    if not student.exists():raise RuntimeError('Versioned depth student must be staged remotely before local label work')
    selected=[];done=[];failures=[];domains={};started=time.time()
    while len(selected)<a.max_episodes:
        candidates=[]
        for f in sorted((a.output/'geometry').glob('*.npz')):
            meta=f.with_suffix('.json')
            if not meta.exists():continue
            m=json.loads(meta.read_text());d=m['dataset']
            if m['split']!='train' or d in ('m3ed','uavscenes') or str(f) in selected:continue
            if domains.get(d,0)>=3:continue
            candidates.append((domains.get(d,0),d,f,m))
        if not candidates:
            producer=json.loads((a.output/'geometry_status.json').read_text())
            if producer.get('state') in ('complete','failed'):break
            write(stage/'status.json',dict(state='waiting_for_more_domains',selected=len(selected),done=len(done),domains=domains,threads=a.threads,failures=failures));time.sleep(30);continue
        _,domain,geometry,meta=min(candidates,key=lambda x:(x[0],x[1],str(x[2])))
        selected.append(str(geometry));domains[domain]=domains.get(domain,0)+1
        alias=stage/'geometry'/f'{geometry.stem}_local_explore2.npz'
        if not alias.exists():alias.symlink_to(os.path.relpath(geometry,alias.parent))
        meta=dict(meta,clip=alias.stem,local_exploration_round=2,original_geometry=str(geometry))
        write(alias.with_suffix('.json'),meta)
        for variant in ('motion_blur','mixed','frame_gap'):
            while available_gib()<10:
                write(stage/'status.json',dict(state='waiting_for_RAM',minimum_available_gib=10,available_gib=available_gib(),done=len(done),threads=a.threads));time.sleep(15)
            write(stage/'status.json',dict(state='running',current=alias.stem,variant=variant,done=len(done),domains=domains,threads=a.threads,available_gib=available_gib(),failures=failures))
            try:
                result=Path(labels.collect((str(alias),variant,str(stage),str(student))))
                m=json.loads(result.with_suffix('.json').read_text())
                if 'sfm' in m.get('trust','').lower():m['supervision_weight']=.25
                m['producer']='local_CPU_extra_robustness_exploration';write(result.with_suffix('.json'),m)
                publish(result,a.output/'utility'/result.name);done.append(result.name)
                print(json.dumps(dict(completed=len(done),file=result.name)),flush=True)
            except Exception as exc:failures.append(dict(clip=alias.stem,variant=variant,error=repr(exc)));print(failures[-1],flush=True)
            gc.collect();ctypes.CDLL(None).malloc_trim(0)
    write(stage/'status.json',dict(state='failed' if failures else 'complete',done=len(done),domains=domains,threads=a.threads,failures=failures,elapsed_seconds=time.time()-started))


if __name__=='__main__':main()
