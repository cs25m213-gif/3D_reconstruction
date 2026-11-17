"""Queue expansion training after existing coverage is verified and GPU is free."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .prepare_available_selector_data import write


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (FileNotFoundError,ValueError):return False


def link_original_utilities(base, corpus):
    folder=corpus/'utility';folder.mkdir(exist_ok=True)
    count=0
    for namespace,source in (('original',base/'utility'),('expanded',base/'complete_20261115/utility')):
        for archive in sorted(source.glob('*.npz')):
            meta=archive.with_suffix('.json')
            if not meta.is_file():raise ValueError(f'Missing label provenance: {archive}')
            for path in (archive,meta):
                dest=folder/f'{namespace}_{path.name}'
                relative=os.path.relpath(path,folder)
                if dest.is_symlink():
                    if os.readlink(dest)!=relative:raise ValueError('Existing utility link differs')
                elif dest.exists():raise ValueError('Refusing to replace existing union utility')
                else:dest.symlink_to(relative)
            count+=1
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--expansion',type=Path,required=True)
    args=parser.parse_args();base=args.base;root=args.expansion;out=root/'training';corpus=root/'corpus'
    original=base/'complete_20261115';status=root/'training_queue.json';pidfile=out/'training.pid'
    # One durable queue owns this phase; never start a duplicate from this entrypoint.
    import fcntl
    lock=(root/'training_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        old=read(original/'supervisor_status.json')
        admission=read(root/'admission_status.json');prep=read(root/'preparation_queue.json')
        if any(s.get('state')=='failed' for s in (old,admission,prep)):
            write(status,dict(state='failed',reason='An upstream phase failed; inspect preserved logs'));return
        if pidfile.exists():
            pid=pidfile.read_text().strip()
            if alive(pid):
                write(status,dict(state='training',pid=int(pid)));time.sleep(30);continue
            completed=read(out/'complete.json')
            if completed.get('state')=='complete':
                from .train_complete_selector import paths
                groups=paths(corpus,out)
                manifests=sorted(out.glob('epoch_*_manifest.json'))
                last=read(manifests[-1]) if manifests else {}
                missing=set(map(str,groups['train']))-set(last.get('training_files',[]))
                if missing:raise RuntimeError('Expansion final epoch omitted training shards')
                result=dict(state='complete',all_final_training_shards_in_last_epoch=True,
                            training_shards=len(groups['train']),
                            auxiliary_frames=read(corpus/'corpus_seal.json')['unique_training_frames'],
                            scope='Expanded selector training complete; full-video mesh improvement not established')
                write(out/'coverage_verified.json',result);write(status,result);return
            write(status,dict(state='failed',reason='Training exited without completed coverage; no automatic restart'));return
        prerequisites=dict(expansion_prepared=read(root/'prepared.json').get('state')=='complete',
                           original_coverage=read(original/'coverage_verified.json').get('state')=='complete',
                           original_supervisor=old.get('state')=='complete')
        if not all(prerequisites.values()):
            write(status,dict(state='queued',waiting_for=prerequisites));time.sleep(30);continue
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:
            write(status,dict(state='waiting_for_free_GPU',gpu_pids=occupied.splitlines()));time.sleep(30);continue
        count=link_original_utilities(base,corpus)
        import torch
        checkpoint=original/'latest.pt'
        previous=torch.load(checkpoint,map_location='cpu',weights_only=False)
        epochs=int(previous['epoch'])+5
        del previous
        free_mib=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        cache_bytes=read(out/'auxiliary_pack.json')['file_bytes']
        mode='gpu-cache' if cache_bytes+7*1024**3 < free_mib*1024**2 else 'sshfs'
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        command=['.venv/bin/python','-u','-m','uav_reconstruction.adaptive_uav.train_complete_selector',
                 '--base',str(corpus),'--output',str(out),'--warmstart',str(checkpoint),
                 '--epochs',str(epochs),'--batch-size','24','--workers','4','--input-mode',mode]
        write(root/'training_launch.json',dict(command=command,original_utility_links=count,
              warmstart=str(checkpoint),new_manifest_sha256=read(corpus/'corpus_seal.json')['manifest_sha256']))
        with (out/'training.log').open('a') as log:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(process.pid)+'\n');time.sleep(30)


if __name__=='__main__':main()
