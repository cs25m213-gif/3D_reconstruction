"""Extra remote CPU workers with isolated scratch and atomic completed-job claims.

Never enters a database used by the original three workers. If an original
worker starts a source while a helper computes it, preserve helper scratch and
let the original finish; do not overwrite its directory or publish duplicate labels.
"""
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def claim_completed_directory(source,target):
    # RENAME_NOREPLACE is required: ordinary rename can replace an empty directory
    # just created by an original worker between its mkdir and marker check.
    libc=ctypes.CDLL(None,use_errno=True)
    function=libc.renameat2
    function.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    function.restype=ctypes.c_int
    result=function(-100,os.fsencode(source),-100,os.fsencode(target),1)
    if result==0:return True
    code=ctypes.get_errno()
    if code in (errno.EEXIST,errno.ENOTEMPTY):return False
    raise OSError(code,os.strerror(code),str(target))


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2
    raise RuntimeError('Cannot measure available RAM')


def worker(plan):
    from . import prepare_selector_sfm_gpu_impl as impl
    job=json.loads(plan.read_text());base=Path(job['base']);output=Path(job['output']);key=job['key']
    name=hashlib.sha256(key.encode()).hexdigest()[:16]
    original=output/'sfm'/name
    if original.exists():return dict(state='skipped_original_already_started',source=key)
    scratch=output/'cpu_gpu_boost'/'work'/name
    (scratch/'geometry').mkdir(parents=True,exist_ok=True)
    try:result=impl.reconstruct((str(base),str(scratch),key,job['records']))
    except impl.GeometryNotAdmitted as exc:
        result=dict(state='complete',source=key,images=len(job['records']),admitted=False,exclusion_reason=str(exc),episodes=[])
        source=scratch/'sfm'/name;source.mkdir(parents=True,exist_ok=True);write(source/'status.json',result)
    source=scratch/'sfm'/name
    if not claim_completed_directory(source,original):
        return dict(state='preserved_helper_scratch_original_started',source=key,scratch=str(scratch))
    published=[]
    for episode in result['episodes']:
        src=Path(episode);dst=output/'geometry'/src.name
        # Hard links publish complete immutable files without data copies or overwrites.
        os.link(src,dst);os.link(src.with_suffix('.json'),dst.with_suffix('.json'));published.append(str(dst))
    result['episodes']=published;result['cpu_helper_threads']=int(os.environ['GSF_CPU_THREADS'])
    write(original/'status.json',result)
    return dict(state='published',source=key,episodes=len(published),admitted=result.get('admitted',True),exclusion_reason=result.get('exclusion_reason'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--threads',type=int,default=8);p.add_argument('--job',type=Path)
    a=p.parse_args()
    if a.job:
        result=worker(a.job);write(a.job.with_suffix('.result.json'),result);print(json.dumps(result),flush=True);return
    root=a.output/'cpu_remote_boost';root.mkdir(exist_ok=True)
    groups={}
    for line in (a.base/'visual_train.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['dataset'] not in ('jb3d','tanks_temples','dronescapes'):continue
        if row['role']!='train':raise ValueError('Protected role in auxiliary training manifest')
        key=row['source']
        if row['dataset']=='dronescapes':key+=':'+Path(row['key']).stem.rsplit('_',1)[0]
        groups.setdefault(key,[]).append(row)
    # Work backwards through the original queue, reducing the chance of catching
    # an original worker processing the same source.
    todo=list(groups);running={};results=[];failures=[]
    while todo or running:
        for key,(process,log,plan) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();del running[key]
            if code==0:results.append(json.loads(plan.with_suffix('.result.json').read_text()))
            else:failures.append(dict(source=key,exit_code=code,log=str(plan.with_suffix('.log'))))
        while todo and len(running)<a.workers:
            if available_gib()-len(running)*4<24:break
            key=todo.pop();name=hashlib.sha256(key.encode()).hexdigest()[:16]
            if (a.output/'sfm'/name).exists():continue
            plan=root/f'{name}.job.json'
            write(plan,dict(base=str(a.base),output=str(a.output),key=key,records=groups[key]))
            env=os.environ.copy();env.update(GSF_CPU_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=plan.with_suffix('.log').open('a')
            process=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--job',str(plan)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[key]=(process,log,plan)
        status=dict(state='running' if todo or running else ('failed' if failures else 'complete'),
            worker_limit=a.workers,threads_per_worker=a.threads,active=[dict(source=k,pid=v[0].pid) for k,v in running.items()],
            queued=len(todo),available_ram_gib=available_gib(),results=results,failures=failures)
        write(root/'status.json',status)
        if todo or running:time.sleep(5)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
