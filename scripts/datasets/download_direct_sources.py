"""Bounded direct-source acquisition. No Google Drive, training, or data deletion."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,shutil,time,zipfile
from pathlib import Path
import requests

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def verify(path,row):
    if path.stat().st_size!=row['bytes']:raise ValueError('Size mismatch')
    checksum=row.get('checksum')
    algorithm=checksum.split(':')[0] if checksum else 'sha256';h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    if checksum and h.hexdigest()!=checksum.split(':',1)[1]:raise ValueError('Checksum mismatch')
    if row['path'].endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    if row['path'].endswith('.h5'):
        with path.open('rb') as f:
            if f.read(8)!=b'\x89HDF\r\n\x1a\n':raise ValueError('HDF5 header missing')
    return {'algorithm':algorithm,'digest':h.hexdigest(),'publisher_checksum':bool(checksum)}

def fetch(row,root):
    destination=root/row['dataset']/row['path'];destination.parent.mkdir(parents=True,exist_ok=True)
    state=root/'_state'/row['dataset']/(row['path'].replace('/','__')+'.json')
    identity={k:row[k] for k in ('url','bytes','checksum') if k in row}
    def status(value,**extra):atomic(state,{'status':value,'dataset':row['dataset'],'file':str(destination),'identity':identity,'updated_epoch':time.time(),**extra})
    if destination.exists():
        try:
            digest=verify(destination,row);status('ready',verification=digest);return True
        except Exception as exc:status('existing_file_failed_verification',error=str(exc));return False
    part=destination.with_suffix(destination.suffix+'.part');record=part.with_suffix(part.suffix+'.source.json')
    if part.exists() and (not record.exists() or json.loads(record.read_text())!=identity):
        status('partial_identity_conflict');return False
    atomic(record,identity)
    if shutil.disk_usage(root).free<row['bytes']-(part.stat().st_size if part.exists() else 0)+100*1024**3:
        status('waiting_for_disk_space',minimum_free_reserve_gib=100);return False
    for attempt in range(4):
        try:
            have=part.stat().st_size if part.exists() else 0
            if have>row['bytes']:raise ValueError('Partial file exceeds expected size')
            if have<row['bytes']:
                headers={'Range':f'bytes={have}-'} if have else {}
                with requests.get(row['url'],headers=headers,stream=True,timeout=(20,120)) as response:
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('Content-Type','').lower():raise ValueError('HTML instead of dataset')
                    if have:
                        cr=response.headers.get('Content-Range','')
                        if response.status_code!=206 or not cr.startswith(f'bytes {have}-') or not cr.endswith('/'+str(row['bytes'])):raise ValueError('Server did not honor resume range')
                    status('downloading',bytes_received=have,bytes_total=row['bytes']);last=time.monotonic();start=last;initial=have
                    with part.open('ab') as f:
                        for chunk in response.iter_content(2*1024*1024):
                            if have+len(chunk)>row['bytes']:raise ValueError('Response exceeds expected size')
                            f.write(chunk);have+=len(chunk)
                            if time.monotonic()-last>=10:
                                status('downloading',bytes_received=have,bytes_total=row['bytes'],bytes_per_second=(have-initial)/max(1,time.monotonic()-start));last=time.monotonic()
            status('verifying',bytes_received=have,bytes_total=row['bytes']);digest=verify(part,row);part.replace(destination)
            status('ready',bytes_received=row['bytes'],bytes_total=row['bytes'],verification=digest)
            print('READY',row['dataset'],row['path'],flush=True);return True
        except Exception as exc:
            status('retry_wait' if attempt<3 else 'failed',error=repr(exc),attempt=attempt+1,bytes_received=part.stat().st_size if part.exists() else 0,bytes_total=row['bytes'])
            print('RETRY',row['dataset'],row['path'],repr(exc),flush=True)
            if attempt<3:time.sleep(min(60,10*2**attempt))
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--files-per-dataset',type=int,default=1,help='Bounded parallel files per source; each partial file retains one writer')
    a=p.parse_args()
    if a.workers<1 or not 1<=a.files_per_dataset<=4:p.error('workers must be positive; files-per-dataset must be 1..4')
    a.root.mkdir(parents=True,exist_ok=True);os.nice(10)
    with (a.root/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads(a.manifest.read_text());rows=plan['files'];done=0
        atomic(a.root/'status.json',{'state':'running','pid':os.getpid(),'total_files':len(rows),'completed':0,'total_bytes':sum(r['bytes'] for r in rows),'training_started':False})
        groups={name:sorted([r for r in rows if r['dataset']==name],key=lambda x:x['bytes']) for name in sorted({r['dataset'] for r in rows})}
        def dataset_job(items):
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.files_per_dataset) as files:
                return sum(files.map(lambda row:fetch(row,a.root),items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(dataset_job,items) for items in groups.values()]
            while any(not f.done() for f in futures):
                states=[]
                for q in (a.root/'_state').glob('**/*.json'):
                    try:states.append(json.loads(q.read_text()))
                    except (OSError,ValueError):pass
                ready=sum(s['status']=='ready' for s in states)
                atomic(a.root/'status.json',{'state':'running','pid':os.getpid(),'total_files':len(rows),'completed':ready,'active':[s for s in states if s['status'] in ('downloading','verifying','retry_wait')],'failed':[s for s in states if s['status']=='failed'],'total_bytes':sum(r['bytes'] for r in rows),'training_started':False,'updated_epoch':time.time()})
                time.sleep(10)
            done=sum(f.result() for f in futures)
        atomic(a.root/'status.json',{'state':'complete' if done==len(rows) else 'incomplete','completed':done,'total_files':len(rows),'training_started':False,'updated_epoch':time.time()})

if __name__=='__main__':main()
