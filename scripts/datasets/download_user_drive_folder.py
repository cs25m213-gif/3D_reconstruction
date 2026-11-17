"""Download an explicitly supplied Drive folder through the configured rclone remote."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);os.nice(10)
    def status(state,**fields):
        d={'state':state,'folder_id':a.folder_id,'remote':'gdrive:','destination':str(a.output/'data'),'pid':os.getpid(),'updated_epoch':time.time(),**fields}
        tmp=a.output/'status.tmp.json';tmp.write_text(json.dumps(d,indent=2));tmp.replace(a.output/'status.json')
    common=['--drive-root-folder-id',a.folder_id,'--contimeout','10s','--timeout','2m','--retries','3','--low-level-retries','3','--tpslimit','4']
    with (a.output/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('inventory')
            listing=json.loads(subprocess.check_output(['rclone','lsjson','gdrive:','--recursive','--hash',*common],text=True))
            (a.output/'source_manifest.json').write_text(json.dumps(listing,indent=2))
            files=[r for r in listing if not r['IsDir']];total=sum(max(0,r['Size']) for r in files)
            dest=a.output/'data';dest.mkdir(exist_ok=True)
            command=['rclone','copy','gdrive:',str(dest),'--checksum','--transfers','2','--checkers','4','--stats','30s','--stats-one-line','--log-level','INFO','--log-file',str(a.output/'rclone.log'),*common]
            (a.output/'command.json').write_text(json.dumps(command,indent=2))
            with (a.output/'worker.log').open('a') as log:
                job=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while job.poll() is None:
                    paths=[p for p in dest.rglob('*') if p.is_file()]
                    status('downloading',child_pid=job.pid,files_total=len(files),bytes_total=total,local_bytes=sum(p.stat().st_size for p in paths),local_files=len(paths))
                    time.sleep(10)
                if job.returncode:raise RuntimeError(f'rclone copy failed: exit {job.returncode}; see rclone.log')
            status('verifying',files_total=len(files),bytes_total=total)
            with (a.output/'verification.log').open('a') as log:
                subprocess.run(['rclone','check','gdrive:',str(dest),'--one-way',*common],stdout=log,stderr=subprocess.STDOUT,check=True)
            status('complete',files_total=len(files),bytes_total=total,verification='rclone one-way checksum check passed',training_started=False)
        except Exception as exc:
            status('failed',error=repr(exc));raise

if __name__=='__main__':main()
