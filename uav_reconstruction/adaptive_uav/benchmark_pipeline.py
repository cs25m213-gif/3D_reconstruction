"""Run real pipeline commands with isolated outputs and explicit timing scope."""
import argparse,json,subprocess,time,shutil,hashlib
from pathlib import Path
from .runtime_metrics import Metrics,save_json


def run_case(root,session,output,options):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'benchmark.json').exists():raise ValueError('Benchmark output already used')
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    sources=list((root/'uav_reconstruction/adaptive_uav').glob('*.py'))+list((root/'uav_reconstruction/configs').glob('stream_*.json'))
    hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        target=snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    command=[str(root/'reconstruct_final.sh'),str(session.resolve()),'--fresh','--output',str(output),*options]
    metrics=Metrics(output/'benchmark_metrics');started=time.monotonic()
    with (output/'pipeline.log').open('w') as log:
        with metrics.scope('full_process'):
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    report={'command':command,'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,
        'source_sha256_at_launch':hashes,
        'source_changed_during_run':[str(p.relative_to(root)) for p in sources if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p.relative_to(root))]],
        'scope':'Fresh cameras, stereo, masks and depth; installed pretrained weights retained',
        'resources':metrics.close()}
    path=output/'run_report.json'
    if path.exists():report['pipeline']=json.loads(path.read_text())
    save_json(output/'benchmark.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions',nargs='+',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=('legacy','preview','balanced','quality'),default='balanced')
    parser.add_argument('--repetitions',type=int,default=1);parser.add_argument('--options',default='[]',help='JSON array of additional CLI strings')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    if args.repetitions<1:parser.error('At least one repetition required')
    options=['--mode',args.mode,*json.loads(args.options)];reports=[]
    for session in args.sessions:
        for index in range(args.repetitions):
            out=args.output/(session.name+f'_{index:02d}')
            report=run_case(root,session,out,options);reports.append({'session':str(session),'repetition':index,**report})
            save_json(args.output/'benchmark_summary.json',reports)
            print(json.dumps({'session':str(session),'repetition':index,'wall_seconds':report['wall_seconds'],'exit_code':report['exit_code']}),flush=True)


if __name__=='__main__':main()
