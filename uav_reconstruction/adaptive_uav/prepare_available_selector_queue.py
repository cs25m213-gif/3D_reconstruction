"""Remote CPU preparation after versioned RGB admission; no GPU or raw transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare_available_selector_data import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion',type=Path,required=True)
    parser.add_argument('--student',type=Path,required=True)
    args=parser.parse_args();root=args.expansion;out=root/'training';out.mkdir(exist_ok=True)
    status=root/'preparation_queue.json'
    import fcntl
    lock=(root/'preparation_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'prepared.json').exists():return
    package=Path(__file__).parent
    write(root/'preparation_source_manifest.json',dict(student=str(args.student),
          student_sha256=hashlib.sha256(args.student.read_bytes()).hexdigest(),
          modules={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in (
              'prepare_available_selector_data.py','prepare_available_selector_geometry.py',
              'prepare_available_selector_queue.py','label_multisource_selector.py',
              'lidar_depth_supervision.py','prepare_multisource_selector.py','selector_input_cache.py')}))
    while True:
        path=root/'admission_status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('state')=='complete':break
        if state.get('state')=='failed':raise RuntimeError('RGB admission failed; preparation not launched')
        write(status,dict(state='waiting_for_RGB_admission'));time.sleep(20)
    write(status,dict(state='packing_remote_RGB'))
    with (root/'packing.log').open('a') as log:
        subprocess.run([sys.executable,'-u','-m','uav_reconstruction.adaptive_uav.selector_input_cache',
                        '--base',str(root/'corpus'),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Install the producer marker before the consumer starts watching.
    write(out/'geometry_status.json',dict(state='running',stage='waiting_for_native_AMtown_producer'))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (root/'geometry.log').open('a') as geometry_log, (root/'labels.log').open('a') as labels_log:
        producer=subprocess.Popen([sys.executable,'-u','-m','uav_reconstruction.adaptive_uav.prepare_available_selector_geometry',
                                   '--expansion',str(root)],env=env,stdout=geometry_log,stderr=subprocess.STDOUT)
        consumer=subprocess.Popen([sys.executable,'-u','-m','uav_reconstruction.adaptive_uav.label_multisource_selector',
                                   '--output',str(out),'--student',str(args.student),'--workers','4'],
                                  env=env,stdout=labels_log,stderr=subprocess.STDOUT)
        write(status,dict(state='preparing_native_geometry_and_measured_gains',geometry_pid=producer.pid,
                          label_pid=consumer.pid,variants=['clean','motion_blur','defocus','mixed','frame_gap']))
        producer_code=producer.wait()
        if producer_code:
            write(out/'geometry_status.json',dict(state='failed',producer_exit_code=producer_code))
        consumer_code=consumer.wait()
    if producer_code or consumer_code:
        write(status,dict(state='failed',geometry_exit_code=producer_code,label_exit_code=consumer_code))
        raise RuntimeError('Expansion preparation failed; existing corpus/checkpoints are unchanged')
    from .prepare_large_selector_labels import VARIANTS
    def missing():
        return [str(path)+'__'+variant for path in (out/'geometry').glob('*.npz')
                for variant in VARIANTS if not all((out/'utility'/f'{path.stem}__{variant}{suffix}').is_file()
                                                  for suffix in ('.npz','.json'))]
    # A producer can publish its final episode just after the consumer scans.
    # Reconcile against the now-stable producer before claiming full coverage.
    if missing():
        with (root/'labels.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','uav_reconstruction.adaptive_uav.label_multisource_selector',
                            '--output',str(out),'--student',str(args.student),'--workers','4'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if missing():raise RuntimeError('Some native-rate geometry variants still lack measured gains')
    utility=json.loads((out/'utility_status.json').read_text())
    if utility.get('state')!='complete':raise RuntimeError('Gain label coverage not complete')
    result=dict(state='complete',utility_shards=utility['completed'],
                scope='All new eligible RGB packed; native AMtown measured-gain labels; other new RGB auxiliary only')
    write(root/'prepared.json',result);write(status,result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--expansion' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--expansion')+1])
            # Failure to acquire a queue lock belongs to the duplicate invocation;
            # do not overwrite the live owner's status.
            if not isinstance(exc,BlockingIOError):
                write(root/'preparation_queue.json',dict(state='failed',error=repr(exc)))
        raise
