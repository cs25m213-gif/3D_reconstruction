"""Bounded clip reconstruction pool; admit jobs only with GPU/RAM headroom.

Own worker descendants can overlap. Unrelated GPU processes cause new admissions
to wait; running processes are never paused or signalled by this scheduler.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .utility_contracts import write_json


def completed(folder):
    try:
        report = json.loads((folder / 'reconstruction/run_report.json').read_text())
        return bool(report.get('result_ready') and Path(report.get('mesh', '')).is_file())
    except (OSError, ValueError):
        return False


def is_descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            rows = Path(f'/proc/{pid}/status').read_text().splitlines()
            pid = int(next(r.split(':', 1)[1] for r in rows if r.startswith('PPid:')))
        except (OSError, ValueError, StopIteration):
            return False
    return False


def admission_reason(root_pid, minimum_vram_mib, minimum_ram_mib, allow_busy=False):
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)
    pids = [int(line.strip()) for line in processes.stdout.splitlines() if line.strip()]
    foreign = [pid for pid in pids if not is_descendant(pid, root_pid)]
    if foreign and not allow_busy:
        return f'unrelated GPU process(es): {foreign}'
    gpu = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True)
    free_gpu = int(gpu.stdout.strip())
    if free_gpu < minimum_vram_mib:
        return f'free VRAM {free_gpu} MiB below admission reserve {minimum_vram_mib} MiB'
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    free_ram = int(memory['MemAvailable'].split()[0]) // 1024
    if free_ram < minimum_ram_mib:
        return f'available RAM {free_ram} MiB below admission reserve {minimum_ram_mib} MiB'
    return None


def reconstruction_command(folder, mapping_seconds=90):
    return ['bash', './reconstruct_final.sh', str(folder / 'session'),
            '--output', str(folder / 'reconstruction'), '--fresh', '--semantic-filter', 'off',
            '--da3-model', 'large', '--detail', 'balanced', '--stereo-policy', 'skip',
            '--dense-recovery-views', '0', '--max-views', '36', '--mapping-seconds', str(mapping_seconds)]


def run_pool(folders, jobs, status_file, wait_for_gpu, allow_busy=False,
             minimum_vram_mib=10000, minimum_ram_mib=8000, threads=4,
             probe=admission_reason, launch=subprocess.Popen, poll_seconds=2.):
    folders = [Path(p) for p in folders]
    initially_complete = [p for p in folders if completed(p)]
    pending = [p for p in folders if not completed(p)]
    active, finished, failed = {}, [], []
    last_wait, waiting = 0., None
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VNEXT_THREADS'):
        env[name] = str(threads)
    env['PYTHONUNBUFFERED'] = '1'

    def save(state='running'):
        write_json(Path(status_file), {'state': state, 'scheduler_pid': os.getpid(), 'jobs': jobs,
                   'threads_per_worker': threads, 'total': len(folders),
                   'ready_at_start': len(initially_complete), 'completed_this_run': [p.name for p in finished],
                   'failed_this_run': [p.name for p in failed], 'remaining': [p.name for p in pending],
                   'active': [{'clip': p.name, 'pid': proc.pid} for p, proc, stream in active.values()],
                   'waiting_reason': waiting, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})

    def reap():
        for pid, (folder, process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); del active[pid]
            if code == 0 and completed(folder):
                finished.append(folder); print(f'completed {folder}', flush=True)
            else:
                failed.append(folder); print(f'failed {folder}: exit={code}; see {folder}/reconstruction.log', flush=True)

    save()
    try:
        while pending or active:
            reap()
            while pending and len(active) < jobs:
                try:
                    waiting = probe(os.getpid(), minimum_vram_mib, minimum_ram_mib, allow_busy)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f'Cannot verify resource availability; no new worker admitted: {exc}') from exc
                if waiting:
                    if not wait_for_gpu:
                        raise RuntimeError('Cannot admit reconstruction: ' + waiting)
                    if time.monotonic() - last_wait >= 60 or last_wait == 0:
                        print('Waiting to admit another clip: ' + waiting, flush=True); last_wait = time.monotonic()
                    break
                folder = pending.pop(0)
                # Per-clip advisory lock also prevents a second compatible pool
                # writing the same output directory.
                import fcntl
                lock = (folder / '.reconstruction.lock').open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close(); raise RuntimeError(f'Clip already has a reconstruction writer: {folder}')
                previous = folder / 'reconstruction.log'
                if previous.exists():
                    # Preserve diagnostics from failed/interrupted attempts.
                    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                    previous.rename(folder / f'reconstruction.previous_{stamp}.log')
                    for name in ('run_report.json', 'camera_selection.json'):
                        report = folder / 'reconstruction' / name
                        if report.exists():
                            shutil.copy2(report, folder / f'{name}.previous_{stamp}')
                stream = previous.open('w')
                try:
                    process = launch(reconstruction_command(folder), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     pass_fds=(lock.fileno(),))
                except BaseException:
                    stream.close(); lock.close(); raise
                lock.close()
                active[process.pid] = (folder, process, stream)
                print(f'reconstructing {folder} [pid={process.pid}; slots={len(active)}/{jobs}; threads={threads}]', flush=True)
                save()
            save()
            if pending or active:
                time.sleep(poll_seconds)
        waiting = None
        save('failed' if failed else 'complete')
    except BaseException:
        # Do not leave orphan writers if the scheduler is interrupted or fails.
        # Drain our workers without signalling any process.
        for folder, process, stream in list(active.values()):
            process.wait()
        reap(); save('interrupted')
        raise
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-root', type=Path, default=Path('runs/vnext_clips_v1'))
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--wait-for-gpu', action='store_true')
    parser.add_argument('--allow-busy-gpu', action='store_true')
    parser.add_argument('--minimum-vram-mib', type=int, default=10000)
    parser.add_argument('--minimum-ram-mib', type=int, default=8000)
    parser.add_argument('--status', type=Path, default=Path('runs/vnext_logs/newclips_status.json'))
    args = parser.parse_args()
    if min(args.threads, args.minimum_vram_mib, args.minimum_ram_mib) < 1:
        parser.error('Resource limits must be positive')
    folders = [p for p in sorted(args.clips_root.iterdir()) if not p.is_symlink() and (p / 'session/video.mp4').is_file()]
    raise SystemExit(run_pool(folders, args.jobs, args.status, args.wait_for_gpu, args.allow_busy_gpu,
                             args.minimum_vram_mib, args.minimum_ram_mib, args.threads))


if __name__ == '__main__':
    main()
