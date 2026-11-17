#!/usr/bin/env python3
"""Resume, verify and extract the required manifest; run in tmux for long jobs."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile
import requests
import argparse
import html

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'vnext_downloads'
STATE = DEST / '_state'
EXTRACT = DEST / 'extracted'
CHUNK = 64 * 1024 * 1024
MIRRORS = {}

class QuotaBlocked(RuntimeError):
    pass

def check_response(response, start, end, size):
    if response.status_code == 206 and response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}':
        if 'text/html' not in response.headers.get('Content-Type', '').lower():
            return
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type.lower():
        body = next(response.iter_content(32768), b'').decode(errors='replace')
        text = html.unescape(re.sub('<[^>]+>', ' ', body))
        if any(term in text.lower() for term in ('quota exceeded', 'too many users have viewed', 'download quota')):
            raise QuotaBlocked('Google Drive download quota exceeded; preserve partial data and retry after cooldown')
        title = re.search(r'<title[^>]*>(.*?)</title>', body, re.I | re.S)
        reason = html.unescape(title.group(1))[:160] if title else 'HTML confirmation/access page'
        raise RuntimeError(f'Archive unavailable: {reason}; HTTP {response.status_code}')
    raise RuntimeError(f'host refused range: HTTP {response.status_code}, {response.headers.get("Content-Range")}; {content_type}')

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def record(key, status, detail=''):
    path = STATE / ('setup_' + key.replace('/', '__') + '.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(file=key, status=status, detail=str(detail), updated=time.time()), indent=2))
    tmp.replace(path)
    log(f'{status}: {key} {detail}')

def valid(path, size, sha):
    if not path.is_file() or path.stat().st_size != size:
        return False
    if sha != '-':
        h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda: f.read(8*1024*1024), b''):
                h.update(data)
        if h.hexdigest() != sha:
            return False
    if path.suffix == '.zip' or path.name.endswith('.zip.part'):
        try:
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None
        except (zipfile.BadZipFile, OSError):
            return False
    if path.suffix == '.bag' or path.name.endswith('.bag.part'):
        with path.open('rb') as f:
            return f.read(13) == b'#ROSBAG V2.0\n'
    return True

def source_request(source):
    if source.startswith('gd:'):
        return 'https://drive.usercontent.google.com/download', {'id': source[3:], 'export': 'download', 'confirm': 't'}
    if source.startswith('hf:'):
        repo_revision, name = source[3:].split(':', 1)
        repo, revision = repo_revision.split('@', 1)
        return f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}', {'download': 'true'}
    raise ValueError('Unsupported source scheme')

def download(path, size, source, progress=None):
    part = path.with_name(path.name + '.part')
    url, params = source_request(source)
    chunk = size if source.startswith('hf:') else CHUNK
    if part.exists():
        with part.open('rb') as f:
            prefix = f.read(64).lower()
        if part.stat().st_size > size or b'<!doctype' in prefix or b'<html' in prefix:
            part.unlink()
    have = part.stat().st_size if part.exists() else 0
    if shutil.disk_usage(path.parent).free < size-have+20*10**9:
        raise RuntimeError('insufficient space for remaining download plus 20 GB reserve')
    started, initial, last_log = time.monotonic(), have, 0.
    with requests.Session() as session:
        while have < size:
            end = min(have + chunk, size) - 1
            for attempt in range(4):
                try:
                    with session.get(url, params=params, headers={'Range': f'bytes={have}-{end}'}, stream=True, timeout=(20,120)) as r:
                        # Some hosts ignore Range on a fresh full-file download.
                        # Never append such a response to a resumed partial file.
                        whole = (have == 0 and end == size-1 and r.status_code == 200
                                 and r.headers.get('Content-Length') == str(size)
                                 and 'text/html' not in r.headers.get('Content-Type', '').lower())
                        if not whole:
                            check_response(r, have, end, size)
                        start = have
                        with part.open('ab') as f:
                            for data in r.iter_content(1024*1024):
                                if have + len(data) > end + 1:
                                    raise RuntimeError('response exceeds requested range')
                                f.write(data)
                                have += len(data)
                                now = time.monotonic()
                                if progress and now-last_log >= 15:
                                    progress(have, size, (have-initial)/max(now-started, .001))
                                    last_log = now
                        if have != end + 1:
                            raise RuntimeError(f'incomplete range at {have}')
                    if have == size or have // (1024**3) != start // (1024**3):
                        log(f'PROGRESS: {path.name} {have/1e9:.2f}/{size/1e9:.2f} GB')
                    break
                except QuotaBlocked:
                    raise
                except Exception:
                    have = part.stat().st_size if part.exists() else 0
                    end = min(have + chunk, size) - 1
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))
    if not valid(part, size, '-'):
        part.rename(part.with_name(part.name + '.invalid'))
        raise RuntimeError('archive CRC or ROS bag header check failed')
    part.replace(path)

def extract(path, group, rel, size):
    if path.suffix != '.zip':
        return str(path)
    output = EXTRACT / group / str(Path(rel).with_suffix(''))
    marker = output / '.geostreamfusion_complete.json'
    if marker.is_file():
        info = json.loads(marker.read_text())
        if info.get('archive_size') == size:
            return str(output)
    temp = output.with_name(output.name + '.extracting')
    temp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        needed = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(DEST).free < needed + 20*10**9:
            raise RuntimeError(f'insufficient extraction space; need {needed/1e9:.1f} GB plus 20 GB reserve')
        for item in z.infolist():
            target = (temp / item.filename).resolve()
            if not target.is_relative_to(temp.resolve()):
                raise RuntimeError('unsafe archive path')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('archive contains symlink')
        z.extractall(temp)
    (temp / marker.name).write_text(json.dumps({'archive': str(path), 'archive_size': size, 'completed': time.time()}))
    if output.exists():
        raise RuntimeError(f'unmarked extraction already exists: {output}')
    temp.rename(output)
    return str(output)

def setup(row):
    _, group, rel, byte_count, sha, source = row
    size = int(byte_count)
    key = f'{group}/{rel}'
    try:
        path = DEST / group / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        old = STATE / ('setup_' + key.replace('/', '__') + '.json')
        if old.exists():
            previous = json.loads(old.read_text())
            if previous.get('status') == 'READY':
                previous_path = Path(previous.get('detail', ''))
                archive = path if path.is_file() else ROOT/group/rel
                if archive.is_file() and archive.stat().st_size == size and previous_path.exists():
                    marker = previous_path/'.geostreamfusion_complete.json'
                    if path.suffix != '.zip' or (marker.exists() and json.loads(marker.read_text()).get('archive_size') == size):
                        log(f'REUSE_READY: {key}')
                        return True
        selected = MIRRORS.get(key, {})
        sha = selected.get('sha256', sha)
        source = selected.get('source', source)
        record(key, 'CHECKING')
        # Reuse exact archives in the dataset root without modifying them.
        existing = ROOT / group / rel
        if valid(path, size, sha):
            pass
        elif valid(existing, size, sha):
            path = existing
        else:
            if path.exists():
                path.rename(path.with_name(path.name + '.invalid'))
            record(key, 'DOWNLOADING')
            def progress(have, total, rate):
                record(key, 'DOWNLOADING', f'{have}/{total} bytes; {rate/1e6:.2f} MB/s; source={source.split(":",1)[0]}')
            download(path, size, source, progress)
            if sha != '-' and not valid(path, size, sha):
                raise RuntimeError('SHA256 mismatch')
            path.with_name(path.name+'.source.json').write_text(json.dumps(
                {'source':source, 'expected_size':size, 'sha256':sha,
                 'provenance':selected.get('provenance','manifest'), 'downloaded':time.time()}, indent=2))
        record(key, 'EXTRACTING' if path.suffix == '.zip' else 'VERIFIED')
        location = extract(path, group, rel, size)
        record(key, 'READY', location)
        return True
    except QuotaBlocked as exc:
        record(key, 'BLOCKED_QUOTA', exc)
        return False
    except Exception as exc:
        record(key, 'RETRY_LATER', exc)
        return False

def main():
    global ROOT, DEST, STATE, EXTRACT, MIRRORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--passes', type=int, default=8)
    parser.add_argument('--retry-hours', type=float, default=24)
    parser.add_argument('--only', default='', help='Optional group/path regular expression')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tier', choices=('required','all'), default='required')
    parser.add_argument('--mirrors', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or args.passes < 1 or args.retry_hours <= 0:
        parser.error('jobs, passes and retry-hours must be positive')
    ROOT = args.root.resolve()
    DEST = ROOT / 'vnext_downloads'
    STATE = DEST / '_state'
    EXTRACT = DEST / 'extracted'
    manifest = (args.manifest or ROOT / 'geostreamfusion_download_datasets.sh').read_text().split('\n#__MANIFEST__\n',1)[1]
    rows = [line.split('\t') for line in manifest.splitlines()
            if line.startswith('required\t') or (args.tier=='all' and line.startswith('all\t'))]
    rows = [row for row in rows if not args.only or re.search(args.only, row[1] + '/' + row[2])]
    MIRRORS = json.loads(args.mirrors.read_text()) if args.mirrors else {}
    # Run available alternate hosts before files already known to be quota-blocked.
    rows.sort(key=lambda r: (not MIRRORS.get(r[1]+'/'+r[2],{}).get('source',r[5]).startswith('hf:'),
                             r[1] != 'matrixcity_block3', int(r[3])))
    if args.dry_run:
        print(json.dumps({'files': len(rows), 'manifest_GB': sum(int(r[3]) for r in rows)/1e9,
                          'root': str(ROOT), 'jobs': args.jobs, 'downloads_started': False}, indent=2))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another dataset download/setup is running')
        def summary(pending, status, next_retry=None):
            statuses = {}
            for row in rows:
                key = row[1] + '/' + row[2]
                file = STATE / ('setup_' + key.replace('/', '__') + '.json')
                statuses[key] = json.loads(file.read_text()) if file.exists() else {'status':'PENDING'}
            value = {'status':status, 'updated':time.time(), 'total':len(rows),
                     'ready':sum(s['status']=='READY' for s in statuses.values()),
                     'pending':len(pending), 'files':statuses, 'next_retry_unix':next_retry}
            temp = STATE/'summary.tmp.json'
            temp.write_text(json.dumps(value,indent=2))
            temp.replace(STATE/'summary.json')
        pending = rows
        for cycle in range(args.passes):
            log(f'PASS {cycle+1}: {len(pending)} selected files to check; tier={args.tier}')
            summary(pending, 'running')
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(setup,row):row for row in pending}
                failed = []
                while futures:
                    done, _ = concurrent.futures.wait(futures, timeout=30,
                                     return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        row = futures.pop(future)
                        if not future.result():
                            failed.append(row)
                    summary(failed+list(futures.values()), 'running')
                pending = failed
            summary(pending, 'complete' if not pending else ('waiting' if cycle+1 < args.passes else 'incomplete'),
                    time.time()+args.retry_hours*3600 if pending and cycle+1 < args.passes else None)
            if not pending:
                log('COMPLETE: all selected files downloaded and verified; ZIPs extracted, TARs/bags retained')
                return
            if cycle+1 < args.passes:
                log(f'WAITING: {len(pending)} files blocked; retrying in {args.retry_hours:g} hours')
                time.sleep(args.retry_hours*3600)
        raise SystemExit('INCOMPLETE: retry window exhausted; see setup_*.json')

if __name__ == '__main__':
    main()
