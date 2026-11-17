#!/usr/bin/env python3
"""Bounded, resumable external RGB/LiDAR acquisition. Default: show a plan only."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log(message):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def probe(asset):
    request = urllib.request.Request(asset['url'], method='HEAD')
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers.get('Content-Length', -1))
        if size != asset['bytes'] or 'text/html' in response.headers.get('Content-Type', '').lower():
            raise ValueError(f"Publisher response changed for {asset['name']}: size={size}")
        return {'size': size, 'etag': response.headers.get('ETag'), 'url': response.geturl()}


def download(asset, directory, state, limit_mib):
    final = directory / asset['name']
    partial = final.with_name(final.name + '.part')
    if final.exists():
        if final.stat().st_size != asset['bytes']:
            raise ValueError(f'{final} has a wrong size; preserving it for inspection')
        return final
    if partial.exists() and partial.stat().st_size > asset['bytes']:
        raise ValueError(f'{partial} exceeds expected size; preserving it')
    remote = probe(asset)
    metadata = state / (asset['name'] + '.source.json')
    if metadata.exists() and partial.exists():
        old = json.loads(metadata.read_text())
        if old.get('url') != remote['url'] or old.get('etag') != remote['etag']:
            raise ValueError(f"Source identity changed for {asset['name']}; partial preserved")
    write_json(metadata, remote)
    directory.mkdir(parents=True, exist_ok=True)
    if partial.exists() and partial.stat().st_size:
        with partial.open('rb') as stream:
            if stream.read(4) != b'PK\x03\x04':
                raise ValueError(f'{partial} is not a ZIP prefix; preserving it')
    if partial.exists() and partial.stat().st_size == asset['bytes']:
        partial.replace(final)
        return final
    log(f"DOWNLOADING {asset['name']} ({asset['bytes']/1e9:.3f} GB); partial={partial}")
    errors = state / (asset['name'] + '.curl.log')
    command = ['curl', '--fail', '--location', '--continue-at', '-', '--silent', '--show-error',
               '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
               '--limit-rate', str(int(limit_mib * 1024 * 1024)), '--output', str(partial), remote['url']]
    for attempt in range(3):
        with errors.open('a') as stream:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stream)
            try:
                previous = partial.stat().st_size if partial.exists() else 0
                tick = time.monotonic()
                while process.poll() is None:
                    time.sleep(1)
                    if time.monotonic() - tick >= 15:
                        have = partial.stat().st_size if partial.exists() else 0
                        now = time.monotonic(); rate = (have - previous) / max(now - tick, .001)
                        log(f"PROGRESS {asset['name']} {have}/{asset['bytes']} bytes; {rate/1024**2:.2f} MiB/s")
                        previous, tick = have, now
                code = process.returncode
            except BaseException:
                # Only our own curl child is signalled; never another user's process.
                process.terminate(); process.wait()
                raise
        if code == 0:
            break
        # Authentication/access failures and rejected resume ranges need intervention.
        if code in (22, 33, 36):
            raise RuntimeError(f'curl exit {code}; partial preserved; inspect {errors}')
        if attempt < 2:
            log(f'Retrying interrupted transfer, curl exit {code}; partial preserved')
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f'Transfer retries exhausted; partial preserved; inspect {errors}')
    if not partial.exists() or partial.stat().st_size != asset['bytes']:
        raise ValueError(f"Transfer incomplete: {asset['name']}; partial preserved")
    with partial.open('rb') as stream:
        if stream.read(4) != b'PK\x03\x04':
            raise ValueError('Server did not return a ZIP; partial preserved')
    partial.replace(final)
    return final


def safe_destination(root, member):
    name = PurePosixPath(member.filename)
    if name.is_absolute() or '..' in name.parts or '\\' in member.filename or stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError('Unsafe ZIP member: ' + member.filename)
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('ZIP member escapes extraction directory')
    return target


def selected_members(archive, asset, drives):
    members = [m for m in archive.infolist() if not m.is_dir()]
    if asset['kind'] == 'derived_depth':
        members = [m for m in members if any(d in PurePosixPath(m.filename).parts for d in drives)]
        if not members:
            raise ValueError('Depth archive contains none of the pilot drives; no ready marker will be written')
    return members


def file_crc(path):
    value = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def verify_extract(asset, path, destination, state, drives, verify_only=False):
    marker = state / (asset['name'] + '.verified.json')
    current = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    with zipfile.ZipFile(path) as archive:
        members = selected_members(archive, asset, drives)
        targets = [(m, safe_destination(destination, m)) for m in members]
        if marker.exists() and not verify_only:
            saved = json.loads(marker.read_text())
            if (saved.get('archive_identity') == current and saved.get('url') == asset['url']
                    and saved.get('drives') == drives
                    and all(p.is_file() and p.stat().st_size == m.file_size for m, p in targets)):
                log('REUSED ' + asset['name'])
                return saved
        log('VERIFYING full ZIP CRC and computing local SHA-256: ' + asset['name'])
        bad = archive.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC failed: ' + bad)
        required = sum(m.file_size for m, target in targets if not target.exists())
        if not verify_only and shutil.disk_usage(destination).free < required + 2 * 1024**3:
            raise RuntimeError('Insufficient extraction space plus 2 GiB reserve')
        for member, target in targets:
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.file_size or file_crc(target) != member.CRC:
                    raise ValueError(f'Existing extracted file differs: {target}; not overwritten')
                continue
            if verify_only:
                raise ValueError(f'Extracted file missing: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.download-extract-part')
            # Reject an unexpected pre-existing symlink before touching temporary output.
            if temporary.is_symlink():
                raise ValueError('Temporary extraction path is a symlink')
            with archive.open(member) as source, temporary.open('wb') as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            temporary.replace(target)
    result = {'name': asset['name'], 'url': asset['url'], 'archive_identity': current,
              'local_sha256': sha256(path), 'publisher_sha256_available': False,
              'full_zip_crc': 'pass', 'extracted_members': len(members), 'drives': drives,
              'kind': asset['kind'], 'status': 'ready_storage_only'}
    write_json(marker, result)
    log(f"READY {asset['name']}: {len(members)} extracted files")
    return result


def plan(manifest, calibration_only=False, with_depth=False):
    return [a for a in manifest['assets'] if (not calibration_only or a['kind'] == 'calibration')
            and (with_depth or not a['optional_depth'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'download', 'verify'), default='plan')
    parser.add_argument('--root', type=Path, default=Path('/home/vishnu/datasets/student_training_v1'))
    parser.add_argument('--manifest', type=Path, default=HERE / 'student_datasets_manifest.json')
    parser.add_argument('--calibration-only', action='store_true', help='Tiny genuine download/extraction smoke')
    parser.add_argument('--with-depth', action='store_true', help='Also fetch the 14.24 GB publisher-wide derived-depth archive')
    parser.add_argument('--limit-mib', type=float, default=10.)
    parser.add_argument('--max-download-gb', type=float, default=5., help='Ceiling for selected compressed archives, before reuse')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema_version') != 1 or args.limit_mib <= 0 or args.max_download_gb <= 0:
        parser.error('Invalid schema or nonpositive resource limit')
    assets = plan(manifest, args.calibration_only, args.with_depth)
    total = sum(a['bytes'] for a in assets)
    if total > args.max_download_gb * 1e9:
        parser.error(f'Selected {total/1e9:.3f} GB exceeds --max-download-gb; choose an explicit larger ceiling')
    summary = {'scope': manifest['scope'], 'mode': args.mode, 'archives': len(assets),
               'compressed_bytes': total, 'compressed_gb': round(total / 1e9, 6),
               'root': str(args.root), 'limit_mib_s': args.limit_mib, 'assets': assets,
               'automatic_training_import': False}
    print(json.dumps(summary, indent=2), flush=True)
    if args.mode == 'plan':
        return
    if args.mode == 'download' and shutil.which('curl') is None:
        parser.error('curl is required; this script does not install packages')
    try:
        os.nice(max(0, 19 - os.nice(0)))
    except OSError:
        pass
    state = args.root / '_state'; state.mkdir(parents=True, exist_ok=True)
    with (state / 'download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This dataset downloader already has an active writer; no process changed')
        archives = args.root / 'archives/kitti'; extracted = args.root / 'extracted/kitti'
        archives.mkdir(parents=True, exist_ok=True); extracted.mkdir(parents=True, exist_ok=True)
        write_json(state / 'manifest_snapshot.json', manifest)
        write_json(state / 'source_register.json', {'publisher_pages': manifest['publisher_pages'],
                   'license_information': manifest['license_information'], 'integrity_note': manifest['integrity_note'],
                   'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        status = {'state': 'running', 'pid': os.getpid(), 'mode': args.mode, 'completed': [],
                  'compressed_bytes': total, 'scope': manifest['scope']}
        write_json(state / 'status.json', status)
        try:
            missing = sum(max(0, a['bytes'] - ((archives/a['name']).stat().st_size if (archives/a['name']).exists()
                          else (archives/(a['name']+'.part')).stat().st_size if (archives/(a['name']+'.part')).exists() else 0)) for a in assets)
            if args.mode == 'download' and shutil.disk_usage(args.root).free < missing + 2 * 1024**3:
                raise RuntimeError('Insufficient download space plus 2 GiB reserve')
            for asset in assets:
                status.update(current=asset['name']); write_json(state / 'status.json', status)
                path = archives / asset['name']
                if args.mode == 'download':
                    path = download(asset, archives, state, args.limit_mib)
                if not path.is_file() or path.stat().st_size != asset['bytes']:
                    raise ValueError(f'Missing/wrong-size archive: {path}')
                row = verify_extract(asset, path, extracted, state, [d['id'] for d in manifest['drives']], args.mode == 'verify')
                status['completed'].append(row); write_json(state / 'status.json', status)
            status.update(state='complete', storage_verified=True, training_ready=False,
                          detail='Storage acquisition complete; dataset adapter, geometry checks and split audit still required')
            write_json(state / 'status.json', status)
            log('COMPLETE: selected storage assets verified; no training started')
        except BaseException as exc:
            status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            write_json(state / 'status.json', status)
            raise


if __name__ == '__main__':
    main()
