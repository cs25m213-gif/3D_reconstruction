"""Versioned, remote-only expansion of a sealed selector corpus.

Visit all training images in the explicitly discovered sources, retain publisher
and flight splits, and deduplicate resized pixels against the original corpus.
This never edits the sealed corpus or treats unqualified sensors as gain labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile

import cv2
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def role_for_flight(name):
    if name.startswith('AMvalley'):
        return 'validation'
    if name.startswith('HKisland'):
        return 'test'
    # GNSS03 also stays reserved; a new spelling cannot bypass a flight holdout.
    if re.fullmatch(r'(AMtown|HKairport)(?:_GNSS)?03', name):
        return 'holdout'
    if re.fullmatch(r'(AMtown|HKairport)(?:_GNSS)?0[12]', name):
        return 'train'
    if name == 'HKairport_GNSS_Evening':
        return 'train'
    return 'unknown'


def discover(root):
    tasks = []

    def add(path, dataset, kind, role, **extra):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        identity = dict(path=str(path), dataset=dataset, kind=kind, role=role,
                        size=stat.st_size, mtime_ns=stat.st_mtime_ns, **extra)
        identity['id'] = dataset + '_' + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        tasks.append(identity)

    vnext = root / 'vnext_downloads'
    for archive in sorted((vnext / 'uavscenes/interval1_CAM_LIDAR').glob('*.zip')):
        flight = archive.stem.removeprefix('interval1_')
        add(archive, 'uavscenes', 'uav_zip', role_for_flight(flight),
            flight=flight, image_directory='interval1_CAM')
    archive = root / 'uavscenes/interval5_CAM_LIDAR.zip'
    for flight in ('HKairport_GNSS01', 'HKairport_GNSS02',
                   'HKairport_GNSS03', 'HKairport_GNSS_Evening'):
        add(archive, 'uavscenes', 'uav_zip', role_for_flight(flight),
            flight=flight, image_directory='interval5_CAM')
    for role, name in (('train', 'train/block_3.tar'), ('test', 'test/block_3_test.tar')):
        add(vnext / 'matrixcity_block3/small_city/aerial' / name,
            'matrixcity', 'tar', role)
    # Calibration sequences are deliberately not training footage. Both actual
    # stereo observation streams belong to the same eee_03 training flight.
    bag = vnext / 'extracted/ntu_viral/sequences/eee_03/eee_03/eee_03.bag'
    for topic in ('/left/image_raw', '/right/image_raw'):
        add(bag, 'ntu_viral', 'bag', 'train', flight='eee_03', topic=topic)
    add(root / 'dronescapes_experts_subset/manifest.json', 'dronescapes',
        'npz_manifest', 'member_scene_roles')
    return tasks


def image_members(task):
    if task['kind'] == 'uav_zip':
        with zipfile.ZipFile(task['path']) as archive:
            suffix = 'interval' + ('1_' if task['image_directory'] == 'interval1_CAM' else '5_')
            suffix += task['flight'] + '/' + task['image_directory'] + '/'
            return sorted(n for n in archive.namelist()
                          if suffix in n and n.lower().endswith(('.jpg', '.jpeg', '.png')))
    if task['kind'] == 'tar':
        with tarfile.open(task['path']) as archive:
            return sorted(n.name for n in archive if n.isfile()
                          and n.name.lower().endswith(('.jpg', '.jpeg', '.png')))
    return None


def frames(task):
    kind = task['kind']
    if kind in ('uav_zip', 'tar'):
        members = image_members(task)
        opener = zipfile.ZipFile if kind == 'uav_zip' else tarfile.open
        with opener(task['path']) as archive:
            for name in members:
                raw = archive.read(name) if kind == 'uav_zip' else archive.extractfile(name).read()
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode {name}')
                stamp = float(Path(name).stem) if kind == 'uav_zip' else None
                yield name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), stamp, task['role']
    elif kind == 'bag':
        from rosbags.highlevel import AnyReader
        with AnyReader([Path(task['path'])]) as reader:
            connections = [c for c in reader.connections if c.topic == task['topic']]
            if len(connections) != 1:
                raise ValueError(f'Missing/ambiguous stream {task["topic"]}')
            count = 0
            for conn, stamp, data in reader.messages(connections=connections):
                msg = reader.deserialize(data, conn.msgtype)
                if msg.encoding.lower() != 'mono8':
                    raise ValueError(f'Unqualified NTU encoding: {msg.encoding}')
                mono = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
                yield f'{conn.topic}:{count}', cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB), stamp / 1e9, 'train'
                count += 1
            if count != connections[0].msgcount:
                raise ValueError('Incomplete NTU camera decode')
    elif kind == 'npz_manifest':
        path = Path(task['path'])
        for row in json.loads(path.read_text())['pairs']:
            role = 'validation' if row['scene'] == 'slanic' else 'train'
            with np.load(path.parent / row['rgb']) as archive:
                image = archive['arr_0'].copy()
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Unqualified Dronescapes RGB encoding')
            yield row['rgb'], image, None, role
    else:
        raise ValueError(kind)


def admit(job):
    task, output, sealed = job
    output, sealed = Path(output), Path(sealed)
    cv2.setNumThreads(1)
    folder = output / 'sources' / task['id']
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    if marker.exists():
        status = json.loads(marker.read_text())
        if status['identity'] != task:
            raise ValueError('Source identity changed')
        return status
    excluded = {json.loads(line)['sha256_rgb224'] for line in
                (sealed / 'visual_train.jsonl').read_text().splitlines()}
    visited = admitted = duplicates = 0
    counts = {}
    members = image_members(task)
    if members is not None and not members:
        raise ValueError(f'No camera images matched source {task["id"]}')
    if task['role'] not in ('train', 'member_scene_roles'):
        if members is None:
            raise ValueError('Protected source requires an index-only adapter')
        counts[task['role']] = len(members)
        status = dict(state='complete', identity=task, counts=counts, visited_rgb=0,
                      admitted=0, duplicates=0, protected_index_only=True)
        write(marker, status)
        return status
    ledger = folder / 'frames.jsonl.partial'
    began = time.time()
    with ledger.open('w') as stream:
        for number, (key, image, stamp, role) in enumerate(frames(task)):
            visited += 1
            counts[role] = counts.get(role, 0) + 1
            if role != 'train':
                continue
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            digest = hashlib.sha256(resized.tobytes()).hexdigest()
            if digest in excluded:
                duplicates += 1
                continue
            excluded.add(digest)
            filename = folder / f'{number:09d}.jpg'
            if not cv2.imwrite(str(filename), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 93]):
                raise OSError(filename)
            row = dict(path=f'../sources/{task["id"]}/{filename.name}',
                       source=task['id'], dataset=task['dataset'], key=key,
                       timestamp=stamp, role='train', sha256_rgb224=digest,
                       flight=task.get('flight'),
                       supervision='RGB auxiliary only; geometry admitted separately')
            stream.write(json.dumps(row) + '\n')
            admitted += 1
            if visited % 500 == 0:
                write(folder / 'progress.json', dict(state='running', visited_rgb=visited,
                      admitted=admitted, duplicates=duplicates, elapsed_seconds=time.time()-began))
    ledger.replace(folder / 'frames.jsonl')
    status = dict(state='complete', identity=task, counts=counts, visited_rgb=visited,
                  admitted=admitted, duplicates=duplicates, elapsed_seconds=time.time()-began)
    write(marker, status)
    return status


def seal(output, sealed, tasks, baseline_hash):
    source = sealed / 'visual_train.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != baseline_hash:
        raise ValueError('Original sealed corpus changed during expansion')
    corpus = output / 'corpus'
    corpus.mkdir(exist_ok=True)
    seen, counts, added = set(), {}, 0
    temp = corpus / 'visual_train.jsonl.partial'
    with temp.open('w') as stream:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row['role'] != 'train':
                raise ValueError('Protected record in original training manifest')
            row['path'] = '../../' + row['path']
            if row['sha256_rgb224'] in seen:
                continue
            seen.add(row['sha256_rgb224'])
            counts[row['dataset']] = counts.get(row['dataset'], 0) + 1
            stream.write(json.dumps(row) + '\n')
        for task in tasks:
            ledger = output / 'sources' / task['id'] / 'frames.jsonl'
            if not ledger.exists():
                continue
            for line in ledger.read_text().splitlines():
                row = json.loads(line)
                if row['role'] != 'train':
                    raise ValueError('Protected record in expansion training manifest')
                if row['sha256_rgb224'] in seen:
                    continue
                seen.add(row['sha256_rgb224'])
                counts[row['dataset']] = counts.get(row['dataset'], 0) + 1
                stream.write(json.dumps(row) + '\n')
                added += 1
    temp.replace(corpus / 'visual_train.jsonl')
    # Reuse original initialization artifacts. Union utility links are installed
    # only after the original preparation/training supervisor finishes.
    for name in ('visual_final', 'selector'):
        link = corpus / name
        if not link.exists():
            link.symlink_to(Path('../..') / name, target_is_directory=True)
    result = dict(state='complete', unique_training_frames=len(seen), added_training_frames=added,
                  per_dataset=counts, original_manifest_sha256=baseline_hash,
                  manifest_sha256=hashlib.sha256((corpus/'visual_train.jsonl').read_bytes()).hexdigest(),
                  deduplication='exact RGB224 pixel hashes; not semantic/perceptual independence',
                  geometry_scope='auxiliary RGB is not automatically a reconstruction gain target')
    write(corpus / 'corpus_seal.json', result)
    write(corpus / 'preparation_status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sealed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_hash = hashlib.sha256((args.sealed/'visual_train.jsonl').read_bytes()).hexdigest()
    expected = json.loads((args.sealed/'corpus_seal.json').read_text())['manifest_sha256']
    if baseline_hash != expected:
        raise ValueError('Original corpus does not match its seal')
    tasks = discover(args.root)
    manifest = args.output / 'admission_plan.json'
    plan = dict(tasks=tasks, original_manifest_sha256=baseline_hash,
                exclusions=['publisher/area/flight holdouts', 'incomplete .part downloads',
                            'calibration-only sequences', 'semantic label masks as RGB',
                            'unqualified GNSS extrinsics', 'unaligned sensor/expert depth as truth'],
                split_extension='HKairport_GNSS01/02/Evening train; GNSS03 holdout; HKisland remains test')
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Admission plan changed; use a new version')
    write(manifest, plan)
    results, failures = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(admit, (t, str(args.output), str(args.sealed))): t for t in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k != 'identity'} |
                                 {'source': futures[future]['id']}), flush=True)
            except Exception as exc:
                failures.append(dict(source=futures[future]['id'], error=repr(exc)))
                print(json.dumps(failures[-1]), flush=True)
            write(args.output/'admission_status.json', dict(state='running', completed=len(results),
                  total=len(tasks), visited_rgb=sum(r['visited_rgb'] for r in results),
                  added_before_global_dedup=sum(r['admitted'] for r in results), failures=failures))
    if failures:
        write(args.output/'admission_status.json', dict(state='failed', failures=failures))
        raise RuntimeError('Expansion failed; original corpus remains unchanged')
    result = seal(args.output, args.sealed, tasks, baseline_hash)
    write(args.output/'admission_status.json', result | {'sources':results})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
