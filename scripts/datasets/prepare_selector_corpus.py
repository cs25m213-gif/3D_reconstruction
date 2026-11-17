"""Remote-only RGB admission for large-selector visual pretraining.

Reads every RGB frame of complete supported sources. Does not invent geometry,
selector rewards or camera motion. Original validation/test roles are preserved.
Writes bounded-resolution JPEGs and per-source ledgers; raw sources stay unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(p)


def flight_role(name):
    if 'AMvalley' in name:
        return 'validation'
    if 'HKisland' in name:
        return 'test'
    if re.search(r'(AMtown|HKairport)03', name):
        return 'holdout'
    if re.search(r'(AMtown|HKairport)0[12]', name):
        return 'train'
    return 'unknown'


def decode_message(msg):
    if hasattr(msg, 'format'):
        bgr = cv2.imdecode(np.asarray(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Undecodable compressed image')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, '8uc1': 1, 'bayer_rggb8': 1,
                'bayer_bggr8': 1, 'bayer_gbrg8': 1, 'bayer_grbg8': 1}
    if enc not in channels:
        raise ValueError(f'Unsupported ROS RGB encoding: {enc}')
    image = data[:, :msg.width * channels[enc]].reshape(msg.height, msg.width, channels[enc])
    if enc == 'rgb8':
        return image
    if enc == 'bgr8':
        return image[:, :, ::-1]
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    code = {'bayer_rggb8': cv2.COLOR_BayerRG2RGB, 'bayer_bggr8': cv2.COLOR_BayerBG2RGB,
            'bayer_gbrg8': cv2.COLOR_BayerGB2RGB, 'bayer_grbg8': cv2.COLOR_BayerGR2RGB}[enc]
    return cv2.cvtColor(image[:, :, 0], code)


def frames(task):
    p, kind = Path(task['path']), task['kind']
    if kind == 'h5':
        import h5py
        with h5py.File(p) as f:
            images, ts = f['ovc/rgb/data'], f['ovc/ts']
            for i in range(len(images)):
                yield f'{i}', images[i], float(ts[i]), task['role']
    elif kind == 'bag':
        from rosbags.highlevel import AnyReader
        with AnyReader([p]) as reader:
            available = [c for c in reader.connections if c.msgtype.endswith('/CompressedImage') or c.msgtype.endswith('/Image')]
            color = [c for c in available if not any(k in c.topic.lower() for k in ('depth', 'infra', 'left', 'right'))]
            if not color:
                # UAVScenes names its sole RGB stream /left_camera/image/compressed.
                compressed = [c for c in available if c.msgtype.endswith('/CompressedImage')]
                if len(compressed) == 1:
                    color = compressed
                else:
                    raise ValueError(f'No unambiguous RGB stream: {[c.topic for c in available]}')
            # Explicit preference for RGB/color/image_raw/compressed; one camera stream per source.
            c = max(color, key=lambda x: ('rgb' in x.topic.lower(), 'color' in x.topic.lower(), x.msgcount))
            for i, (conn, stamp, data) in enumerate(reader.messages(connections=[c])):
                msg = reader.deserialize(data, conn.msgtype)
                yield f'{c.topic}:{i}', decode_message(msg), stamp / 1e9, task['role']
    elif kind == 'zip':
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            normalized = {n.replace('\\', '/'): n for n in names}
            publisher_train = set()
            for n in names:
                if n.replace('\\', '/').endswith('/train_list.txt'):
                    publisher_train.update(Path(s.strip().replace('\\', '/')).name for s in z.read(n).decode().splitlines() if s.strip())
            for n in sorted(names):
                key = n.replace('\\', '/')
                if not key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                role = task['role']
                if task['dataset'] == 'uavscenes':
                    role = flight_role(key)
                    if '/interval5_CAM/' not in key:
                        continue
                if task['dataset'] == 'uavlight' and publisher_train and Path(key).name not in publisher_train:
                    role = 'publisher_holdout'
                image = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Undecodable member {n}')
                yield key, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, role
            videos = [n for n in names if n.lower().endswith(('.mp4', '.avi', '.mov'))]
            for n in videos:
                # ZIP videos require seeking. The temporary copy stays on the remote data disk.
                with tempfile.TemporaryDirectory(dir=task['temp']) as temp:
                    file = Path(temp) / ('video' + Path(n).suffix)
                    with z.open(n) as inp, file.open('wb') as out:
                        shutil.copyfileobj(inp, out, 8*1024**2)
                    yield from video_frames(file, task['role'], prefix=n)
    elif kind == 'npz_manifest':
        meta=json.loads(p.read_text())
        for row in meta['pairs']:
            role='validation' if row['scene']=='slanic' else 'train'
            with np.load(p.parent/row['rgb']) as d:
                image=d['arr_0']
            yield row['rgb'], image, None, role
    elif kind == 'video':
        yield from video_frames(p, task['role'])
    else:
        raise ValueError(kind)


def video_frames(path, role, prefix='video'):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield f'{prefix}:{i}', cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cap.get(cv2.CAP_PROP_POS_MSEC)/1000., role
            i += 1
    finally:
        cap.release()
    if expected > 0 and abs(i-expected) > 2:
        raise ValueError(f'Incomplete video decode: expected {expected}, got {i}')


def prepare(task):
    cv2.setNumThreads(1)
    folder = Path(task['output']) / 'sources' / task['id']
    marker = folder / 'complete.json'
    identity = {k: task[k] for k in ('path', 'size', 'mtime_ns', 'role', 'kind')}
    if marker.exists():
        result = json.loads(marker.read_text())
        if result['identity'] != identity:
            raise ValueError('Source changed; use a new experiment root')
        return result
    folder.mkdir(parents=True, exist_ok=True)
    counts, unique = {}, set()
    index = folder / 'frames.jsonl.partial'
    began = time.time()
    try:
        with index.open('w') as ledger:
            for number, (key, rgb, stamp, role) in enumerate(frames(task)):
                counts[role] = counts.get(role, 0) + 1
                # Preserve holdout source inventory without preparing it for training.
                if role != 'train':
                    continue
                resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                digest = hashlib.sha256(resized.tobytes()).hexdigest()
                if digest in unique:
                    continue
                unique.add(digest)
                rel = Path('sources') / task['id'] / f'{number:09d}.jpg'
                file = Path(task['output']) / rel
                ok = cv2.imwrite(str(file), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
                if not ok:
                    raise OSError(file)
                ledger.write(json.dumps({'path': str(rel), 'source': task['id'], 'dataset': task['dataset'],
                                         'key': key, 'timestamp': stamp, 'role': role, 'sha256_rgb224': digest}) + '\n')
        index.replace(folder / 'frames.jsonl')
        result = {'identity': identity, 'dataset': task['dataset'], 'id': task['id'],
                  'counts_raw_rgb': counts, 'unique_training_frames': len(unique),
                  'seconds': time.time()-began, 'status': 'complete'}
        write(marker, result)
        return result
    except Exception as e:
        result = {'identity': identity, 'id': task['id'], 'dataset': task['dataset'],
                  'status': 'failed', 'error': repr(e), 'counts_raw_rgb': counts}
        write(folder / 'failure.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/vishnu/datasets'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'temp').mkdir(exist_ok=True)
    splits = json.loads((a.root / 'geostreamfusion_prepared_20261107/uavlight_v1/splits.json').read_text())
    tasks, seen = [], set()

    def add(path, dataset, kind, role):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        stat = path.stat()
        tasks.append({'path': str(path), 'dataset': dataset, 'kind': kind, 'role': role,
                      'id': dataset + '_' + hashlib.sha256(str(path).encode()).hexdigest()[:16],
                      'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                      'output': str(a.output), 'temp': str(a.output / 'temp')})

    for path in sorted((a.root/'geostreamfusion_direct_20261107/m3ed').rglob('*_data.h5')):
        role = 'validation' if 'fast_flight_2' in path.name else ('test' if 'forest_road' in path.name else 'train')
        add(path, 'm3ed', 'h5', role)
    for path in sorted((a.root/'geostreamfusion_direct_20261107/uavlight').rglob('*.zip')):
        add(path, 'uavlight', 'zip', splits[path.stem.split('-')[0]])
    for path in sorted((a.root/'geostreamfusion_direct_20261107/jb3d').glob('*.zip')):
        add(path, 'jb3d', 'zip', 'train')
    # Prefer the largest downloaded copy of a flight; small/truncated copies are inventoried separately.
    bags = {}
    for path in list(a.root.glob('*.bag')) + list((a.root/'drive_imports').rglob('*.bag')):
        if path.name not in bags or path.stat().st_size > bags[path.name].stat().st_size:
            bags[path.name] = path
    for path in bags.values():
        add(path, 'uavscenes', 'bag', flight_role(path.name))
    path = a.root/'uavscenes/interval5_CAM_LIDAR.zip'
    if path.exists():
        add(path, 'uavscenes', 'zip', 'member_flight_roles')
    for path in sorted((a.root/'tanks_temples_user_drive_20261107').rglob('*.zip')):
        add(path, 'tanks_temples', 'zip', 'train')
    # Barn.mp4 overlaps Barn.zip; include all distinct frames; hashes deduplicate prepared RGB across sources.
    for path in sorted((a.root/'tanks_temples_user_drive_20261107').rglob('*.mp4')):
        add(path, 'tanks_temples', 'video', 'train')
    for path in (a.root/'student_training_v1/archives/kitti').glob('*_sync.zip'):
        add(path,'kitti','zip','validation' if '2011_09_30_drive_0016' in path.name else 'train')
    path=a.root/'dronescapes_rgb_selector/manifest.json'
    if path.exists():
        add(path,'dronescapes','npz_manifest','member_scene_roles')
    inventory = []
    roots = [a.root/'geostreamfusion_direct_20261107', a.root/'drive_imports', a.root/'uavscenes',
             a.root/'tanks_temples_user_drive_20261107',a.root/'student_training_v1/archives',
             a.root/'dronescapes_rgb_selector']
    source_paths = {t['path'] for t in tasks}
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            included = str(path.resolve()) in source_paths
            reason = 'RGB adapter scheduled' if included else ('incomplete_download' if path.name.endswith('.part') else
                      'geometry_or_sensor_or_metadata_not_direct_RGB' if path.suffix.lower() in ('.h5','.ply','.txt','.json','.log') else 'not_independently_admitted')
            inventory.append({'path': str(path), 'bytes': path.stat().st_size, 'rgb_adapter': included, 'reason': reason})
    write(a.output/'inventory.json', {'files': inventory, 'total_bytes': sum(r['bytes'] for r in inventory), 'tasks': tasks})
    result = []
    with ProcessPoolExecutor(a.workers) as pool:
        futures = {pool.submit(prepare, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            result.append(r)
            print(json.dumps(r), flush=True)
            write(a.output/'preparation_status.json', {'state': 'running', 'completed_sources': len(result),
                  'total_sources': len(tasks), 'training_frames': sum(x.get('unique_training_frames',0) for x in result),
                  'failed': [x for x in result if x['status'] != 'complete']})
    if any(r['status'] != 'complete' for r in result):
        write(a.output/'preparation_status.json', {'state': 'failed', 'sources': result})
        raise SystemExit('At least one source failed; see per-source failure.json')
    dedup = set()
    with (a.output/'visual_train.jsonl.tmp').open('w') as target:
        for t in tasks:
            for line in (a.output/'sources'/t['id']/'frames.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['sha256_rgb224'] not in dedup:
                    target.write(line+'\n')
                    dedup.add(row['sha256_rgb224'])
    (a.output/'visual_train.jsonl.tmp').replace(a.output/'visual_train.jsonl')
    write(a.output/'preparation_status.json', {'state': 'complete', 'sources': result,
          'unique_training_frames': len(dedup), 'all_training_rgb_visited': True,
          'note': 'Only RGB visual pretraining; no reconstruction rewards fabricated for uncalibrated datasets.'})


if __name__ == '__main__':
    main()
