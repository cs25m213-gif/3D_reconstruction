"""Convert publisher-posed UAVLight ZIPs into auxiliary student caches on the data server.

No fabricated teacher/metric targets: zero depth/confidence explicitly disable teacher loss.
Sparse SfM anchors and same-session, causal RGB sources provide auxiliary supervision.
Publisher test images are excluded from train caches. Tile variants share a scene role.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np

VERSION = 1
SHAPE = (192, 256)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > .001:
        raise ValueError('Invalid COLMAP quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def images(text):
    stream = iter(text.splitlines())
    for line in stream:
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError('Invalid COLMAP image record')
        observations = next(stream).split()  # Empty observation lines are significant.
        if len(observations) % 3:
            raise ValueError('Invalid COLMAP observations')
        p = np.eye(4)
        p[:3, :3] = rotation([float(x) for x in fields[1:5]])
        p[:3, 3] = [float(x) for x in fields[5:8]]
        obs = [(float(observations[i]), float(observations[i+1]), int(observations[i+2]))
               for i in range(0, len(observations), 3)]
        yield dict(id=int(fields[0]), camera=int(fields[8]), name=fields[9], P=p, obs=obs)


def members(z):
    result = {}
    for name in z.namelist():
        normalized = name.replace('\\', '/')
        p = PurePosixPath(normalized)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe archive member')
        if normalized in result:
            raise ValueError('Duplicate archive member')
        result[normalized] = name
    return result


def prepare(archive, output, role):
    digest = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    scene = archive.stem
    marker = output / scene / 'manifest.json'
    identity = dict(version=VERSION, sha256=source_hash, role=role)
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['identity'] != identity:
            raise ValueError('Existing conversion identity differs; use a new output directory')
        for clip in previous['clips']:
            cache = output / clip['cache']
            if len(list(cache.glob('frame_*.jpg.npz'))) != clip['frames']:
                raise ValueError('Completed cache is missing files')
        return previous
    clips, rejected, ignored = [], [], 0
    with zipfile.ZipFile(archive) as z:
        mapping = members(z)
        def read(suffix):
            matches = [n for n in mapping if n.endswith('/' + suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {suffix}, found {len(matches)}')
            return z.read(mapping[matches[0]]).decode('utf-8-sig')
        split = dict(line.rsplit(',', 1) for line in read('split.csv').splitlines() if line.strip())
        cameras = {}
        for line in read('sparse/0/cameras.txt').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            cid, model, w, h, *params = line.split()
            if model != 'PINHOLE' or len(params) != 4:
                raise ValueError(f'Unsupported camera: {model}; undistortion is required')
            fx, fy, cx, cy = map(float, params)
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
            if not np.isfinite(k).all() or min(fx, fy) <= 0:
                raise ValueError('Invalid intrinsics')
            cameras[int(cid)] = (int(w), int(h), k)
        rows = list(images(read('sparse/0/images.txt')))
        train_ids = {r['id'] for r in rows if split.get(r['name']) == 'train'}
        points = {}
        for line in read('sparse/0/points3D.txt').splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            f = line.split()
            # Require support from at least two publisher training images.
            support = sum(int(x) in train_ids for x in f[8::2])
            if float(f[7]) <= 4 and support >= 2:
                points[int(f[0])] = np.asarray(f[1:4], dtype=float)
        sessions = {}
        for row in rows:
            if split.get(row['name']) != 'train':
                ignored += 1
                continue
            session = str(PurePosixPath(row['name']).parent)
            sessions.setdefault(session, []).append(row)
        for session, sequence in sorted(sessions.items()):
            # Separate recording sessions: never create a temporal pair across flights.
            sequence.sort(key=lambda r: float(PurePosixPath(r['name']).stem))
            name = scene + '__' + session.replace('/', '_')
            cache = output / scene / name / 'reconstruction/multiview/depth_cache'
            cache.mkdir(parents=True, exist_ok=True)
            infos = []
            for row in sequence:
                w, h, k = cameras[row['camera']]
                p = row['P']
                xyz, xy = [], []
                for x, y, pid in row['obs']:
                    if pid in points:
                        xyz.append(points[pid]); xy.append([x, y])
                if len(xyz) < 30:
                    rejected.append(dict(image=row['name'], reason='insufficient_tracked_points')); continue
                cam = np.asarray(xyz) @ p[:3, :3].T + p[:3, 3]
                uv = cam @ k.T
                uv = uv[:, :2] / np.maximum(cam[:, 2:], 1e-9)
                error = np.linalg.norm(uv - np.asarray(xy), axis=1)
                valid = (cam[:, 2] > 0) & np.isfinite(cam).all(1) & (error <= 3)
                valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                anchor = np.full(SHAPE, np.inf, np.float32)
                pix = (uv[valid] * [SHAPE[1]/w, SHAPE[0]/h]).astype(int)
                np.minimum.at(anchor, (pix[:, 1], pix[:, 0]), cam[valid, 2])
                anchor[~np.isfinite(anchor)] = 0
                if np.count_nonzero(anchor) < 30:
                    rejected.append(dict(image=row['name'], reason='insufficient_reprojection_checked_anchors')); continue
                member = f'{scene}/images/{row["name"]}'
                rgb = cv2.imdecode(np.frombuffer(z.read(mapping[member]), np.uint8), cv2.IMREAD_COLOR)
                if rgb is None or rgb.shape[:2] != (h, w):
                    raise ValueError(f'Image/calibration size mismatch: {member}')
                rgb = cv2.cvtColor(cv2.resize(rgb, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                small_k = k.copy(); small_k[0] *= SHAPE[1]/w; small_k[1] *= SHAPE[0]/h
                index = len(infos)
                file = cache / f'frame_{index:06d}.jpg.npz'
                with file.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, rgb=rgb, K=small_k, P=p, anchors=anchor,
                                        depth=np.zeros(SHAPE, np.float32), confidence=np.zeros(SHAPE, np.float32))
                file.with_suffix('.tmp').replace(file)
                infos.append(dict(index=index, source_image=row['name'], timestamp=float(PurePosixPath(row['name']).stem),
                                  anchors=int(np.count_nonzero(anchor)), reprojection_median_px=float(np.median(error[valid]))))
            write_json(cache.parent.parent.parent / 'source_frames.json', infos)
            if len(infos) >= 3:
                clips.append(dict(name=name, scene='uavlight:' + scene.split('-')[0], role=role,
                                  cache=str(cache.relative_to(output)), frames=len(infos),
                                  supervision='sfm_anchors_and_photometric', selector_ready=False))
    result = dict(identity=identity, scene=scene, clips=clips, rejected=rejected,
                  publisher_nontrain_excluded=ignored, teacher_available=False, measured_depth_available=False,
                  caveat='Publisher SfM used multiple views. Auxiliary supervision only; not independent geometry ground truth.')
    write_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    archives = sorted(args.source.rglob('*.zip'))
    if not archives:
        raise SystemExit('No complete UAVLight archives')
    groups = sorted({p.stem.split('-')[0] for p in archives})
    # Stable SHA order, shared by all variants of a physical tile; freeze to disk.
    groups.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    held = max(1, len(groups)//5)
    roles = {g: ('test' if i < held else 'validation' if i < 2*held else 'train') for i, g in enumerate(groups)}
    split_file = args.output / 'splits.json'
    if split_file.exists() and json.loads(split_file.read_text()) != roles:
        raise SystemExit('Input scene groups changed; use a new versioned output directory')
    write_json(split_file, roles)
    manifest = dict(schema='geostreamfusion.auxiliary.v1', dataset='UAVLight', clips=[], scenes=[], failures=[])
    for archive in archives:
        try:
            result = prepare(archive, args.output, roles[archive.stem.split('-')[0]])
            manifest['clips'].extend(result['clips'])
            manifest['scenes'].append(dict(scene=archive.stem, excluded=result['publisher_nontrain_excluded'], rejected=len(result['rejected'])))
        except Exception as exc:
            manifest['failures'].append(dict(scene=archive.stem, error=str(exc)))
        write_json(args.output / 'manifest.json', manifest)
        write_json(args.output / 'status.json', dict(state='preparing', pid=os.getpid(), scenes_processed=len(manifest['scenes']),
                   failures=manifest['failures'], clips=len(manifest['clips']), training_started=False))
    write_json(args.output / 'status.json', dict(state='complete' if not manifest['failures'] else 'partial_failure',
               scenes_processed=len(manifest['scenes']), failures=manifest['failures'],
               clips=len(manifest['clips']), frames=sum(c['frames'] for c in manifest['clips']), training_started=False))


if __name__ == '__main__':
    main()
