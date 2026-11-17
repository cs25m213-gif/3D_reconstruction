"""Build training clips from full-rate UAVScenes flights, in the existing cache layout.

link     Makes the existing clips visible under the new roots, using symlinks.
plan     Chooses windows from each flight's pose file, mixing straight runs with turns, because
         turns are what near-parallel flights lack. Writes the exact file list to fetch.
build    Writes clip folders from fetched files. Each then needs one reconstruction, which is
         what produces its SfM model and DA3 cache.

Clips follow the interval-5 rhythm (every fifth frame, 2 Hz) so old and new data stay compatible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .utility_contracts import load_contract, write_json

STEP, FRAMES = 5, 60
SEQUENCE_FOLDER = 'interval1_{flight}/interval1_{flight}'


def link(contract: dict, root: Path) -> dict:
    data = contract['data']
    cache, lidar = root / data['cache_root'], root / data['lidar_root']
    (lidar / 'all_infos').mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    made = 0
    for folder in sorted((root / data['source_cache_root']).iterdir()):
        target = cache / folder.name
        if (folder / 'selected_infos.json').exists() and not target.exists():
            target.symlink_to(folder.resolve(), target_is_directory=True)
            made += 1
    for folder in sorted((root / data['source_lidar_root']).iterdir()):
        target = lidar / folder.name
        if folder.name != 'all_infos' and folder.is_dir() and not target.exists():
            target.symlink_to(folder.resolve(), target_is_directory=True)
    for file in sorted((root / data['source_lidar_root'] / 'all_infos').glob('*.json')):
        target = lidar / 'all_infos' / file.name
        if not target.exists():
            target.symlink_to(file.resolve())
    return {'linked_clips': made}


def choose_windows(poses: np.ndarray, times: np.ndarray, per_flight: int, used: list) -> list:
    """Half the windows that turn the most, half evenly spaced, none overlapping an existing clip."""
    span = STEP * FRAMES
    starts = [s for s in range(0, len(poses) - span, span // 2)
              if times[s + span - 1] - times[s] < 1.2 * span * np.median(np.diff(times))
              and not any(abs(s - u) < span for u in used)]
    if not starts:
        return []
    def turning(s):
        axes = poses[s:s + span:STEP, :2, 0]
        return float(np.abs(np.unwrap(np.arctan2(axes[:, 1], axes[:, 0]))).max() - np.abs(np.unwrap(np.arctan2(axes[:, 1], axes[:, 0]))).min())
    ranked = sorted(starts, key=turning, reverse=True)
    picked = []
    for s in ranked[:per_flight // 2] + [starts[int(i)] for i in np.linspace(0, len(starts) - 1, per_flight)]:
        if len(picked) < per_flight and not any(abs(s - p) < span for p in picked):
            picked.append(int(s))
    return sorted(picked)


def plan(contract: dict, root: Path, extracted: Path, per_flight: int, output: Path) -> dict:
    """Reads only sampleinfos files; these can be fetched first without any images."""
    data, rows, files = contract['data'], [], []
    for flight in contract['clips']['flights']:
        infos = json.loads((extracted / SEQUENCE_FOLDER.format(flight=flight) / 'sampleinfos_interpolated.json').read_text())
        poses = np.array([i['T4x4'] for i in infos], np.float64)
        times = np.array([float(Path(i['OriginalImageName']).stem) for i in infos])
        used = []
        for folder in sorted((root / data['source_cache_root']).glob(f'{flight}_*/selected_infos.json')):
            existing = json.loads(folder.read_text())
            names = {i['OriginalImageName']: k for k, i in enumerate(infos)}
            if existing and existing[0]['OriginalImageName'] in names:
                used.append(names[existing[0]['OriginalImageName']])
        for number, start in enumerate(choose_windows(poses, times, per_flight, used), start=2):
            positions = list(range(start, start + STEP * FRAMES, STEP))
            rows.append({'clip': f'{flight}_{number}', 'flight': flight, 'positions': positions})
            for p in positions:
                stem = Path(infos[p]['OriginalImageName']).stem
                base = SEQUENCE_FOLDER.format(flight=flight)
                files += [f'{base}/interval1_CAM/{stem}.jpg', f'{base}/interval1_LIDAR/image{stem}_lidar*.txt']
    write_json(output / 'clip_plan.json', {'step': STEP, 'frames': FRAMES, 'clips': rows})
    (output / 'clip_files.txt').write_text('\n'.join(files) + '\n')
    return {'clips': len(rows), 'patterns': len(files)}


def build(contract: dict, root: Path, extracted: Path, output: Path) -> dict:
    data = contract['data']
    table = json.loads((output / 'clip_plan.json').read_text())
    made = []
    for row in table['clips']:
        flight, clip = row['flight'], row['clip']
        folder, lidar = root / data['cache_root'] / clip, root / data['lidar_root'] / clip
        if (folder / 'session/video.mp4').exists():
            continue
        base = extracted / SEQUENCE_FOLDER.format(flight=flight)
        infos = json.loads((base / 'sampleinfos_interpolated.json').read_text())
        shutil.copyfile(base / 'sampleinfos_interpolated.json', root / data['lidar_root'] / 'all_infos' / f'{flight}.json') if not (root / data['lidar_root'] / 'all_infos' / f'{flight}.json').exists() else None
        selected = [infos[p] for p in row['positions']]
        for part in ('source/frames', 'source/lidar', 'session'):
            (folder / part).mkdir(parents=True, exist_ok=True)
        lidar.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, info in enumerate(selected):
            stem = Path(info['OriginalImageName']).stem
            image_file, scan = base / 'interval1_CAM' / f'{stem}.jpg', next((base / 'interval1_LIDAR').glob(f'image{stem}_lidar*.txt'))
            shutil.copyfile(image_file, folder / 'source/frames' / f'frame_{index:06d}.jpg')
            shutil.copyfile(scan, folder / 'source/lidar' / f'frame_{index:06d}.txt')
            shutil.copyfile(scan, lidar / f'frame_{index:06d}.txt')
            image_time, lidar_time = float(stem), float(scan.stem.split('_lidar')[1])
            rows.append({'frame': index, 'image_time': image_time, 'lidar_time': lidar_time, 'sync_offset_ms': (image_time - lidar_time) * 1000})
        write_json(folder / 'selected_infos.json', selected)
        write_json(lidar / 'manifest.json', rows)
        k = np.asarray(selected[0]['P3x3'])
        width, height = int(selected[0]['Width']), int(selected[0]['Height'])
        write_json(folder / 'session/camera_intrinsics.json', {'is_available': True, 'calibration_exact': True, 'width': width, 'height': height,
                                                               'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2]), 'distortion': [0, 0, 0, 0]})
        write_json(folder / 'session/metadata.json', {'complete': True, 'video_width': width, 'video_height': height, 'video_fps': 2, 'duration_seconds': FRAMES / 2.,
                                                      'camera_intrinsics': {'lens_intrinsic_calibration': [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), 0], 'pixel_array_size': [width, height]},
                                                      'source': f'UAVScenes interval1 {flight} clip {clip}; RGB-only reconstruction'})
        # The first clips' encoder, so old and new depth caches see the same artefacts. Last, because the video marks a finished clip.
        partial = folder / 'session/video.partial.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '2', '-i', str(folder / 'source/frames/frame_%06d.jpg'),
                        '-c:v', 'libx264', '-threads', '2', '-crf', '17', '-pix_fmt', 'yuv420p', str(partial)], check=True)
        partial.replace(folder / 'session/video.mp4')
        made.append(clip)
    return {'built': made}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('link', 'plan', 'build'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--extracted', type=Path, help='Folder holding the interval1_<flight> directories')
    parser.add_argument('--per-flight', type=int, default=6)
    parser.add_argument('--output', type=Path, default=Path('runs/vnext_dataset_v2'))
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'link':
        result = link(contract, root)
    elif args.command == 'plan':
        result = plan(contract, root, args.extracted, args.per_flight, args.output)
    else:
        result = build(contract, root, args.extracted, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
