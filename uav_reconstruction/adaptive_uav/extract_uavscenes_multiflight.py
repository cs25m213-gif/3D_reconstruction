"""Extract a bounded, reproducible UAVScenes RGB/LiDAR subset by complete flight.

Uses only the Python standard library so it can run beside a remote dataset ZIP.
Training roles are assigned by physical area, not by adjacent frames.
"""
import argparse
import json
from pathlib import Path
import re
import zipfile


ROLES = {'AMtown01': 'train', 'HKairport01': 'train',
         'AMvalley01': 'validation', 'HKisland03': 'test'}


def extract(archive: Path, output: Path, per_flight: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result = {'archive': str(archive), 'per_flight_limit': per_flight,
              'splits': {role: [] for role in ('train', 'validation', 'test')}}
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for scene, role in ROLES.items():
            base = f'interval5_CAM_LIDAR/interval5_{scene}/'
            infos = json.loads(z.read(base + 'sampleinfos_interpolated.json'))
            by_name = {row['OriginalImageName']: row for row in infos}
            image_names = sorted(n for n in names if n.startswith(base + 'interval5_CAM/') and n.endswith('.jpg'))
            lidar_names = {re.search(r'/image([0-9.]+)_lidar', n).group(1): n
                           for n in names if n.startswith(base + 'interval5_LIDAR/')
                           and re.search(r'/image([0-9.]+)_lidar', n)}
            pairs = []
            for image in image_names:
                filename = Path(image).name
                stamp = Path(image).stem
                if filename in by_name and stamp in lidar_names:
                    pairs.append((image, lidar_names[stamp], by_name[filename]))
            if len(pairs) < per_flight:
                selected = pairs
            else:
                chosen = [round(i * (len(pairs) - 1) / (per_flight - 1)) for i in range(per_flight)]
                selected = [pairs[i] for i in chosen]
            target = output / scene
            (target / 'rgb').mkdir(parents=True, exist_ok=True)
            (target / 'lidar').mkdir(parents=True, exist_ok=True)
            rows = []
            for index, (image, lidar, info) in enumerate(selected):
                stem = f'frame_{index:06d}'
                rgb_out = target / 'rgb' / (stem + '.jpg')
                lidar_out = target / 'lidar' / (stem + '.txt')
                if not rgb_out.exists():
                    rgb_out.write_bytes(z.read(image))
                if not lidar_out.exists():
                    lidar_out.write_bytes(z.read(lidar))
                lidar_stamp = float(re.search(r'_lidar([0-9.]+)\.txt', lidar).group(1))
                image_stamp = float(Path(image).stem)
                rows.append({'scene': scene, 'role': role, 'index': index, 'info': info,
                             'image_timestamp': image_stamp, 'lidar_timestamp': lidar_stamp,
                             'sync_offset_ms': (image_stamp - lidar_stamp) * 1000,
                             'rgb': str(rgb_out), 'lidar': str(lidar_out)})
            (target / 'selected_infos.json').write_text(json.dumps([r['info'] for r in rows]) + '\n')
            (target / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
            result['splits'][role].append({'scene': scene, 'frames': len(rows),
                                           'max_sync_offset_ms': max(abs(r['sync_offset_ms']) for r in rows)})
    (output / 'split_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-flight', type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.output, args.per_flight), indent=2))
