"""Extract LiDAR scans matching existing UAVScenes teacher-cache frame timestamps."""
import argparse
import json
from pathlib import Path
import re
import zipfile


def extract(archive: Path, requests: Path, output: Path) -> dict:
    jobs = json.loads(requests.read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = {'requested': 0, 'found': 0, 'missing': [], 'batches': {}}
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        for batch, infos in jobs.items():
            if not infos:
                continue
            scene = infos[0]['OriginalSceneName']
            prefix = f'interval5_CAM_LIDAR/interval5_{scene}/interval5_LIDAR/'
            lookup = {}
            for name in members:
                if name.startswith(prefix):
                    match = re.search(r'/image([0-9.]+)_lidar([0-9.]+)\.txt$', name)
                    if match:
                        lookup[match.group(1)] = (name, float(match.group(2)))
            folder = output / batch
            folder.mkdir(parents=True, exist_ok=True)
            rows = []
            for i, info in enumerate(infos):
                result['requested'] += 1
                stamp = Path(info['OriginalImageName']).stem
                if stamp not in lookup:
                    result['missing'].append(f'{batch}:{stamp}')
                    continue
                name, lidar_time = lookup[stamp]
                target = folder / f'frame_{i:06d}.txt'
                if not target.exists():
                    target.write_bytes(z.read(name))
                rows.append({'frame': i, 'image_time': float(stamp), 'lidar_time': lidar_time,
                             'sync_offset_ms': (float(stamp) - lidar_time) * 1000})
                result['found'] += 1
            (folder / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
            result['batches'][batch] = {'scene': scene, 'frames': len(rows)}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.archive, args.requests, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ('missing', 'batches')}, indent=2))
