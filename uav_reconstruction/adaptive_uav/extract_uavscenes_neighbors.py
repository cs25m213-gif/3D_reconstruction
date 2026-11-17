"""Extract nearby UAVScenes RGB frames for metric LiDAR training targets."""
import argparse
import json
from pathlib import Path
import zipfile


def extract(archive: Path, scene_dir: Path, scene: str = 'AMtown01') -> dict:
    base = f'interval5_CAM_LIDAR/interval5_{scene}/interval5_CAM/'
    rows = json.loads((scene_dir / 'manifest.json').read_text())
    all_infos = json.loads((scene_dir / 'all_sampleinfos.json').read_text())
    info_by_name = {row['OriginalImageName']: row for row in all_infos}
    out = scene_dir / 'neighbor_rgb'
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    with zipfile.ZipFile(archive) as z:
        images = sorted(name for name in z.namelist() if name.startswith(base) and name.endswith('.jpg'))
        position = {Path(name).name: i for i, name in enumerate(images)}
        for row in rows:
            filename = row['info']['OriginalImageName']
            if filename not in position:
                continue
            index = position[filename]
            candidates = []
            for offset in (-4, -3, 3, 4):
                other = index + offset
                if not (0 <= other < len(images)):
                    continue
                member = images[other]
                name = Path(member).name
                if name not in info_by_name:
                    continue
                target = out / name
                if not target.exists():
                    target.write_bytes(z.read(member))
                candidates.append({'rgb': str(target.relative_to(scene_dir)), 'info': info_by_name[name], 'offset': offset})
            result[str(row['index'])] = candidates
    (scene_dir / 'neighbor_manifest.json').write_text(json.dumps(result) + '\n')
    return {'targets': len(result), 'neighbor_images': len(list(out.glob('*.jpg')))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--scene', default='AMtown01')
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.scene_dir, args.scene), indent=2))
