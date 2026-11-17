"""Stage corrected JB3D text geometry beside, never over, downloaded archives.

This is an intermediate preparation stage, not admission to supervised training.
Video frame identity, radial undistortion and LiDAR alignment still require checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from prepare_uavlight import rotation, write_json

REPO = 'https://api.github.com/repos/JB3D-Dataset/Multi-View-Stereo'
UNAFFECTED = {'Main_stage_area_I', 'Main_stage_area_II', 'Flight_to_tower', 'Tower', 'Park_area'}


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def corrected_text(text, delta, image_file):
    output, observation = [], False
    for line in text.splitlines():
        if line.startswith('#'):
            output.append(line); continue
        if image_file and observation:
            output.append(line); observation = False; continue
        if not line.strip():
            output.append(line); continue
        fields = line.split(maxsplit=9) if image_file else line.split()
        if image_file:
            r = rotation([float(x) for x in fields[1:5]])
            t = np.array(fields[5:8], float) - r @ delta
            fields[5:8] = [format(x, '.17g') for x in t]
            observation = True
        else:
            point = np.array(fields[1:4], float) + delta
            fields[1:4] = [format(x, '.17g') for x in point]
        output.append(' '.join(fields))
    return '\n'.join(output) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision_file = args.output / 'correction_source.json'
    if revision_file.exists():
        revision = json.loads(revision_file.read_text())['revision']
    else:
        revision = json.loads(fetch(REPO + '/commits/main'))['sha']
        write_json(revision_file, dict(repository=REPO, revision=revision))
    rows = []
    for archive in sorted(args.source.glob('*.zip')):
        scene = archive.stem
        if scene.startswith('All_RGB_images'):
            rows.append(dict(scene=scene, state='reference_geometry_only', admitted=False)); continue
        output = args.output / scene
        output.mkdir(parents=True, exist_ok=True)
        done = output / 'manifest.json'
        if done.exists():
            rows.append(json.loads(done.read_text())); continue
        try:
            delta = np.zeros(3)
            transform_hash = None
            if scene not in UNAFFECTED:
                url = f'https://raw.githubusercontent.com/JB3D-Dataset/Multi-View-Stereo/{revision}/Translation_correction/{scene}/transform.txt'
                transform = fetch(url)
                matrix = np.loadtxt(transform.decode().splitlines())
                if matrix.shape != (4,4) or not np.allclose(matrix[:3,:3], np.eye(3)) or not np.allclose(matrix[3], [0,0,0,1]):
                    raise ValueError('Expected pure-translation correction')
                delta = matrix[:3,3]
                transform_hash = hashlib.sha256(transform).hexdigest()
                (output / 'transform.txt').write_bytes(transform)
            files = {}
            with zipfile.ZipFile(archive) as z:
                for name in ['cameras.txt', 'images.txt', 'images_30fps.txt', 'points3D.txt', 'polygon.txt', 'video.SRT']:
                    if name not in z.namelist():
                        continue
                    raw = z.read(name)
                    files[name] = hashlib.sha256(raw).hexdigest()
                    text = raw.decode('utf-8-sig')
                    if name.startswith('images') or name == 'points3D.txt':
                        text = corrected_text(text, delta, name.startswith('images'))
                    (output / name).write_text(text)
                videos = [dict(member=i.filename, bytes=i.file_size, crc=i.CRC) for i in z.infolist() if i.filename.lower().endswith('.mp4')]
            row = dict(scene=scene, state='corrected_metadata_ready', admitted=False, source_archive=str(archive),
                       correction_revision=revision, transform_sha256=transform_hash, delta_m=delta.tolist(),
                       source_metadata_sha256=files, videos=videos,
                       blockers=['Validate video frame number convention against poses',
                                 'Undistort RADIAL images and intrinsics consistently',
                                 'Build and validate registered LiDAR targets before geometry supervision'],
                       split_group='JB3D_shared_site',
                       note='All JB3D scenes share a site: do not claim independent scene generalization from nearby flights. Original PLY is uncorrected; use corrected text geometry here.')
            write_json(done, row)
        except Exception as exc:
            row = dict(scene=scene, state='preparation_failed', admitted=False, error=str(exc))
        rows.append(row)
        write_json(args.output / 'manifest.json', dict(scenes=rows, training_started=False))
    write_json(args.output / 'manifest.json', dict(scenes=rows, training_started=False))


if __name__ == '__main__':
    main()
