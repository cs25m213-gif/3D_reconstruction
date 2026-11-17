"""Build a dense OpenMVS mesh from a saved PyCOLMAP camera map and source MP4."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import pycolmap


def ply_counts(path):
    with Path(path).open('rb') as stream:
        header = stream.read(4096).split(b'end_header', 1)[0].decode('ascii', 'replace')
    vertices = re.search(r'^element vertex (\d+)$', header, re.M)
    faces = re.search(r'^element face (\d+)$', header, re.M)
    return int(vertices.group(1)) if vertices else 0, int(faces.group(1)) if faces else 0


def run_dense(video: Path, model: Path, output: Path, openmvs: Path, runtime: Path | None,
              width: int = 1920, threads: int = 8, resolution_level: int = 0,
              geometric_iters: int = 2, source_images: Path | None = None):
    video, model, output, openmvs = map(lambda p: Path(p).resolve(),
                                        (video, model, output, openmvs))
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'dense_metrics.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({'source_video': str(video), 'source_model': str(model),
                   'image_width': width, 'dense_resolution_level': resolution_level,
                   'geometric_iters': geometric_iters})
    if source_images is not None:
        source_images = Path(source_images).resolve()
        report['source_images'] = str(source_images)

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def command(stage, binary, args, expected):
        expected = output / expected
        if expected.is_file() and expected.stat().st_size > 0:
            print(f'[{stage}] reusing {expected}', flush=True)
            return
        executable = openmvs / binary
        if not executable.is_file():
            raise FileNotFoundError(executable)
        if runtime:
            runtime_lib = Path(runtime).resolve() / 'usr/lib/x86_64-linux-gnu'
            launcher = [str(runtime_lib / 'ld-linux-x86-64.so.2'),
                        '--library-path', str(runtime_lib), str(executable)]
        else:
            launcher = [str(executable)]
        logfile = output / f'{stage}.log'
        tick = time.monotonic()
        print(f'[{stage}] running; log: {logfile}', flush=True)
        with logfile.open('w') as stream:
            subprocess.run(launcher + args, cwd=output, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if not expected.is_file() or expected.stat().st_size == 0:
            raise RuntimeError(f'{stage} did not produce {expected}; see {logfile}')
        report.setdefault('stage_seconds', {})[stage] = time.monotonic() - tick
        save()

    reconstruction = pycolmap.Reconstruction(model)
    images = sorted(image.name for image in reconstruction.images.values() if image.has_pose)
    if len(images) < 5:
        raise RuntimeError(f'Only {len(images)} camera poses; dense stereo needs more views')
    staged_images = output / 'source_images'
    staged_images.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open {video}')
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if video_width <= 0 or video_height <= 0:
        raise RuntimeError('Source video dimensions are invalid')
    if source_images is not None:
        first = cv2.imread(str(source_images / images[0]))
        if first is None:
            raise RuntimeError(f'Missing selected source image: {source_images / images[0]}')
        height = round(first.shape[0] * width / first.shape[1])
    else:
        height = round(video_height * width / video_width)
    report['registered_images'] = len(images)
    report['source_frame_count'] = frame_count
    report['source_fps'] = source_fps
    frame_indices = [int(re.fullmatch(r'frame_(\d+)\.jpg', name).group(1))
                     for name in images]
    report['registered_video_span_s'] = [min(frame_indices) / source_fps,
                                         max(frame_indices) / source_fps]
    refined_metrics = model.parent / 'refinement_metrics.json'
    if refined_metrics.is_file():
        selected = json.loads(refined_metrics.read_text()).get('selected_images')
        if selected:
            report['selected_images'] = selected
            report['registration_fraction'] = len(images) / selected
    save()
    try:
        for n, name in enumerate(images, 1):
            target = staged_images / name
            if target.is_file():
                continue
            match = re.fullmatch(r'frame_(\d+)\.jpg', name)
            if not match:
                raise ValueError(f'Image name has no source frame index: {name}')
            index = int(match.group(1))
            if source_images is not None:
                original = source_images / name
                image = cv2.imread(str(original))
                if image is None:
                    raise RuntimeError(f'Could not read selected image {original}')
                if image.shape[1] == width and image.shape[0] == height:
                    shutil.copy2(original, target)
                else:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f'Could not save {target}')
            else:
                if index >= frame_count or not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(f'Could not seek source frame {index}')
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not read source frame {index}')
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Could not save {target}')
            if n % 10 == 0 or n == len(images):
                print(f'[source frames] {n}/{len(images)}', flush=True)
    finally:
        capture.release()
    highres_model = output / 'highres_model'
    if not (highres_model / 'cameras.bin').is_file():
        for camera in reconstruction.cameras.values():
            if abs(camera.width / camera.height - width / height) > 0.002:
                raise RuntimeError('Video and reconstruction camera aspect ratios differ')
            camera.rescale(width, height)
        highres_model.mkdir(exist_ok=True)
        reconstruction.write(highres_model)
    if not (output / 'sparse/cameras.bin').is_file():
        print('[undistort] preparing high resolution views', flush=True)
        tick = time.monotonic()
        pycolmap.undistort_images(output, highres_model, staged_images, num_threads=threads)
        report.setdefault('stage_seconds', {})['undistort'] = time.monotonic() - tick
        save()
    command('import', 'InterfaceCOLMAP', ['-i', '.', '-o', 'scene.mvs'], 'scene.mvs')
    command('densify', 'DensifyPointCloud',
            ['-i', 'scene.mvs', '-o', 'scene_dense.mvs', '--max-threads', str(threads),
             '--resolution-level', str(resolution_level), '--min-resolution', '960',
             '--number-views', '5', '--number-views-fuse', '3',
             '--geometric-iters', str(geometric_iters),
             '--estimate-roi', '0', '--crop-to-roi', '0'], 'scene_dense.ply')
    report['dense_points'], _ = ply_counts(output / 'scene_dense.ply')
    save()
    if report['dense_points'] < 1000:
        raise RuntimeError(f"Only {report['dense_points']} dense points; meshing would be misleading")
    command('mesh', 'ReconstructMesh',
            ['-i', 'scene_dense.mvs', '-p', 'scene_dense.ply',
             '-o', 'scene_mesh.mvs', '--max-threads', str(threads)], 'scene_mesh.ply')
    report['mesh_vertices'], report['mesh_faces'] = ply_counts(output / 'scene_mesh.ply')
    report['status'] = 'mesh_complete'
    save()
    command('texture', 'TextureMesh',
            ['-i', 'scene_dense.mvs', '-m', 'scene_mesh.ply',
             '-o', 'scene_textured_glb.mvs', '--export-type', 'glb',
             '--max-threads', str(threads)], 'scene_textured_glb.glb')
    report['textured_glb_bytes'] = (output / 'scene_textured_glb.glb').stat().st_size
    report['texture_files'] = [path.name for path in sorted(output.glob('scene_textured_glb_*.png'))]
    report['status'] = 'textured_mesh_complete'
    save()
    print(json.dumps(report, indent=2), flush=True)
    return output / 'scene_mesh.ply'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--openmvs', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--resolution-level', type=int, default=0)
    parser.add_argument('--geometric-iters', type=int, default=2,
                        help='OpenMVS geometric consistency passes; use 0 to fuse cached depth maps with less memory')
    parser.add_argument('--source-images', type=Path,
                        help='Selected, already oriented RGB keyframes; use their pixels instead of seeking the raw video')
    parser.add_argument('--three-assets', type=Path,
                        help='Copy local Three.js modules and write a browser mesh viewer')
    args = parser.parse_args()
    run_dense(args.video, args.model, args.output, args.openmvs, args.runtime,
              args.width, args.threads, args.resolution_level, args.geometric_iters,
              args.source_images)
    if args.three_assets:
        from .mesh_viewer import write_mesh_viewer
        print(write_mesh_viewer(args.output, args.three_assets))


if __name__ == '__main__':
    main()
