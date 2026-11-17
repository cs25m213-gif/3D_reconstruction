"""One-command saved-video reconstruction with an adaptive sparse policy."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

from .depth_normals import NORMAL_POLICY

POLICY = {'version': 4, 'dense_views': 128, 'dense_size': 960,
          'fusion_min_pixels': 3, 'rotation': 'none',
          'source_planner': 2, 'normal_policy': NORMAL_POLICY}


def file_identity(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def start_viewer(output):
    root = Path(os.environ.get('GEOSTREAM_WORKSPACE', str(Path.cwd()))).resolve()
    runs = root / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    latest = runs / 'latest_reconstruction'
    if latest.is_symlink() or not latest.exists():
        temporary = runs / f'.latest_reconstruction.{os.getpid()}'
        temporary.symlink_to(output.resolve(), target_is_directory=True)
        temporary.replace(latest)
        directory = latest
    else:
        directory = output
    identity = hashlib.sha256(str(root).encode()).hexdigest()
    (output / 'viewer_identity.txt').write_text(identity)
    service_path = runs / 'viewer_service.json'
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(('8.8.8.8', 80))
            address = sock.getsockname()[0]
        except OSError:
            address = '127.0.0.1'
    if service_path.is_file():
        try:
            service = json.loads(service_path.read_text())
            if service['directory'] == str(directory):
                with urllib.request.urlopen(f"http://127.0.0.1:{service['port']}/viewer_identity.txt", timeout=2) as response:
                    if response.read().decode() == identity:
                        return f"http://{address}:{service['port']}/"
        except (OSError, ValueError, KeyError):
            pass
    for port in range(8091, 8111):
        with socket.socket() as sock:
            try:
                sock.bind(('0.0.0.0', port))
            except OSError:
                continue
        with (output / 'http.log').open('a') as log:
            subprocess.run(['setsid', '-f', sys.executable, '-m', 'http.server', str(port),
                            '--bind', '0.0.0.0', '--directory', str(directory)],
                           stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
        service_path.write_text(json.dumps({'port': port, 'directory': str(directory)}, indent=2) + '\n')
        return f'http://{address}:{port}/'
    raise RuntimeError('No free viewer port between 8091 and 8110')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Session folder or saved video file')
    parser.add_argument('--output', type=Path, help='Optional output folder; identical inputs resume automatically')
    parser.add_argument('--no-serve', action='store_true', help='Write the viewer without starting its HTTP server')
    parser.add_argument('--hybrid', action='store_true',
                        help='Use PatchMatch depth, aligned prior and CPU TSDF; skip OpenMVS meshing')
    parser.add_argument('--multiview', action='store_true',
                        help='Use pose-conditioned DA3 and export supported and completed coloured meshes')
    args = parser.parse_args()
    if args.hybrid and args.multiview:
        parser.error('Choose either --hybrid or --multiview')
    source = args.input.resolve()
    video = source / 'video.mp4' if source.is_dir() else source
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata_path = video.parent / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    associated = source.is_dir() or metadata.get('video_file') == video.name
    if not associated:
        metadata = {}
    inputs = {'video': file_identity(video), 'metadata': metadata}
    if associated:
        for name in ('sensors.csv', 'gps.csv'):
            path = video.parent / name
            if path.is_file():
                inputs[name] = file_identity(path)
    project = Path(__file__).resolve().parent.parent
    implementations = ('fast_session.py', 'adaptive_sfm.py', 'dense_cuda.py', 'dense_coverage.py',
                       'depth_normals.py', 'dense_quality.py', 'planar_completion.py',
                       'mesh_viewer.py', 'pycolmap_backend.py', 'online.py', 'hybrid_dense.py', 'multiview_dense.py')
    code = {name: hashlib.sha256((project / 'adaptive_uav' / name).read_bytes()).hexdigest()
            for name in implementations}
    policy = {**POLICY, 'hybrid': bool(args.hybrid), 'multiview': bool(args.multiview)}
    manifest = {'inputs': inputs, 'policy': policy, 'implementation': code}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:10]
    root = Path(os.environ.get('GEOSTREAM_WORKSPACE', str(Path.cwd())))
    label = source.name if source.is_dir() else source.stem
    output = (args.output or root / 'runs' / f'{label}_automatic_{key}').resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'pipeline_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output belongs to different input/policy; omit --output to select the matching run automatically')
    if not manifest_path.exists() and (output / 'online/current_map.json').exists():
        raise ValueError('Legacy output has no input manifest; omit --output to create a verified run')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    prepared = output / 'input'
    prepared.mkdir(exist_ok=True)
    if not (prepared / 'video.mp4').exists():
        (prepared / 'video.mp4').symlink_to(video)
    (prepared / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name in ('sensors.csv', 'gps.csv'):
        if name in inputs and not (prepared / name).exists():
            (prepared / name).symlink_to(video.parent / name)
    page = '''<!doctype html><meta charset="utf-8"><title>Reconstruction</title>
<style>body{background:#101a23;color:#eef6fc;font:18px system-ui;max-width:820px;margin:40px auto}a{color:#8bdcff}li{margin:14px 0}</style>
<h1>Reconstruction: NAME</h1><p id="state">Starting…</p><p id="quality"></p>
<ul><li><a href="online/viewer.html">Camera map</a></li><li><a href="dense/mesh_viewer.html">Colored mesh</a></li>
REPORT_LINKS<li><a href="run_report.json">Run report</a></li></ul>
<p>Output links become available as their stages finish.</p><script>
async function update(){try{let r=await fetch('run_report.json',{cache:'no-store'});let s=await r.json();
let stage=s.status;if(stage==='dense_running'){try{let d=await(await fetch('dense/dense_metrics.json',{cache:'no-store'})).json();stage=d.status}catch{}}
document.getElementById('state').textContent=stage.replaceAll('_',' ')+(s.error?' — '+s.error:'');
document.getElementById('quality').textContent=s.quality_status?'Coverage: '+s.quality_status:''}catch{}}update();setInterval(update,3000);
</script>'''.replace('NAME', html.escape(label)).replace(
        'REPORT_LINKS',
        '<li><a href="dense/hybrid_dense_report.json">Measured / supported / inferred coverage</a></li>'
        if args.hybrid else '<li><a href="dense/quality/index.html">Coverage report</a></li>')
    if args.multiview:
        page = page.replace('dense/mesh_viewer.html', 'multiview/mesh_viewer.html').replace(
            'dense/quality/index.html', 'multiview/coverage.html')
    (output / 'index.html').write_text(page)
    print(f'Output: {output}', flush=True)
    if not args.no_serve:
        try:
            url = start_viewer(output)
        except (OSError, RuntimeError) as error:
            (output / 'viewer_service_error.txt').write_text(str(error) + '\n')
            print(f'Viewer server unavailable: {error}. Reconstruction continues.', flush=True)
        else:
            (output / 'viewer_url.txt').write_text(url + '\n')
            print(f'Viewer: {url}', flush=True)
    report_path = output / 'run_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        expected = (('multiview/scene_completed.glb', 'multiview/multiview_report.json') if args.multiview else
                    ('dense/scene_hybrid_mesh.glb', 'dense/hybrid_dense_report.json')
                    if args.hybrid else ('dense/scene_colored.glb', 'dense/quality_report.json'))
        if report.get('status') == 'complete' and all((output / name).is_file() for name in expected):
            print('Reusing finished artifacts with matching input and policy. '
                  f'Quality: {report.get("quality_status", "unverified")}', flush=True)
            return
    command = [sys.executable, '-u', '-m', 'adaptive_uav.fast_session', str(prepared),
               '--output', str(output), '--dense-views', str(POLICY['dense_views']),
               '--dense-size', str(POLICY['dense_size']), '--fusion-min-pixels', '3',
               '--rotation', POLICY['rotation'], '--automatic']
    if args.hybrid:
        command.append('--hybrid')
    if args.multiview:
        command.append('--multiview')
    with (output / 'pipeline.log').open('a') as log:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit(code)
    mesh = 'scene_hybrid_mesh.glb' if args.hybrid else 'scene_colored.glb'
    result_mesh = output/'multiview/scene_completed.glb' if args.multiview else output/'dense'/mesh
    print(f'Colored mesh: {result_mesh}', flush=True)


if __name__ == '__main__':
    main()
