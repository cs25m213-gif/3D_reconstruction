"""Stage 0: snapshot and verify the frozen controls before any vNext experiment.

Reads source, checkpoints and cache metadata only. No model runs and nothing
outside --output is written. Metric reproduction belongs to Stage 2.
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np

from .utility_contracts import load_contract, sha256_file, sha256_lines, write_json, write_status

SOURCE_GLOBS = ('uav_reconstruction/adaptive_uav/*.py', 'uav_reconstruction/configs/*.json', '*.sh')
PACKAGES = ('torch', 'torchvision', 'numpy', 'scipy', 'opencv-python', 'open3d', 'transformers', 'pycolmap')
DEFAULT_OPTIONS = {'uav_reconstruction/adaptive_uav/final_reconstruct.py': ('mode', 'detail', 'da3-model', 'depth-backend')}
NORMALIZATION = {'grid_height_width': [192, 256], 'target_validity_open_interval': [.25, 4.], 'min_anchors': 30,
                 'depth_planes': 24, 'depth_unit': 'camera-axis Z', 'runtime_pose': 'camera-to-world',
                 'student_pose': 'world-to-camera',
                 'scale_denominator': 'median valid positive reference sparse anchors, visual units',
                 'measured_normalized': 'd_measured_m / (alpha * s_visual)',
                 'alpha_source': 'scale_calibration.json: camera-centre Umeyama fit, not LiDAR'}


def source_snapshot(root: Path) -> dict:
    files = sorted({path for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in files}
    return {'revision': 'not a git repository; file hashes are the revision',
            'tree_sha256': sha256_lines(f'{sha} {name}' for name, sha in hashes.items()),
            'file_count': len(hashes), 'files': hashes}


def environment(root: Path) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True).stdout.strip()
    snapshot = root / 'ENVIRONMENT_SNAPSHOT.json'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'gpu': driver or None,
            'packages': versions, 'environment_snapshot_sha256': sha256_file(snapshot) if snapshot.exists() else None}


def tensor_values(state: dict) -> int:
    return sum(int(value.numel()) for value in state.values() if hasattr(value, 'numel'))


def jsonable(value):
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return None


def inspect_checkpoint(path: Path) -> dict:
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = next((payload[key] for key in ('model', 'state_dict', 'student') if isinstance(payload.get(key), dict)),
                 payload)
    metadata = {key: jsonable(value) for key, value in payload.items()
                if not hasattr(value, 'keys') or key in ('args', 'config', 'metrics')}
    return {'top_level_keys': sorted(map(str, payload)), 'state_tensor_values': tensor_values(state),
            'metadata': metadata}


def checkpoints(root: Path, contract: dict) -> tuple[dict, list]:
    rows, failures = {}, []
    for name, expected in contract['protected_checkpoints'].items():
        path = root / expected['path']
        if not path.exists():
            rows[name] = {'path': expected['path'], 'status': 'missing'}
            failures.append(f'{name}: missing')
            continue
        actual = sha256_file(path)
        rows[name] = {'path': expected['path'], 'bytes': path.stat().st_size, 'sha256': actual,
                      'recorded_sha256': expected['sha256'], 'match': actual == expected['sha256'],
                      **inspect_checkpoint(path)}
        if actual != expected['sha256']:
            failures.append(f'{name}: hash mismatch')
    return rows, failures


def splits(root: Path, contract: dict) -> tuple[dict, list]:
    from .train_depth_student_lidar import ROLES, discover
    data = contract['data']
    scale_file = root / data['scale_calibration']
    scales = json.loads(scale_file.read_text())
    paths, scenes = discover(root / data['cache_root'], root / data['sidecars'], scales)
    rows, failures = {}, []
    for role, files in paths.items():
        relative = sorted(str(Path(file).relative_to(root)) for file in files)
        expected = contract['expected_split_examples'].get(role)
        rows[role] = {'examples': len(relative), 'recorded_examples': expected, 'scenes': scenes[role],
                      'identity_sha256': sha256_lines(relative)}
        if expected != len(relative):
            failures.append(f'{role}: {len(relative)} examples, recorded {expected}')
    by_scene = {scene: role for role, names in scenes.items() for scene in names}
    overlap = sorted(set(scenes['train']) & (set(scenes['validation']) | set(scenes['test'])))
    if overlap:
        failures.append(f'flights in more than one role: {overlap}')
    return {'mapping_source': 'ROLES embedded in train_depth_student_lidar.py',
            'area_roles': ROLES, 'roles': rows, 'flight_roles': by_scene,
            'test_role_note': 'HKisland is diagnostic, not a sealed final test.',
            'scale_calibration_sha256': sha256_file(scale_file),
            'usable_scale_batches': sum(bool(row.get('usable')) for row in scales.values())}, failures


def teacher_contexts(root: Path, contract: dict) -> tuple[dict, list]:
    cache_root = root / contract['data']['cache_root']
    models, reports, anchors, lengths, identity = collections.Counter(), {}, collections.Counter(), collections.Counter(), []
    for batch in sorted(p for p in cache_root.iterdir() if (p / 'selected_infos.json').exists()):
        report = batch / 'reconstruction/run_report.json'
        if report.exists():
            text = report.read_text()
            reports[batch.name] = sha256_file(report)
            models.update(set(re.findall(r'depth-anything/[\w.-]+', text)))
        for file in sorted((batch / 'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz')):
            with np.load(file) as cached:
                context = [str(name) for name in cached['context']]
                anchors[str(cached['anchor_source'])] += 1
            lengths[len(context)] += 1
            identity.append(f'{batch.name}/{file.name}:' + ','.join(context))
    failures = [] if len(models) == 1 else [f'expected one teacher model identity, found {sorted(models)}']
    return {'models_seen_in_run_reports': dict(models), 'batches_with_reports': len(reports),
            'run_report_sha256': reports, 'cached_views': len(identity),
            'context_length_counts': {str(key): value for key, value in sorted(lengths.items())},
            'anchor_source_counts': dict(anchors), 'context_identity_sha256': sha256_lines(identity)}, failures


def production_defaults(root: Path) -> dict:
    rows = {}
    for file, options in DEFAULT_OPTIONS.items():
        text = (root / file).read_text()
        for option in options:
            match = re.search(rf"add_argument\('--{option}'.*?default='([^']+)'", text, re.S)
            rows[option] = match.group(1) if match else None
    return {'final_reconstruct_defaults': rows, 'policy_source': 'keyframe_policy.py, hashed under source.files',
            'recurrent_updater': 'disabled per plan; selectors and bins opt-in'}


def freeze(contract: dict, contract_path: Path, root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    write_status(output, 'stage0_freeze', 'running', None, None)
    checks, manifest = {}, {'schema_version': 1, 'stage': 'stage0_freeze',
                            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': platform.node(),
                            'contract_path': str(contract_path), 'contract_sha256': sha256_file(contract_path),
                            'scope': 'Inventory only. Recorded metrics were not reproduced here; that is Stage 2.'}
    manifest['source'] = source_snapshot(root)
    manifest['environment'] = environment(root)
    manifest['checkpoints'], checks['checkpoints'] = checkpoints(root, contract)
    manifest['splits'], checks['splits'] = splits(root, contract)
    manifest['teacher'], checks['teacher'] = teacher_contexts(root, contract)
    manifest['normalization'] = NORMALIZATION
    manifest['production'] = production_defaults(root)
    defaults = manifest['production']['final_reconstruct_defaults']
    checks['production'] = [] if defaults.get('depth-backend') == 'da3' else [f'depth default changed: {defaults}']
    failures = [item for rows in checks.values() for item in rows]
    manifest['gate'] = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
                        'criterion': 'All inputs resolve; checkpoints match; split counts match; one teacher identity; DA3 stays default.',
                        'limitations': ['Source has no git history, so older code cannot be recovered from hashes.',
                                        'Recorded metrics are not reproduced until Stage 2.'],
                        'next_eligible_stage': 'stage1_geometry' if not failures else None}
    write_json(output / 'frozen_manifest.json', manifest)
    write_json(output / 'resolved_config.json', contract)
    (output / 'report.md').write_text(render(manifest))
    write_status(output, 'stage0_freeze', 'complete', manifest['gate']['status'],
                 manifest['gate']['next_eligible_stage'], '; '.join(failures))
    return manifest


def render(manifest: dict) -> str:
    gate, split_rows = manifest['gate'], manifest['splits']['roles']
    lines = [f"# Stage 0 baseline freeze: {gate['status']}", '', manifest['scope'], '',
             f"Source tree SHA-256: `{manifest['source']['tree_sha256']}` ({manifest['source']['file_count']} files)", '',
             '| Checkpoint | Hash match | Tensor values | Bytes |', '|---|---|---:|---:|']
    for name, row in manifest['checkpoints'].items():
        lines.append(f"| {name} | {row.get('match')} | {row.get('state_tensor_values')} | {row.get('bytes')} |")
    lines += ['', '| Role | Examples | Recorded | Flights |', '|---|---:|---:|---|']
    for role, row in split_rows.items():
        lines.append(f"| {role} | {row['examples']} | {row['recorded_examples']} | {', '.join(row['scenes'])} |")
    teacher = manifest['teacher']
    lines += ['', f"Teacher identity in run reports: {teacher['models_seen_in_run_reports']}",
              f"Cached views: {teacher['cached_views']}; context lengths: {teacher['context_length_counts']}",
              f"Anchor sources: {teacher['anchor_source_counts']}",
              f"Production defaults: {manifest['production']['final_reconstruct_defaults']}", '', '## Failures', '']
    lines += [f'- {item}' for item in gate['failures']] or ['None.']
    lines += ['', '## Limitations', ''] + [f'- {item}' for item in gate['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--dry-run', action='store_true', help='Validate the contract and input paths; write nothing')
    parser.add_argument('--reuse-existing',action='store_true',help='Verify immutable control identities without overwriting a prior freeze')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    inputs = [contract['depth_checkpoint'], *contract['data'].values(),
              *(row['path'] for row in contract['protected_checkpoints'].values())]
    missing = [path for path in inputs if not (root / path).exists()]
    if args.dry_run:
        print(json.dumps({'contract': contract['contract_id'], 'inputs': len(inputs), 'missing': missing}, indent=2))
        raise SystemExit(1 if missing else 0)
    if (args.output / 'frozen_manifest.json').exists():
        if args.reuse_existing:
            saved=json.loads((args.output/'frozen_manifest.json').read_text())
            if saved.get('gate',{}).get('status')!='PASS' or saved.get('contract_sha256')!=sha256_file(args.config):
                raise SystemExit('Existing baseline failed or its contract changed; choose a new baseline directory')
            for name,row in contract['protected_checkpoints'].items():
                if sha256_file(root/row['path'])!=row['sha256']:
                    raise SystemExit('Protected checkpoint changed: '+name)
            print('Reused immutable baseline controls; current code identity is recorded in the stage signature.')
            return
        raise SystemExit(f'{args.output} already holds a frozen manifest; use a new experiment directory')
    manifest = freeze(contract, args.config.resolve(), root, args.output)
    print(json.dumps(manifest['gate'], indent=2))
    raise SystemExit(0 if manifest['gate']['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
