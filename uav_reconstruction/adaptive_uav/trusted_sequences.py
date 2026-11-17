"""Canonical frame records, split manifests and leakage checks over the existing caches.

Large arrays stay where they are. Reference geometry lives in its own namespace, and the
policy loader refuses it, so a selector cannot see it even by mistake.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utility_contracts import load_contract, sha256_file, sha256_lines, write_json, write_status

ROLES = {'AMtown': 'train', 'HKairport': 'train', 'AMvalley': 'validation', 'HKisland': 'diagnostic_test'}
REFERENCE_FIELDS = ('reference_geometry_ref', 'measured_depth')


def split_of(scene: str) -> str:
    return next(role for prefix, role in ROLES.items() if scene.startswith(prefix))


def frame_records(root: Path, contract: dict, gate: dict | None):
    cache_root = root / contract['data']['cache_root']
    for batch in sorted(p for p in cache_root.iterdir() if (p / 'selected_infos.json').exists()):
        infos = json.loads((batch / 'selected_infos.json').read_text())
        scene = infos[0]['OriginalSceneName']
        if 'GNSS' in scene:
            continue
        lidar = root / contract['data']['lidar_root'] / batch.name / 'manifest.json'
        rows = {r['frame']: r for r in json.loads(lidar.read_text())} if lidar.exists() else {}
        status = ((gate or {}).get('flights', {}).get(scene) or {}).get('status', 'UNRESOLVED')
        for index, info in enumerate(infos):
            row = rows.get(index, {})
            yield {'schema_version': 1, 'frame_id': f"{scene}/{info['OriginalImageName']}", 'sequence_id': batch.name,
                   'location_id': scene[:-2], 'flight_id': scene, 'split': split_of(scene), 'dataset_id': 'uavscenes',
                   'rgb': {'path': str((batch / 'source/frames' / f'frame_{index:06d}.jpg').relative_to(root)),
                           'width': info['Width'], 'height': info['Height'], 'order': 'BGR on disk'},
                   'time': {'sensor_time': row.get('image_time'), 'lidar_time': row.get('lidar_time'),
                            'sync_offset_ms': row.get('sync_offset_ms'), 'video_pts': None},
                   'camera': {'K': info['P3x3'], 'model': 'pinhole, zero distortion metadata', 'source': 'UAVScenes map camera'},
                   'reference_geometry_ref': {'namespace': 'offline_only', 'status': status,
                                              'T_world_from_camera': info['T4x4']},
                   'measured_depth': {'unit': 'camera-Z metres', 'sidecar': f"{contract['data']['sidecars']}/{batch.name}/frame_{index:06d}.npz",
                                      'status': status}}


def validate_splits(records) -> list[str]:
    """Whole flights and locations must keep one role, and a frame must appear once."""
    problems, seen, flights, locations = [], set(), {}, {}
    for row in records:
        if row['frame_id'] in seen:
            problems.append(f"duplicate frame {row['frame_id']}")
        seen.add(row['frame_id'])
        flights.setdefault(row['flight_id'], set()).add(row['split'])
        locations.setdefault(row['location_id'], set()).add(row['split'])
    problems += [f'flight {k} in {sorted(v)}' for k, v in flights.items() if len(v) > 1]
    problems += [f'location {k} in {sorted(v)}' for k, v in locations.items() if len(v) > 1]
    return problems


class PolicyInputLoader:
    """What a selector may read. Asking for reference data is an error, not an empty value."""

    def __init__(self, records: dict):
        self.records = records

    def __getitem__(self, frame_id: str) -> dict:
        row = self.records[frame_id]
        return {k: v for k, v in row.items() if k not in REFERENCE_FIELDS}

    def reference(self, frame_id: str):
        raise PermissionError('Reference geometry is evaluator-only and not available to a policy')


def build(contract: dict, root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    gate_path = root / contract['geometry_gate']
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    records = list(frame_records(root, contract, gate))
    problems = validate_splits(records)
    with (output / 'frames.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    splits = {}
    for row in records:
        splits.setdefault(row['split'], set()).add(row['flight_id'])
    splits = {k: sorted(v) for k, v in splits.items()}
    write_json(output / 'splits.json', {'schema_version': 1, 'roles': splits, 'calibration': 'batch suffix _0 within a flight',
                                        'final_test': [], 'final_test_note': 'No sealed location exists yet; HKisland is diagnostic only.'})
    manifest = {'schema_version': 1, 'frames': len(records), 'flights': sorted({r['flight_id'] for r in records}),
                'identity_sha256': sha256_lines(sorted(r['frame_id'] for r in records)), 'leakage_problems': problems,
                'geometry_gate_sha256': sha256_file(gate_path) if gate else None}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build',))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    if args.dry_run:
        print(json.dumps({'cache_root': contract['data']['cache_root']}))
        return
    manifest = build(contract, args.root.resolve(), args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'flights'}, indent=2))
    raise SystemExit(1 if manifest['leakage_problems'] else 0)


if __name__ == '__main__':
    main()
