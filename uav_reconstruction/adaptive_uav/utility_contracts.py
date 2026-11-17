"""Versioned vNext contract loading and shared artifact helpers.

Later stages add state, budget, action and label schemas here. Values a stage
needs but the contract lists as unresolved are rejected rather than defaulted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

CONTRACT_SCHEMA_VERSION = 1
GATE_STATUSES = ('VERIFIED', 'QUALIFIED', 'FAILED', 'UNRESOLVED')
REQUIRED_KEYS = ('schema_version', 'contract_id', 'dataset_manifest', 'geometry_gate', 'baseline_manifest',
                 'depth_checkpoint', 'data', 'protected_checkpoints', 'expected_split_examples',
                 'geometry', 'execution', 'unresolved')


def load_contract(path: Path, needs: tuple[str, ...] = ()) -> dict:
    contract = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_KEYS if key not in contract]
    if missing:
        raise ValueError(f'Contract {path} is missing {missing}')
    if contract['schema_version'] != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported contract schema {contract['schema_version']}")
    blocked = [key for key in needs if key in contract['unresolved']]
    if blocked:
        raise ValueError(f'Contract leaves {blocked} unresolved; resolve them before this stage')
    return contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines) -> str:
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Atomic strict JSON: a NaN raises instead of reaching a gate file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, sort_keys=False) + '\n'
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(handle, 'w') as stream:
            stream.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_status(output: Path, stage: str, status: str, gate: str | None, next_stage: str | None,
                 detail: str = '') -> None:
    write_json(Path(output) / 'status.json',
               {'schema_version': 1, 'stage': stage, 'status': status, 'gate': gate,
                'next_eligible_stage': next_stage, 'detail': detail,
                'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')})


THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('VNEXT_THREADS', 4)))  # Native libraries otherwise take every core.

ACTIONS = ('NEW_KEYFRAME', 'TRACK_ONLY', 'SKIP', 'RELOCALIZE')
FIDELITIES = ('F_DEPTH', 'P_LOCAL', 'R_ROLLOUT')
INTERVENTIONS = ('add', 'remove', 'replace', 'source_change', 'relocalize', 'track_only', 'skip')
EXECUTION_STATUSES = ('complete', 'registration_failed', 'over_budget', 'censored')
# Whitelist: only these reach a policy. Reference errors, future poses and final-map
# completeness are absent by construction, and a new feature must be added here on purpose.
QUERY_FEATURES = ('age_s', 'baseline_over_depth', 'rotation_from_last', 'parallax', 'tracked_features',
                  'sharpness', 'map_keyframes', 'map_coverage', 'map_redundancy', 'graph_edges', 'graph_components',
                  'remaining_seconds', 'remaining_depth_calls', 'tracking_ok', 'queue_load', 'best_overlap', 'best_shared_tracks')
TOKEN_FEATURES = ('age_s', 'relative_baseline', 'relative_rotation', 'shared_tracks', 'parallax', 'depth_confidence',
                  'degree', 'is_bridge', 'verified', 'same_component', 'valid')
# One embedding per measured question: the same frame can answer each differently.
HEADS = (('F_DEPTH', 'add'), ('P_LOCAL', 'add'), ('R_ROLLOUT', 'add'), ('F_DEPTH', 'remove'), ('F_DEPTH', 'replace'),
         ('R_ROLLOUT', 'track_only'), ('R_ROLLOUT', 'skip'))
DECISION_HEAD = HEADS.index(('R_ROLLOUT', 'add'))
FORBIDDEN_PREFIXES = ('reference', 'measured', 'future', 'final', 'lidar', 'gt_')


def head_index(fidelity: str, intervention: str) -> int:
    return HEADS.index((fidelity, intervention))


def feature_vector(values: dict, names: tuple) -> list[float]:
    """Order causal features; a key outside the whitelist is an error, not ignored."""
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ValueError(f'Features outside the causal whitelist: {unknown}')
    return [float(values.get(name, 0.)) for name in names]


def assert_causal_names(names) -> None:
    leaked = [name for name in names if name.lower().startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        raise ValueError(f'Reference or future fields cannot be policy inputs: {leaked}')


def state_identity(**fields) -> str:
    """Hash everything a label depends on, so a changed budget or graph is a new state."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:24]


class Budget:
    """Explicit remaining work. Spent cost is never refunded by later removal."""

    def __init__(self, seconds: float, depth_calls: int, matches: int):
        if min(seconds, depth_calls, matches) < 0:
            raise ValueError('Budgets must be nonnegative')
        self.initial = dict(seconds=float(seconds), depth_calls=int(depth_calls), matches=int(matches))
        self.spent = dict(seconds=0., depth_calls=0, matches=0)

    def charge(self, seconds: float = 0., depth_calls: int = 0, matches: int = 0) -> None:
        if min(seconds, depth_calls, matches) < 0:
            raise ValueError('Charges must be nonnegative; past work is not refunded')
        self.spent['seconds'] += seconds
        self.spent['depth_calls'] += depth_calls
        self.spent['matches'] += matches

    def remaining(self, key: str) -> float:
        return max(self.initial[key] - self.spent[key], 0)

    def feasible(self, seconds: float = 0., depth_calls: int = 0, matches: int = 0) -> bool:
        return (seconds <= self.remaining('seconds') and depth_calls <= self.remaining('depth_calls')
                and matches <= self.remaining('matches'))

    def fraction(self) -> float:
        return self.remaining('seconds') / self.initial['seconds'] if self.initial['seconds'] else 0.

    def state(self) -> dict:
        """What the policy may see. Absolute, so tight and loose budgets look different."""
        return {'seconds': self.remaining('seconds'), 'depth_calls': self.remaining('depth_calls')}

    def to_dict(self) -> dict:
        return {'initial': self.initial, 'spent': self.spent}


def validate_record(record: dict) -> None:
    required = ('experiment_id', 'state_id', 'candidate', 'intervention', 'fidelity', 'budget_id', 'flight', 'split',
                'evaluation_protocol_id', 'reference_status', 'execution_status', 'control', 'intervened', 'cost', 'features')
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f'Record missing {missing}')
    if record['fidelity'] not in FIDELITIES or record['intervention'] not in INTERVENTIONS:
        raise ValueError('Unknown fidelity or intervention')
    if record['execution_status'] not in EXECUTION_STATUSES:
        raise ValueError(f"Unknown execution status {record['execution_status']}")
