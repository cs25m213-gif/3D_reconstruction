"""Stage 4: turn measured branch pairs into ranking groups, and audit whether they deserve trust."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .utility_contracts import load_contract, validate_record, write_json, write_status

ALLOWED_SPLITS = ('train', 'validation')


def read_labels(folder: Path) -> list:
    rows = []
    for file in sorted(folder.glob('labels*.jsonl')):
        rows += [json.loads(line) for line in file.read_text().splitlines()]
    return rows


def group_key(row: dict) -> str:
    if row.get('schema_version', 0) >= 3 and row['fidelity'] == 'R_ROLLOUT':
        return '|'.join((row['state_id'], row['budget_id'], row['fidelity'], 'actions',
                         str(row['candidate']), row['evaluation_protocol_id']))
    return '|'.join((row['state_id'], row['budget_id'], row['fidelity'], row['intervention'], row['evaluation_protocol_id']))


def ranking_pairs(rows: list, margin: float) -> list:
    """Only inside one group, and never across a difference smaller than measured noise."""
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            gap = a['targets']['utility'] - b['targets']['utility']
            if abs(gap) > margin:
                pairs.append([a['experiment_id'], b['experiment_id']] if gap > 0 else [b['experiment_id'], a['experiment_id']])
    return pairs


def agreement(pairs: list) -> dict:
    if len(pairs) < 4:
        return {'pairs': len(pairs), 'spearman': None, 'sign_agreement': None}
    a, b = map(np.asarray, zip(*pairs))
    rho = spearmanr(a, b).statistic if np.ptp(a) and np.ptp(b) else float('nan')
    return {'pairs': len(pairs), 'spearman': None if np.isnan(rho) else float(rho), 'sign_agreement': float(np.mean(np.sign(a) == np.sign(b)))}


def audit(records: list) -> dict:
    """The first questions of the whole project, answered from measurements."""
    adds = [r for r in records if r['intervention'] == 'add']
    by_key = collections.defaultdict(dict)
    for row in adds:
        by_key[(row['state_id'], row['candidate'], row['budget_id'])][row['fidelity']] = row['targets']['utility']
    proxy = {other: agreement([(v['F_DEPTH'], v[other]) for v in by_key.values() if 'F_DEPTH' in v and other in v]) for other in ('P_LOCAL', 'R_ROLLOUT')}
    frames = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in adds:
        if row['fidelity'] == 'F_DEPTH':
            # Keyed by map, not state: the same map on another budget is not a different map.
            frames[(row['clip'], row['candidate'])][tuple(row.get('prefix', ()))].append(row['targets']['delta_surface'])
    spreads = [np.ptp([np.mean(v) for v in states.values()]) for states in frames.values() if len(states) > 1]
    budgets = collections.defaultdict(dict)
    for row in adds:
        if row['fidelity'] == 'R_ROLLOUT':
            budgets[(row['state_id'], row['candidate'])][row['budget_id']] = row['targets']['utility']
    compared = [tuple(v.values()) for v in budgets.values() if len(v) == 2]
    failures = collections.defaultdict(list)
    for row in adds:
        if row['fidelity'] == 'P_LOCAL':
            failures[row['flight']].append(row['execution_status'] == 'registration_failed')
    return {'proxy_fidelity': proxy, 'same_frame_multiple_maps': len(spreads), 'state_dependence_median_spread': float(np.median(spreads)) if spreads else None,
            'budget_changes_sign_fraction': float(np.mean([np.sign(a) != np.sign(b) for a, b in compared])) if compared else None,
            'outcomes': dict(collections.Counter(r['execution_status'] for r in records)),
            'registration_failed_by_flight': {flight: float(np.mean(v)) for flight, v in sorted(failures.items())},
            'interventions': dict(collections.Counter(r['intervention'] for r in records)),
            'utility_sign': dict(collections.Counter(str(int(np.sign(r['targets']['utility']))) for r in records))}


def build(records: list, margin: float) -> tuple:
    groups = collections.defaultdict(list)
    for row in records:
        validate_record(row)
        if row['execution_status'] != 'censored':
            groups[group_key(row)].append(row)
    table = [{'group': key, 'members': [r['experiment_id'] for r in rows], 'pairs': ranking_pairs(rows, margin), 'split': rows[0]['split'],
              'fidelity': rows[0]['fidelity'], 'intervention': rows[0]['intervention']} for key, rows in sorted(groups.items())]
    return table, audit(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    records = read_labels(args.output)
    if args.dry_run:
        print(json.dumps({'records': len(records)}))
        return
    if not records:
        write_status(args.output, 'stage4_dataset', 'blocked', 'NO_LABELS', None, 'No flight had an eligible reference.')
        raise SystemExit('No labels: pass the geometry gate first, or generate diagnostic labels on purpose.')
    leaked = sorted({r['flight'] for r in records if r['split'] not in ALLOWED_SPLITS})
    if leaked:
        raise SystemExit(f'Labels may only come from train and validation flights: {leaked}')
    margin = contract['utility']['tie_margin']
    table, report = build(records, margin)
    with (args.output / 'ranking_groups.jsonl').open('w') as stream:
        for row in table:
            stream.write(json.dumps(row) + '\n')
    trusted = bool(records) and all(r['reference_status'] in ('VERIFIED', 'QUALIFIED')
                                   and not r.get('detail',{}).get('diagnostic_labels',False) for r in records)
    generated=args.output/'generation_report.json'
    if generated.exists() and json.loads(generated.read_text()).get('diagnostic_labels',False):
        trusted=False
    online = trusted and all(r.get('detail', {}).get('causal_online_evidence') is True for r in records)
    report.update(records=len(records), groups=len(table), pairs=sum(len(r['pairs']) for r in table), tie_margin=margin,
                  trusted_reference=trusted, causal_online_evidence=online,
                  scope='Offline development unless causal pose/anchor evidence is separately established')
    write_json(args.output / 'quality_audit.json', report)
    write_status(args.output, 'stage4_dataset', 'complete', 'TRUSTED' if trusted else 'DIAGNOSTIC', 'stage5_selector' if trusted else None)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
