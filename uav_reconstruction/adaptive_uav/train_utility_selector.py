"""Stage 5B: train memory_utility_v1 from measured counterfactual labels.

Ranking first, as the proposal says: inside one state and budget, the frame that really helped
must score higher. Regression, cost, calibrated uncertainty, keep/skip and source-pair terms join
it where labels exist. Early stopping uses other flights, and every score is printed beside the
simple "least covered view" rule, because a network that cannot beat it has no reason to exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .build_utility_dataset import read_labels
from .graph_selector import ACTIONS, MemoryUtilitySelector
from .selector_policy import collate
from .utility_contracts import DECISION_HEAD, HEADS, QUERY_FEATURES, TOKEN_FEATURES, head_index, load_contract, sha256_file, sha256_lines, write_json, write_status

TARGETS = ('delta_coverage', 'delta_surface', 'delta_connectivity', 'utility')
ADD_HEADS = tuple(i for i, (_, intervention) in enumerate(HEADS) if intervention == 'add')
COVERAGE = QUERY_FEATURES.index('map_coverage')


def load_samples(folder: Path, tie_margin=.002) -> tuple:
    samples = {}
    actions = {}
    for row in read_labels(folder):
        if row['execution_status'] == 'censored':
            continue
        with np.load(folder / 'features' / Path(row['features']).name) as data:
            arrays = {k: data[k] for k in ('query', 'visual', 'tokens', 'token_visual')}
        samples[row['experiment_id']] = {**arrays, 'question': head_index(row['fidelity'], row['intervention']), 'split': row['split'],
                                         'clip': row['clip'], 'flight': row['flight'], 'raw': np.float32([row['targets'][k] for k in TARGETS]),
                                         'cost': float(np.sign(row['targets']['delta_cost'])*np.log1p(abs(row['targets']['delta_cost'])))}
        if row.get('schema_version',0)>=3 and row['fidelity']=='R_ROLLOUT':
            key=(row['state_id'],row['candidate'],row['budget_id'])
            actions.setdefault(key,[]).append(row)
            samples[row['experiment_id']]['action_target'] = None
    for records in actions.values():
        if {r['action'] for r in records}!={'SKIP','TRACK_ONLY','NEW_KEYFRAME'}:
            continue  # A censored/unmeasured alternative cannot support a best-action label.
        feasible = [r for r in records if r['execution_status']=='complete']
        ranked = sorted(feasible,key=lambda r:r['targets']['utility'],reverse=True)
        if len(ranked)>=2 and ranked[0]['targets']['utility']-ranked[1]['targets']['utility']>tie_margin:
            target = ACTIONS.index(ranked[0]['action'])
            for r in records:
                samples[r['experiment_id']]['action_target'] = target
    groups = [json.loads(line) for line in (folder / 'ranking_groups.jsonl').read_text().splitlines()]
    return samples, groups


def load_sources(folder: Path) -> list:
    file = folder / 'source_labels.jsonl'
    if not file.exists():
        return []
    rows = []
    for line in file.read_text().splitlines():
        row = json.loads(line)
        with np.load(folder / 'source_features' / Path(row['features']).name) as data:
            rows.append({**row, **{k: data[k] for k in ('query', 'visual', 'tokens', 'token_visual')}, 'question': DECISION_HEAD})
    return rows


def split_groups(groups: list, samples: dict) -> tuple:
    """Validation flights if labelled. Otherwise hold out a whole flight, never a neighbouring clip."""
    train = [g for g in groups if g['split'] == 'train']
    held = [g for g in groups if g['split'] == 'validation']
    if held:
        return train, held, 'validation flights'
    flights = sorted({samples[g['members'][0]]['flight'] for g in groups})
    if len(flights) < 2:
        return groups, [], 'none'
    chosen = set(flights[-max(1, len(flights) // 4):])
    return ([g for g in groups if samples[g['members'][0]]['flight'] not in chosen],
            [g for g in groups if samples[g['members'][0]]['flight'] in chosen], f'held-out flights {sorted(chosen)}')


def normalize(samples: dict, train_groups: list) -> tuple:
    """Per question, from training data only: each measures a different scale."""
    mean, std = np.zeros((len(HEADS), len(TARGETS))), np.ones((len(HEADS), len(TARGETS)))
    names = {m for g in train_groups for m in g['members']}
    for head in range(len(HEADS)):
        values = np.stack([samples[n]['raw'] for n in names if samples[n]['question'] == head] or [np.zeros(len(TARGETS))])
        mean[head], std[head] = values.mean(0), np.maximum(values.std(0), 1e-3)
    for row in samples.values():
        row['targets'] = (row['raw'] - mean[row['question']]) / std[row['question']]
        row['rank_mean'], row['rank_scale'] = float(mean[row['question'],3]),float(std[row['question'],3])
    return mean, std


def loss_fn(model, samples: dict, members: list, pairs: list, weights: dict) -> tuple:
    rows = [samples[name] for name in members]
    out = model(*collate(rows, next(model.parameters()).device))
    predicted = out['utility'][:, 0]
    target = torch.as_tensor(np.stack([r['targets'] for r in rows]), device=predicted.device, dtype=predicted.dtype)
    regression = F.smooth_l1_loss(predicted, target)
    cost = F.smooth_l1_loss(out['cost'][:, 0], torch.as_tensor([r['cost'] for r in rows], device=predicted.device, dtype=predicted.dtype))
    error = (predicted[:, 3] - target[:, 3]).detach()
    log_variance = out['log_variance'][:, 0].clamp(-6, 6)
    uncertainty = (.5 * (torch.exp(-log_variance) * error ** 2 + log_variance)).mean()
    where = {name: i for i, name in enumerate(members)}
    rank = predicted.sum() * 0
    if pairs:
        better = torch.as_tensor([where[a] for a, _ in pairs], device=predicted.device)
        worse = torch.as_tensor([where[b] for _, b in pairs], device=predicted.device)
        scores = predicted[:,3]
        if len({r['question'] for r in rows})>1:
            scales=torch.as_tensor([r.get('rank_scale',1.) for r in rows],device=predicted.device)
            offsets=torch.as_tensor([r.get('rank_mean',0.) for r in rows],device=predicted.device)
            scores=scores*scales+offsets
        rank = F.softplus(-(scores[better] - scores[worse])).mean()
    action = predicted.sum() * 0
    verified = [i for i,r in enumerate(rows) if r.get('action_target') is not None or
                ('action_target' not in r and r['question'] in ADD_HEADS)]
    if verified:
        labels = torch.as_tensor([rows[i]['action_target'] if 'action_target' in rows[i] else
                                 ACTIONS.index('NEW_KEYFRAME') if rows[i]['raw'][3]>0 else ACTIONS.index('TRACK_ONLY')
                                 for i in verified],device=predicted.device)
        action = F.cross_entropy(out['action'][verified, 0], labels)
    total = (weights['rank'] * rank + weights['regression'] * regression + weights['cost'] * cost +
             weights['uncertainty'] * uncertainty + weights['action'] * action)
    return total, {'rank': float(rank.detach()), 'regression': float(regression.detach()), 'cost': float(cost.detach()),
                   'uncertainty': float(uncertainty.detach()), 'action': float(action.detach()), 'pairs': len(pairs)}


def source_loss(model, row: dict) -> torch.Tensor:
    """Lower depth error must mean a higher pair score, pair against pair."""
    output = model(*collate([row], next(model.parameters()).device), pairs=True)['pair'][0, 0]
    scores = torch.stack([output[a, b] for a, b, _ in row['pairs']])
    errors = torch.as_tensor([e for _, _, e in row['pairs']], device=scores.device, dtype=scores.dtype)
    gap = (scores[None] - scores[:, None])
    order = errors[None] < errors[:, None]
    return F.softplus(-gap[order]).mean() if order.any() else scores.sum() * 0


@torch.no_grad()
def evaluate(model, samples: dict, groups: list, head: int | None = DECISION_HEAD) -> dict:
    model.eval()
    correct, baseline, total, baseline_pairs, errors = 0, 0, 0, 0, []
    for group in groups:
        rows = [samples[m] for m in group['members']]
        if head is not None and not any(r['question'] == head for r in rows):
            continue
        predicted = model(*collate(rows, next(model.parameters()).device))['utility'][:, 0]
        scores=predicted[:,3]
        if len({r['question'] for r in rows})>1:
            scores=scores*torch.as_tensor([r.get('rank_scale',1.) for r in rows],device=scores.device)+\
                   torch.as_tensor([r.get('rank_mean',0.) for r in rows],device=scores.device)
        where = {name: i for i, name in enumerate(group['members'])}
        for a, b in group['pairs']:
            i, j = where[a], where[b]
            correct += int(scores[i] > scores[j])
            # Action alternatives share the same candidate/map features. A tied
            # coverage score provides no ordering and must not be called a loss.
            a_coverage, b_coverage = rows[i]['query'][COVERAGE], rows[j]['query'][COVERAGE]
            if not np.isclose(a_coverage, b_coverage, rtol=0, atol=1e-7):
                baseline += int(a_coverage < b_coverage)
                baseline_pairs += 1
            total += 1
        errors += (predicted[:, 3].cpu().numpy() - np.array([r['targets'][3] for r in rows])).tolist()
    return {'pair_accuracy': correct / total if total else None, 'least_covered_rule_accuracy': baseline / baseline_pairs if baseline_pairs else None,
            'least_covered_rule_pairs': baseline_pairs, 'pairs': total,
            'utility_mae_normalized': float(np.abs(errors).mean()) if errors else None}


@torch.no_grad()
def uncertainty_limit(model, samples: dict, groups: list) -> tuple:
    """Set from held-out decisions: the runtime falls back on the least certain fifth."""
    values, errors = [], []
    for group in groups:
        rows = [samples[m] for m in group['members']]
        out = model(*collate(rows, next(model.parameters()).device))
        values += out['log_variance'][:, 0].cpu().tolist()
        errors += (out['utility'][:, 0, 3].cpu().numpy() - np.array([r['targets'][3] for r in rows])).tolist()
    if not values:
        return 6., None
    correlation = float(np.corrcoef(np.exp(np.array(values) / 2), np.abs(errors))[0, 1]) if len(values) > 2 and np.std(values) > 0 else None
    return float(np.percentile(values, 80)), correlation


def save_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def training_identity(folder, contract, descriptor):
    """Resume only the same supervision, features, settings and implementation."""
    files = sorted(p for p in folder.rglob('*') if p.is_file() and
                   (p.name.startswith('labels') or p.name in ('source_labels.jsonl', 'ranking_groups.jsonl', 'quality_audit.json')
                    or p.parent.name in ('features', 'source_features')))
    source = Path(__file__).parent
    identity = {'inputs': [(str(p.relative_to(folder)), sha256_file(p)) for p in files],
                'selector': contract['selector'], 'utility': contract.get('utility', {}), 'descriptor': descriptor,
                'code': {n: sha256_file(source / n) for n in ('train_utility_selector.py', 'graph_selector.py', 'selector_policy.py', 'utility_contracts.py')}}
    return sha256_lines([json.dumps(identity, sort_keys=True)])


def training_sources(sources, groups, samples):
    flights = {samples[m]['flight'] for g in groups for m in g['members']}
    return [row for row in sources if row['split'] == 'train' and row['flight'] in flights]


def train(contract, labels: Path, output: Path, device, epochs, allow_diagnostic: bool, resume: bool, descriptor: str):
    audit = json.loads((labels / 'quality_audit.json').read_text())
    if not audit['trusted_reference'] and not allow_diagnostic:
        raise SystemExit('Labels rest on an unresolved reference. Pass --allow-diagnostic for a non-promotable run.')
    cfg = contract['selector']
    labels_hash = sha256_lines(sorted(f.read_text() for f in labels.glob('labels*.jsonl')))
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    samples, groups = load_samples(labels,contract.get('utility',{}).get('tie_margin',.002))
    if not samples:
        raise SystemExit('No usable selector samples')
    sources = load_sources(labels)
    training, held, holdout = split_groups([g for g in groups if g['members']], samples)
    mean, std = normalize(samples, training)
    if not training:
        raise SystemExit('No selector training groups')
    source_train = training_sources(sources, training, samples)
    identity = training_identity(labels, contract, descriptor)
    config = dict(visual_dim=samples[next(iter(samples))]['visual'].size, query_dim=len(QUERY_FEATURES), token_dim=len(TOKEN_FEATURES),
                  hidden=cfg['hidden_dim'], heads=cfg['heads'], layers=cfg['layers'], questions=len(HEADS))
    model = MemoryUtilitySelector(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    output.mkdir(parents=True, exist_ok=True)
    start, steps, best, patience, skipped = 0, 0, None, 0, 0
    if resume and (output / 'latest.pt').exists():
        saved = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if saved.get('training_identity') != identity:
            raise SystemExit('Selector data, features, config or code changed; use a fresh experiment instead of resuming')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, steps, best, patience = saved['epoch'] + 1, saved['optimizer_steps'], saved['best'], saved['patience']
        torch.set_rng_state(saved['rng']['torch']); np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])

    def payload(epoch, metrics, limit, correlation):
        return {'architecture_id': MemoryUtilitySelector.architecture_id, 'model_config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'optimizer_steps': steps, 'best': best, 'patience': patience, 'config': cfg, 'metrics': metrics, 'labels_sha256': labels_hash,
                'descriptor': descriptor, 'training_identity': identity, 'diagnostic_labels': not audit['trusted_reference'], 'decision_head': DECISION_HEAD, 'target_mean': mean.tolist(), 'target_std': std.tolist(),
                'causal_online_evidence':audit.get('causal_online_evidence',False),
                'log_variance_limit': limit, 'uncertainty_error_correlation': correlation, 'source_head_trained': bool(source_train),
                'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}}

    write_json(output / 'config.json', {'selector': cfg, 'labels_sha256': labels_hash, 'model': config, 'parameters': sum(p.numel() for p in model.parameters()),
                                        'device': str(device), 'holdout': holdout, 'descriptor': descriptor, 'groups': {'train': len(training), 'held': len(held)}, 'source_groups': len(source_train)})
    write_status(output, 'stage5_selector', 'running', None, None)
    log, weights, curve, reached = (output / 'optimizer_steps.jsonl').open('a'), cfg['loss_weights'], [], set()
    final_epoch = start
    for epoch in range(start, epochs or cfg['epochs']):
        final_epoch = epoch
        model.train()
        order = list(training)
        random.shuffle(order)
        totals = []
        for group in order:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_fn(model, samples, group['members'], group['pairs'], weights)
            if source_train and weights['source']:
                loss = loss + weights['source'] * source_loss(model, random.choice(source_train))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            if not torch.isfinite(norm):
                skipped += 1
                continue
            optimizer.step()
            steps += 1
            totals.append(float(loss.detach()))
            reached |= {name.split('.')[0] for name, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
            log.write(json.dumps({'step': steps, 'epoch': epoch, 'loss': totals[-1], 'norm': float(norm), **parts}) + '\n')
        metrics = {'epoch': epoch, 'train_loss': float(np.mean(totals)) if totals else None, 'train': evaluate(model, samples, training),
                   'held': evaluate(model, samples, held), 'skipped_updates': skipped}
        curve.append(metrics)
        print(json.dumps(metrics), flush=True)
        scored = metrics['held'] if metrics['held']['pair_accuracy'] is not None else metrics['train']
        score = scored['pair_accuracy'] or 0.
        limit, correlation = uncertainty_limit(model, samples, held or training)
        if best is None or score > best:
            best, patience = score, 0
            save_atomic(output / 'best.pt', payload(epoch, metrics, limit, correlation))
        else:
            patience += 1
        save_atomic(output / 'latest.pt', payload(epoch, metrics, limit, correlation))
        if epoch + 1 >= cfg.get('min_epochs', 0) and patience >= cfg['patience']:
            break
    log.close()
    selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    chosen = selected['metrics']['held']
    comparable = chosen.get('least_covered_rule_pairs', 0) == chosen.get('pairs', 0) and chosen.get('pairs', 0) > 0
    beat = bool(comparable and chosen.get('pair_accuracy') is not None and chosen.get('least_covered_rule_accuracy') is not None
                and chosen['pair_accuracy'] > chosen['least_covered_rule_accuracy'])
    report = {'curve': curve, 'best_score': best, 'optimizer_steps': steps, 'skipped_updates': skipped, 'gradient_reached': sorted(reached), 'holdout': holdout,
              'beats_least_covered_rule_on_held_data': beat, 'least_covered_comparison_applicable': comparable,
              'selected_epoch': selected['epoch'], 'selected_held': chosen, 'training_identity': identity, 'source_head_trained': bool(source_train),
              'action_head_scope':'Three-way where paired sequential labels have a verified non-tied winner; legacy labels remain keep/track only',
              'descriptor': descriptor, 'diagnostic_labels': not audit['trusted_reference'], 'last_epoch': final_epoch, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(output / 'validation.json', report)
    write_status(output, 'stage5_selector', 'complete', 'DIAGNOSTIC' if report['diagnostic_labels'] else 'TRAINED', 'stage5_finetune')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=Path('runs/vnext_utility_v1'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--allow-diagnostic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config, ('utility.quality_protocol', 'utility.scalarization'))
    if args.dry_run:
        print(json.dumps({'parameters': sum(p.numel() for p in MemoryUtilitySelector().parameters()), 'heads': HEADS}))
        return
    generation = args.labels / 'generation_report.json'
    descriptor = json.loads(generation.read_text()).get('descriptor', 'thumbnail384') if generation.exists() else 'thumbnail384'
    report = train(contract, args.labels, args.output, torch.device(args.device), args.epochs, args.allow_diagnostic, args.resume, descriptor)
    print(json.dumps({k: v for k, v in report.items() if k != 'curve'}, indent=2))


if __name__ == '__main__':
    main()
