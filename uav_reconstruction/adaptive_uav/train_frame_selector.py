"""Train an offline GRU baseline from deduplicated causal selector event logs.

This is policy distillation, not a learned geometry oracle. It is never used by
the reconstruction pipeline without separate held-out geometric validation.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn


FEATURES = ('log_blur', 'motion_pixels', 'raw_motion_pixels',
            'motion_threshold_pixels', 'rotation_compensated', 'imu_valid',
            'gyro_max_rad_s', 'seconds_since_selected')


def discover(runs):
    by_video = {}
    for path in sorted(runs.rglob('online_events.jsonl')):
        manifest = path.parent.parent / 'pipeline_manifest.json'
        if not manifest.is_file():
            continue
        try:
            video = Path(json.loads(manifest.read_text())['inputs']['video']['path']).resolve()
            if not video.is_file():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
            frames = [r for r in rows if r.get('event') == 'frame' and 'selected' in r]
            if len(frames) < 30:
                continue
            # One record per input video prevents repeated reconstruction runs
            # of the same footage from leaking across train and validation.
            key = (str(video), video.stat().st_size)
            candidate = (len(frames), str(path), frames)
            if key not in by_video or candidate[:2] > by_video[key][:2]:
                by_video[key] = candidate
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    sequences = []
    for (video, _), (_, path, frames) in sorted(by_video.items()):
        features, labels = [], []
        previous_selected = None
        for row in frames:
            stamp = float(row['source_timestamp_s'])
            imu = row.get('imu') or {}
            values = [np.log1p(max(0., float(row.get('blur', 0.)))),
                      float(row.get('motion_pixels', 0.)),
                      float(row.get('raw_motion_pixels', 0.)),
                      float(row.get('motion_threshold_pixels', 0.)),
                      float(bool(row.get('rotation_compensated', False))),
                      float(bool(imu.get('valid', False))),
                      float(imu.get('gyro_max_rad_s', 0.) or 0.),
                      min(10., stamp - previous_selected) if previous_selected is not None else 10.]
            features.append(values)
            selected = int(bool(row['selected']))
            labels.append(selected)
            if selected:
                previous_selected = stamp
        x = np.asarray(features, np.float32)
        y = np.asarray(labels, np.float32)
        if np.isfinite(x).all() and 0 < y.sum() < len(y):
            sequences.append({'video': video, 'events': path, 'x': x, 'y': y})
    return sequences


class Selector(nn.Module):
    def __init__(self, feature_count, hidden=48):
        super().__init__()
        self.gru = nn.GRU(feature_count, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def metrics(model, sequences, mean, std, device):
    model.eval()
    targets, probabilities = [], []
    with torch.no_grad():
        for sequence in sequences:
            x = torch.as_tensor((sequence['x'] - mean) / std, device=device).unsqueeze(0)
            probabilities.extend(torch.sigmoid(model(x)).squeeze(0).cpu().numpy().tolist())
            targets.extend(sequence['y'].tolist())
    truth = np.asarray(targets, bool)
    pred = np.asarray(probabilities) >= .5
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {'frames': len(truth), 'accuracy': float((tp + tn) / max(1, len(truth))),
            'balanced_accuracy': float(.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))),
            'keep_recall': float(tp / max(1, tp + fn)),
            'skip_recall': float(tn / max(1, tn + fp)),
            'always_keep_accuracy': float(truth.mean()) if len(truth) else 0.,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}}


def save_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=18)
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sequences = discover(args.runs)
    if len(sequences) < 5:
        raise RuntimeError(f'Need five distinct labelled videos; found {len(sequences)}')
    # Fixed group split; no frames from a video appear in two partitions.
    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    validation = sequences[-2:-1]
    test = sequences[-1:]
    train = sequences[:-2]
    if not any((s['y'] == 0).any() for s in validation):
        validation = next(([s] for s in sequences[:-1] if (s['y'] == 0).any()), validation)
        held_out = {s['video'] for s in validation + test}
        train = [s for s in sequences if s['video'] not in held_out]
    features = np.concatenate([s['x'] for s in train])
    mean = features.mean(0)
    std = features.std(0)
    std[std < 1e-4] = 1.
    counts = {'train': train, 'validation': validation, 'test': test}
    manifest = {'objective': 'imitate existing keep/skip decisions; no geometry outcome labels',
                'features': FEATURES, 'device': device, 'seed': args.seed,
                'groups': {name: [{'video': s['video'], 'events': s['events'],
                                    'frames': len(s['y']), 'kept': int(s['y'].sum())} for s in group]
                           for name, group in counts.items()},
                'limitations': ['Pseudo-labels inherit the current selector policy.',
                                'Only 11 distinct videos; no learned model is deployed.',
                                'seconds_since_selected is teacher-forced during offline training.']}
    save_json(out / 'dataset_manifest.json', manifest)
    model = Selector(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    labels = np.concatenate([s['y'] for s in train])
    pos_weight = torch.tensor([(labels == 0).sum() / max(1, labels.sum())],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_score, best_epoch, stale = -1., 0, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train)
        rng.shuffle(order)
        losses = []
        for sequence in order:
            x = torch.as_tensor((sequence['x'] - mean) / std, device=device).unsqueeze(0)
            y = torch.as_tensor(sequence['y'], device=device).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.item()))
        val = metrics(model, validation, mean, std, device)
        record = {'status': 'running', 'epoch': epoch, 'max_epochs': args.epochs,
                  'train_loss': float(np.mean(losses)), 'validation': val,
                  'elapsed_seconds': time.monotonic() - started,
                  'best_epoch': best_epoch, 'best_balanced_accuracy': best_score}
        payload = {'model': model.state_dict(), 'features': FEATURES,
                   'mean': mean.tolist(), 'std': std.tolist(),
                   'epoch': epoch, 'validation': val, 'manifest': manifest}
        temporary = out / 'latest.tmp.pt'
        torch.save(payload, temporary)
        temporary.replace(out / 'latest.pt')
        if val['balanced_accuracy'] > best_score + 1e-4:
            best_score, best_epoch, stale = val['balanced_accuracy'], epoch, 0
            temporary = out / 'best.tmp.pt'
            torch.save(payload, temporary)
            temporary.replace(out / 'best.pt')
        else:
            stale += 1
        save_json(out / 'status.json', record)
        print(f"epoch {epoch}: loss {record['train_loss']:.4f}; val balanced accuracy {val['balanced_accuracy']:.3f}; best {best_score:.3f}", flush=True)
        if stale >= args.patience:
            break
    best = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best['model'])
    test_result = metrics(model, test, mean, std, device)
    final = {'status': 'complete', 'epochs': epoch, 'best_epoch': best_epoch,
             'validation': best['validation'], 'test': test_result,
             'elapsed_seconds': time.monotonic() - started,
             'model_deployed': False, 'checkpoint': str(out / 'best.pt')}
    save_json(out / 'status.json', final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
