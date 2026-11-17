"""Offline CNN+GRU distillation from the recorded frame-selection policy.

Uses whole-video holdouts and never edits the reconstruction's active selector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from .train_frame_selector import discover, metrics as scalar_metrics, save_json


class VisualSelector(nn.Module):
    def __init__(self, feature_count=8, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.gru = nn.GRU(64 + feature_count, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, images, features):
        # Process a short clip in batches to bound peak activation memory.
        encoded = torch.cat([self.encoder(chunk).flatten(1)
                             for chunk in images.split(32)], dim=0)
        sequence = torch.cat((encoded, features), dim=-1).unsqueeze(0)
        return self.head(self.gru(sequence)[0]).squeeze(0).squeeze(-1)


def image_cache(sequence, cache_dir):
    video = Path(sequence['video'])
    key = hashlib.sha256(f'{video}:{video.stat().st_size}:{video.stat().st_mtime_ns}'.encode()).hexdigest()[:16]
    path = cache_dir / (key + '.npy')
    if path.is_file():
        images = np.load(path, mmap_mode='r')
        if len(images) == len(sequence['y']):
            return np.asarray(images)
    rows = [json.loads(line) for line in Path(sequence['events']).read_text().splitlines()
            if line.startswith('{')]
    indices = [int(row['source_index']) for row in rows if row.get('event') == 'frame' and 'selected' in row]
    if len(indices) != len(sequence['y']):
        raise RuntimeError(f'Event count changed for {video}')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    frames = []
    wanted = iter(indices)
    target = next(wanted, None)
    index = -1
    try:
        while target is not None:
            if not capture.grab():
                raise RuntimeError(f'Video ended before frame {target}: {video}')
            index += 1
            if index != target:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f'Decode failed at frame {target}: {video}')
            h, w = frame.shape[:2]
            size = (max(1, round(96*w/max(h,w))), max(1, round(96*h/max(h,w))))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            top, left = (96-size[1])//2, (96-size[0])//2
            padded = cv2.copyMakeBorder(small, top, 96-size[1]-top,
                                        left, 96-size[0]-left, cv2.BORDER_CONSTANT)
            frames.append(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            target = next(wanted, None)
    finally:
        capture.release()
    images = np.asarray(frames, dtype=np.uint8)
    np.save(path, images)
    return images


def evaluate(model, group, mean, std, device):
    model.eval()
    truth, probabilities = [], []
    with torch.no_grad():
        for item in group:
            image = torch.as_tensor(item['images'], device=device).permute(0, 3, 1, 2).float() / 255.
            features = torch.as_tensor((item['x']-mean)/std, device=device)
            probabilities.extend(torch.sigmoid(model(image, features)).cpu().numpy().tolist())
            truth.extend(item['y'].tolist())
    t = np.asarray(truth, bool)
    p = np.asarray(probabilities) >= .5
    tp, tn = int((p&t).sum()), int((~p&~t).sum())
    fp, fn = int((p&~t).sum()), int((~p&t).sum())
    return {'frames': len(t), 'accuracy': (tp+tn)/len(t),
            'balanced_accuracy': .5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
            'keep_recall': tp/max(1,tp+fn), 'skip_recall': tn/max(1,tn+fp),
            'always_keep_accuracy': float(t.mean()),
            'confusion': {'tp':tp,'tn':tn,'fp':fp,'fn':fn}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'frame_cache'
    cache.mkdir(exist_ok=True)
    torch.manual_seed(27)
    np.random.seed(27)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = torch.load(args.baseline / 'best.pt', map_location='cpu', weights_only=True)
    mean, std = np.asarray(base['mean'],np.float32), np.asarray(base['std'],np.float32)
    split = json.loads((args.baseline / 'dataset_manifest.json').read_text())['groups']
    available = {s['video']:s for s in discover(args.runs)}
    groups = {}
    started = time.monotonic()
    for name, entries in split.items():
        groups[name] = [available[e['video']] for e in entries]
        for item in groups[name]:
            item['images'] = image_cache(item, cache)
            print(f"cached {name}: {Path(item['video']).parent.name}, {len(item['images'])} frames",flush=True)
    save_json(out/'dataset_manifest.json',{'source_manifest':str(args.baseline/'dataset_manifest.json'),
        'groups':split,'image_size':[96,96],
        'objective':'imitate recorded keep/skip decisions; not geometry outcome training',
        'teacher_forcing':'time since last selected frame uses historical policy decisions',
        'model_deployed':False})
    model = VisualSelector(len(mean)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    labels = np.concatenate([s['y'] for s in groups['train']])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(labels==0).sum()/max(1,labels.sum())],dtype=torch.float32,device=device))
    best, best_epoch, stale = -1., 0, 0
    for epoch in range(1,args.epochs+1):
        model.train()
        order = np.random.permutation(len(groups['train']))
        losses = []
        for i in order:
            item=groups['train'][i]
            image=torch.as_tensor(item['images'],device=device).permute(0,3,1,2).float()/255.
            features=torch.as_tensor((item['x']-mean)/std,device=device)
            target=torch.as_tensor(item['y'],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(image,features),target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            losses.append(float(loss.item()))
        val=evaluate(model,groups['validation'],mean,std,device)
        score=val['balanced_accuracy']
        payload={'model':model.state_dict(),'mean':mean.tolist(),'std':std.tolist(),
                 'epoch':epoch,'validation':val}
        temporary=out/'latest.tmp.pt'
        torch.save(payload,temporary)
        temporary.replace(out/'latest.pt')
        if score>best+1e-4:
            best,best_epoch,stale=score,epoch,0
            temporary=out/'best.tmp.pt'
            torch.save(payload,temporary)
            temporary.replace(out/'best.pt')
        else:
            stale+=1
        save_json(out/'status.json',{'status':'running','epoch':epoch,'best_epoch':best_epoch,
            'train_loss':float(np.mean(losses)),'validation':val,
            'elapsed_seconds':time.monotonic()-started,'model_deployed':False})
        print(f'epoch {epoch}/{args.epochs}: loss {np.mean(losses):.4f}; validation balanced accuracy {score:.3f}; best {best:.3f}',flush=True)
        if stale>=args.patience:
            break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    final={'status':'complete','epochs':epoch,'best_epoch':best_epoch,
           'validation':checkpoint['validation'],
           'test':evaluate(model,groups['test'],mean,std,device),
           'elapsed_seconds':time.monotonic()-started,'model_deployed':False,
           'checkpoint':str(out/'best.pt')}
    save_json(out/'status.json',final)
    print(json.dumps(final,indent=2),flush=True)


if __name__=='__main__':
    main()
