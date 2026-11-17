"""Remote-corpus visual pretraining. All eligible RGBs are visited in complete passes.

No geometry reward is invented for uncalibrated data. This phase learns semantic
features from frozen DINO and a separately labelled corruption-recognition task.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .frame_degradations import random_degradation, degrade
from .large_selector_model import SelectorVisualEncoder


class Images(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        bgr = cv2.imread(str(self.root/row['path']))
        if bgr is None:
            raise OSError(self.root/row['path'])
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), i


def admitted(root):
    rows, seen = [], set()
    manifest = root/'visual_train.jsonl'
    files = [manifest] if manifest.exists() else [p.parent/'frames.jsonl' for p in sorted((root/'sources').glob('*/complete.json'))]
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['role'] != 'train':
                raise RuntimeError('Held-out frame in training manifest')
            if row['sha256_rgb224'] not in seen:
                rows.append(row)
                seen.add(row['sha256_rgb224'])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full-passes', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--init', type=Path)
    p.add_argument('--sealed-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'complete.json').exists():
        return
    torch.manual_seed(42)
    np.random.seed(42)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    # Do not contend with another user's job.
    import subprocess
    occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError(f'GPU already occupied by PIDs {occupied}')
    model = SelectorVisualEncoder().cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    teacher = torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),
                             'dinov2_vits14', source='local', pretrained=True).cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    mean, std = model.mean, model.std
    log = (a.output/'steps.jsonl').open('a')
    complete_passes, epoch, step = 0, 0, 0
    began = time.time()
    while complete_passes < a.full_passes:
        rows = admitted(a.corpus)
        status_path = a.corpus/'preparation_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state':'waiting'}
        if status['state'] == 'failed':
            # Bootstrap training may continue on complete sources until the remote resume repairs them.
            print('Corpus has failed sources; waiting for preparation resume.', flush=True)
            time.sleep(15)
            continue
        if not rows:
            time.sleep(10)
            continue
        epoch += 1
        full = status['state'] == 'complete' and (a.corpus/'visual_train.jsonl').is_file()
        if a.sealed_only and not (a.corpus/'corpus_seal.json').exists():
            time.sleep(10)
            continue
        dataset = Images(a.corpus, rows)
        loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                            pin_memory=True, drop_last=False, prefetch_factor=2)
        visited, sums = set(), []
        model.train()
        for images, indices in loader:
            clean = images.cuda(non_blocking=True).float()/255
            kind, strength = random_degradation(len(clean), .5, clean.device)
            damaged = degrade(clean, kind, strength)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    target = F.normalize(teacher((clean-mean)/std).float(), dim=-1)
                embedding, corruption = model(damaged)
                semantic = (1-(embedding.float()*target).sum(-1)).mean()
                fault = F.cross_entropy(corruption.float(), kind+1)
                loss = semantic+.1*fault
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pretraining loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite pretraining gradient')
            optimizer.step()
            step += 1
            visited.update(indices.tolist())
            sums.append(float(loss.detach()))
            if step % 25 == 0 or step == 1:
                row = {'phase':'visual_pretraining', 'epoch':epoch, 'full_corpus_pass':full,
                       'step':step, 'loss':float(loss), 'semantic':float(semantic),
                       'corruption_ce':float(fault), 'gradient_norm':float(norm),
                       'frames_in_pass':len(rows), 'visited':len(visited), 'elapsed_seconds':time.time()-began}
                log.write(json.dumps(row)+'\n');log.flush()
                print(json.dumps(row), flush=True)
        if len(visited) != len(rows):
            raise RuntimeError('Full-pass coverage failure')
        complete_passes += int(full)
        manifest_hash = hashlib.sha256('\n'.join(r['sha256_rgb224'] for r in rows).encode()).hexdigest()
        state = {'model':model.state_dict(), 'epoch':epoch, 'step':step, 'complete_passes':complete_passes,
                 'training_frames':len(rows), 'manifest_sha256':manifest_hash,
                 'parameters':sum(p.numel() for p in model.parameters()),
                 'scope':'RGB representation and corruption pretraining, not fusion-utility supervision'}
        temp = a.output/'latest.tmp'
        torch.save(state, temp)
        temp.replace(a.output/'latest.pt')
        print(json.dumps({'pass_complete':epoch,'full':full,'frames':len(rows),
                          'complete_passes':complete_passes,'mean_loss':float(np.mean(sums))}),flush=True)
    (a.output/'complete.json').write_text(json.dumps({k:v for k,v in state.items() if k!='model'},indent=2))
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
