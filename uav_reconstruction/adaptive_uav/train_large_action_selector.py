"""Train a larger ADD/SWAP selector after remote RGB pretraining and measured labels.

All derived samples, descriptors, checkpoints and logs remain in --corpus (a remote
mount). No labels are inferred from corruption severity. AMtown03 selects epochs;
AMvalley never supplies training gradients. The compact head is a matched control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import bin_memory_selector as bm
from .large_selector_model import LargeActionSelector, SelectorVisualEncoder


def save(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=80)
    a = p.parse_args()
    root, output = a.corpus, a.corpus/'selector'
    output.mkdir(exist_ok=True)
    while not (root/'visual_final/complete.json').exists():
        print('Waiting for two passes over the sealed complete remote RGB corpus.',flush=True)
        time.sleep(30)
    if not (root/'corpus_seal.json').exists() or json.loads((root/'utility_status.json').read_text())['state']!='complete':
        raise RuntimeError('Both preparation stages must be complete')
    if (output/'complete.json').exists():
        return
    device = torch.device('cuda')
    torch.manual_seed(20261114)
    torch.set_num_threads(4)
    encoder = SelectorVisualEncoder().to(device)
    state = torch.load(root/'visual_final/latest.pt',map_location='cpu',weights_only=False)
    encoder.load_state_dict(state['model'])
    encoder.eval().requires_grad_(False)
    encoder_hash = hashlib.sha256((root/'visual_final/latest.pt').read_bytes()).hexdigest()
    bank = root/'utility_features'
    bank.mkdir(exist_ok=True)
    groups={'inner':[], 'holdout':[], 'validation':[], 'full':[]}
    ledger=[]
    for path in sorted((root/'utility').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['split'] not in ('train','validation'):
            raise RuntimeError('Unexpected split')
        with np.load(path) as d:
            images=d['rgb']
            parts=[]
            for start in range(0,len(images),64):
                x=torch.from_numpy(images[start:start+64].transpose(0,3,1,2).copy()).to(device).float()/255
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    z,_=encoder(x)
                parts.append(z.float().cpu().numpy())
            z=np.concatenate(parts)
            np.save(bank/(path.stem+'.npy'),z)
            t,occ=d['t'],d['a']; has=(occ>=0).astype(np.float32)
            x=np.concatenate([d['x'],d['quality'][t],d['quality'][np.maximum(occ,0)]*has[:,None]],1).astype(np.float32)
            row=(z[t],z[np.maximum(occ,0)],has,x,d['y'])
        key='validation' if meta['split']=='validation' else ('holdout' if meta['flight']=='AMtown03' else 'inner')
        groups[key].append(row)
        if meta['split']=='train':groups['full'].append(row)
        ledger.append({'file':path.name,'rows':len(t),'group':key,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (output/'label_manifest.json').write_text(json.dumps(ledger,indent=2))
    tensors={k:tuple(torch.from_numpy(np.concatenate([r[i] for r in rows])).to(device) for i in range(5))
             for k,rows in groups.items()}
    if any(len(tensors[k][4])==0 for k in tensors):raise RuntimeError('Empty split')
    width=tensors['inner'][3].shape[1]
    models={'large':LargeActionSelector(width), 'compact':bm.BinScorer(384,width)}
    log=(output/'steps.jsonl').open('a')
    began=time.time()

    def fit(kind, train, epochs, holdout=None):
        torch.manual_seed(42)
        model=(LargeActionSelector(width) if kind=='large' else bm.BinScorer(384,width)).to(device)
        model.mu.copy_(train[3].mean(0));model.sd.copy_(train[3].std(0).clamp_min(.001))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4 if kind=='large' else 3e-4,weight_decay=.02)
        best,best_epoch,best_state=float('inf'),epochs,None

        def loss(data, index=None):
            candidate,occupant,has,x,y=data if index is None else [t[index] for t in data]
            pred=model(candidate,occupant,has,x)
            terms=F.smooth_l1_loss(pred,(20*y).clamp(-4,4),reduction='none')+.2*F.binary_cross_entropy_with_logits(pred,(y>.001).float(),reduction='none')
            # Keep the rarer ADD task visible; separate heads still see their correct states.
            parts=[terms[mask].mean() for mask in (has<.5,has>=.5) if mask.any()]
            return torch.stack(parts).mean()

        for epoch in range(1,epochs+1):
            model.train();values=[]
            for idx in torch.randperm(len(train[4]),device=device).split(128):
                optimizer.zero_grad(set_to_none=True)
                value=loss(train,idx)
                if not torch.isfinite(value):raise RuntimeError('Nonfinite utility loss')
                value.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite utility gradient')
                optimizer.step();values.append(float(value.detach()))
            model.eval()
            with torch.no_grad():
                held=float(loss(holdout)) if holdout is not None else None
            if held is not None and held<best:
                best,best_epoch=held,epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row={'kind':kind,'scope':'inner' if holdout is not None else 'full_refit','epoch':epoch,
                 'train_loss':float(np.mean(values)), 'holdout_loss':held,'elapsed_seconds':time.time()-began}
            print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if holdout is not None and epoch-best_epoch>=12:break
        if best_state is not None:model.load_state_dict(best_state)
        return model,best_epoch

    config={'encoder_sha256':encoder_hash,'encoder_training':state['scope'],
            'encoder_training_frames':state['training_frames'],'encoder_complete_passes':state['complete_passes'],
            'counts':{k:len(v[4]) for k,v in tensors.items()},'state_dim':width,
            'scope':'Reference-pose, fixed-context TSDF gain prediction; no mesh autograd.',
            'parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
            'encoder_parameters':sum(p.numel() for p in encoder.parameters()),
            'dataset_roles':'All eligible corpus RGB trains visual encoder; only geometry-qualified gains train utility.'}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    for kind in models:
        inner,epochs=fit(kind,tensors['inner'],a.epochs,tensors['holdout'])
        save(output/(kind+'_inner.pt'),{'model':inner.state_dict(),'kind':kind,'state_dim':width,'epochs':epochs,'encoder_sha256':encoder_hash})
        full,_=fit(kind,tensors['full'],epochs)
        save(output/(kind+'_full.pt'),{'model':full.state_dict(),'kind':kind,'state_dim':width,'epochs':epochs,'encoder_sha256':encoder_hash})
    (output/'complete.json').write_text(json.dumps({'state':'complete','elapsed_seconds':time.time()-began,**config},indent=2))
    print('UTILITY TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    main()
