"""Joint RGB-encoder/action-head optimization with incremental geometric labels.

The depth backend is versioned and fixed: changing it invalidates measured utility
labels. This updates the visual selector end to end, not a differentiable mesh or
joint depth student. All eligible auxiliary RGB is visited each corpus epoch.
"""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .large_selector_model import SelectorVisualEncoder, LargeActionSelector
from .pretrain_large_selector import Images, admitted
from .frame_degradations import random_degradation, degrade
from .selector_input_cache import GPUImages, ShardMemory


def save(path,data):
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)


@lru_cache(8)
def arrays(path):
    with np.load(path) as z:
        return {k:z[k] for k in ('rgb','t','a','x','quality','y')}


class Decisions(Dataset):
    def __init__(self, paths, memory=None):
        self.items=[];self.paths=paths;rows=[];counts={};self.memory=memory
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as z:n=len(z['y'])
            domain=meta.get('dataset','uavscenes');counts[domain]=counts.get(domain,0)+n
            rows.append((path,n,domain,float(meta.get('supervision_weight',1.))))
        total=sum(counts.values())
        for path,n,domain,trust in rows:
            weight=trust*total/(max(1,len(counts))*counts[domain])
            self.items.extend((str(path),i,weight) for i in range(n))

    def __len__(self):return len(self.items)

    def __getitem__(self,i):
        path,j,weight=self.items[i];z=self.memory.get(path) if self.memory else arrays(path);t=int(z['t'][j]);a=int(z['a'][j]);has=a>=0
        state=np.concatenate((z['x'][j],z['quality'][t],z['quality'][a] if has else np.zeros(6))).astype(np.float32)
        images=np.stack((z['rgb'][t],z['rgb'][a if has else t])).transpose(0,3,1,2).copy()
        return torch.from_numpy(images),torch.from_numpy(state),float(has),float(z['y'][j]),weight


def utility_loss(logit,y,has,weight):
    terms=F.smooth_l1_loss(logit,(20*y).clamp(-4,4),reduction='none')
    terms=terms+.2*F.binary_cross_entropy_with_logits(logit,(y>.001).float(),reduction='none')
    groups=[(terms[m]*weight[m]).sum()/weight[m].sum().clamp_min(1e-6) for m in (has<.5,has>=.5) if m.any()]
    return torch.stack(groups).mean()


def paths(base, extra):
    groups={'train':[],'holdout':[],'evaluation':[]}
    for root in (base/'utility',extra/'utility'):
        for p in sorted(root.glob('*.npz')):
            if not p.with_suffix('.json').exists():continue
            m=json.loads(p.with_suffix('.json').read_text())
            if m['split']=='train':key='holdout' if m['flight']=='AMtown03' else 'train'
            elif m['split']=='validation':key='evaluation'
            else:continue
            groups[key].append(p)
    return groups


KEYS=('rgb','t','a','x','quality','y')


class WindowMemory:
    """RAM-only utility-shard cache with background prefetch (nothing is written to local disk).

    Used with ShardWindowSampler: shards are visited in a random order, a window at a time, and
    the next window is decompressed from the mount by worker threads while the current one trains.
    """
    def __init__(self,max_gib,threads=6):
        self.limit=int(max_gib*1024**3);self.used=0;self.cache=OrderedDict();self.lock=threading.Lock()
        self.pending={};self.pool=ThreadPoolExecutor(threads);self.loads=0;self.hits=0;self.waits=0

    @staticmethod
    def _read(path):
        with np.load(path) as z:return {k:z[k] for k in KEYS}

    def _insert(self,path,row):
        size=sum(v.nbytes for v in row.values())
        with self.lock:
            self.pending.pop(path,None)
            if path in self.cache:return
            while self.cache and self.used+size>self.limit:
                _,old=self.cache.popitem(last=False);self.used-=sum(v.nbytes for v in old.values())
            self.cache[path]=row;self.used+=size

    def _fetch(self,path):
        row=self._read(path);self._insert(path,row);return row

    def prefetch(self,paths):
        with self.lock:
            for p in paths:
                if p not in self.cache and p not in self.pending:self.pending[p]=self.pool.submit(self._fetch,p)

    def get(self,path):
        with self.lock:
            if path in self.cache:self.hits+=1;self.cache.move_to_end(path);return self.cache[path]
            future=self.pending.get(path)
        if future is not None:self.waits+=1;return future.result()
        self.loads+=1;return self._fetch(path)


class ShardWindowSampler(Sampler):
    """Shuffle shard order each pass; shuffle rows within a window of shards; prefetch the next window."""
    def __init__(self,items,window,memory,seed=20261116):
        self.groups=OrderedDict()
        for i,(path,_,_) in enumerate(items):self.groups.setdefault(path,[]).append(i)
        self.window,self.memory,self.seed,self.passes,self.n=window,memory,seed,0,len(items)

    def __len__(self):return self.n

    def __iter__(self):
        rng=random.Random(self.seed+self.passes);self.passes+=1
        order=list(self.groups);rng.shuffle(order)
        for w in range(0,len(order),self.window):
            block=order[w:w+self.window]
            self.memory.prefetch(block+order[w+self.window:w+2*self.window])
            rows=[i for p in block for i in self.groups[p]];rng.shuffle(rows)
            yield from rows


def infinite(loader):
    while True:
        for b in loader:yield b


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--warmstart',type=Path)
    p.add_argument('--input-mode',choices=('auto','gpu-cache','sshfs'),default='auto')
    p.add_argument('--utility-memory-gib',type=float,default=12.)
    p.add_argument('--wait-for-pid',type=int)
    p.add_argument('--shard-window',type=int,default=48,help='>0: shard-local shuffling with RAM prefetch (gpu-cache mode)')
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(20261115);np.random.seed(20261115);random.seed(20261115);torch.set_num_threads(4);cv2.setNumThreads(1)
    encoder=SelectorVisualEncoder().cuda();head=LargeActionSelector(36).cuda()
    encoder.load_state_dict(torch.load(a.base/'visual_final/latest.pt',map_location='cpu',weights_only=False)['model'])
    # The old full head includes AMtown03; reinitialize it from the inner head to
    # preserve the held-flight early-stopping boundary for this new experiment.
    head.load_state_dict(torch.load(a.base/'selector/large_inner.pt',map_location='cpu',weights_only=False)['model'])
    teacher=torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=True).cuda().eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
    cached=a.input_mode=='gpu-cache' or (a.input_mode=='auto' and (a.output/'auxiliary_pack.json').exists())
    auxiliary=GPUImages(a.base,a.output) if cached else Images(a.base,admitted(a.base))
    auxloader=None if cached else DataLoader(auxiliary,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True)
    windowed=cached and a.shard_window>0
    memory=(WindowMemory(a.utility_memory_gib) if windowed else ShardMemory(a.utility_memory_gib)) if cached else None
    if a.wait_for_pid:
        print(json.dumps(dict(input_cache_ready=True,waiting_for_previous_pid=a.wait_for_pid)),flush=True)
        while Path(f'/proc/{a.wait_for_pid}/stat').exists():
            try:state=Path(f'/proc/{a.wait_for_pid}/stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:break
            if state=='Z':break
            time.sleep(1)
        if a.warmstart and not a.warmstart.exists():raise RuntimeError('Handoff checkpoint is missing; refusing to reset trained weights')
    best=float('inf');began=time.time();step=0;patience=0;signature=None;start=1
    latest=a.output/'latest.pt'
    resume=a.warmstart if a.warmstart else latest
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        encoder.load_state_dict(state['encoder']);head.load_state_dict(state['head'])
        if 'optimizer' in state:optimizer.load_state_dict(state['optimizer'])
        else:print('Resume checkpoint contains model weights only; AdamW state is initialized afresh.',flush=True)
        # Intermediate checkpoints restart their incomplete corpus epoch, so a
        # loader optimization cannot silently skip images from that pass.
        finished=state.get('epoch_complete',not a.warmstart)
        start=state['epoch']+int(finished);step=state['step'];best=state.get('best_holdout',float('inf'));signature=state.get('label_signature');patience=state.get('patience',0)
        print(json.dumps(dict(resume=str(resume),step=step,optimizer_restored='optimizer' in state,corpus_epoch_restarted=not finished)),flush=True)
    print(json.dumps({'state':'training','auxiliary_frames':len(auxiliary),'trainable_encoder':True,'trainable_head':True,'depth_backend_fixed':True,'input_mode':'gpu-cache' if cached else 'sshfs'}),flush=True)
    log=(a.output/'steps.jsonl').open('a')
    final=False
    for epoch in range(start,a.epochs+1):
        groups=paths(a.base,a.output)
        if memory and not windowed:
            cache_start=time.time()
            for i,path in enumerate(groups['train']+groups['holdout']):
                memory.get(str(path))
                if i%100==0:print(json.dumps(dict(utility_cache_shards=i+1,bytes=memory.used,seconds=time.time()-cache_start)),flush=True)
        train=Decisions(groups['train'],memory);held=Decisions(groups['holdout'],memory)
        if not len(train) or not len(held):raise RuntimeError('Empty training or held-flight data')
        sig=hashlib.sha256('\n'.join(map(str,groups['train'])).encode()).hexdigest()
        if sig!=signature:signature=sig;patience=0;best=float('inf')
        source_counts={}
        for f in groups['train']:
            meta=json.loads(f.with_suffix('.json').read_text());d=meta.get('dataset','uavscenes');source_counts[d]=source_counts.get(d,0)+meta['rows']
        ledger={'epoch':epoch,'sources':source_counts,'training_rows':len(train),'held_rows':len(held),'training_files':list(map(str,groups['train'])),'evaluation_excluded':list(map(str,groups['evaluation']))}
        (a.output/f'epoch_{epoch:03d}_manifest.json').write_text(json.dumps(ledger,indent=2))
        if windowed:loader=DataLoader(train,batch_size=a.batch_size,sampler=ShardWindowSampler(train.items,a.shard_window,memory),num_workers=0,pin_memory=True)
        else:loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0 if cached else a.workers,pin_memory=True)
        hloader=DataLoader(held,batch_size=a.batch_size,num_workers=0 if cached else 2,pin_memory=True)
        aux_batches=(len(auxiliary)+a.batch_size-1)//a.batch_size if cached else len(auxloader)
        batches=max(aux_batches,len(loader));uit=infinite(loader)
        def auxiliary_batches():
            while True:
                if cached:yield from auxiliary.batches(a.batch_size)
                else:
                    for pixels,_ in auxloader:yield pixels.cuda(non_blocking=True).float()/255
        ait=auxiliary_batches()
        encoder.train();head.train();total=0.;interval_wait=0.;interval_step=0.;interval_count=0
        for iteration in range(batches):
            tick=time.perf_counter()
            images,x,has,y,weight=next(uit);clean=next(ait)
            interval_wait+=time.perf_counter()-tick
            images=images.cuda(non_blocking=True).float()/255;x=x.cuda();has=has.cuda().float();y=y.cuda().float();weight=weight.cuda().float()
            kinds,strength=random_degradation(len(clean),.5,clean.device);damaged=degrade(clean,kinds,strength)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                embedding,_=encoder(images.flatten(0,1));candidate,occupant=embedding.reshape(len(images),2,-1).unbind(1)
                pred=head(candidate,occupant,has,x);utility=utility_loss(pred.float(),y,has,weight)
                aux,corruption=encoder(damaged)
                with torch.no_grad():target=F.normalize(teacher((clean-encoder.mean)/encoder.std).float(),dim=-1)
                semantic=(1-(aux.float()*target).sum(-1)).mean();fault=F.cross_entropy(corruption.float(),kinds+1)
                loss=utility+.05*semantic+.01*fault
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();en=torch.nn.utils.clip_grad_norm_(encoder.parameters(),2.);hn=torch.nn.utils.clip_grad_norm_(head.parameters(),2.)
            if not torch.isfinite(en+hn) or min(float(en),float(hn))<=0:raise RuntimeError('Both selector encoder and head need finite nonzero gradients')
            optimizer.step();step+=1;total+=float(loss.detach())
            interval_step+=time.perf_counter()-tick;interval_count+=1
            if step%25==0 or step==1:
                row=dict(epoch=epoch,step=step,iteration=iteration+1,epoch_batches=batches,loss=float(loss.detach()),utility=float(utility.detach()),semantic=float(semantic.detach()),encoder_grad=float(en),head_grad=float(hn),source_rows=source_counts,elapsed_seconds=time.time()-began,seconds_per_batch=interval_step/interval_count,data_wait_seconds=interval_wait/interval_count,input_mode='gpu-cache' if cached else 'sshfs',utility_cache_hits=memory.hits if memory else None,utility_shard_loads=memory.loads if memory else None)
                text=json.dumps(row);print(text,flush=True);log.write(text+'\n');log.flush()
                interval_wait=interval_step=0.;interval_count=0
            if step%500==0:
                save(a.output/'in_progress.pt',dict(encoder=encoder.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,epoch_complete=False,iteration=iteration+1,step=step,best_holdout=best,patience=patience,label_signature=signature,scope='incomplete corpus epoch; resume replays this pass'))
        encoder.eval();head.eval();values=[]
        with torch.no_grad():
            for images,x,has,y,weight in hloader:
                images=images.cuda().float()/255;x=x.cuda();has=has.cuda().float();y=y.cuda().float();weight=weight.cuda().float()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    emb,_=encoder(images.flatten(0,1));c,o=emb.reshape(len(images),2,-1).unbind(1);pred=head(c,o,has,x)
                values.append((float(utility_loss(pred.float(),y,has,weight)),len(y)))
        validation=sum(v*n for v,n in values)/sum(n for _,n in values)
        improved=validation<best;best=min(best,validation);patience=0 if improved else patience+1
        state=dict(encoder=encoder.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,epoch_complete=True,step=step,best_holdout=best,patience=patience,label_signature=signature,source_rows=source_counts,auxiliary_frames=len(auxiliary),scope='RGB selector encoder and action head trained jointly; fixed versioned depth; reference-pose proxy gains')
        save(latest,state)
        if improved:save(a.output/'best_holdout.pt',state)
        print(json.dumps(dict(epoch_complete=epoch,mean_training_loss=total/batches,held_loss=validation,source_rows=source_counts)),flush=True)
        status=a.output/'utility_status.json';label_status=json.loads(status.read_text()) if status.exists() else {}
        if label_status.get('state')=='failed':raise RuntimeError('Geometry/label preparation failed: inspect remote statuses; do not declare complete')
        final=label_status.get('state')=='complete' and set(groups['train'])==set(paths(a.base,a.output)['train'])
        if epoch>=5 and patience>=5 and final:break
    result=dict(state='complete' if final else 'waiting_for_remaining_geometry_labels',epoch=epoch,step=step,best_holdout=best,source_rows=source_counts,elapsed_seconds=time.time()-began)
    (a.output/'training_status.json').write_text(json.dumps(result,indent=2))
    if final:(a.output/'complete.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
