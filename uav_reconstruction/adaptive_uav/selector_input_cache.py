"""Sequential remote RGB packing and bounded, memory-only training caches."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch


def read_rgb(path):
    bgr=cv2.imread(str(path))
    if bgr is None or bgr.shape!=(224,224,3):raise ValueError(f'Invalid corpus RGB: {path}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def pack(base, output):
    manifest=base/'visual_train.jsonl';digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    if any(r['role']!='train' for r in rows):raise ValueError('Held-out image in training pack')
    marker=output/'auxiliary_pack.json';target=output/'auxiliary_rgb.npy'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous['manifest_sha256']!=digest:raise ValueError('Pack manifest differs')
        if target.stat().st_size!=previous['file_bytes']:raise ValueError('Truncated existing pack')
        return previous
    partial=output/'auxiliary_rgb.partial.npy'
    bank=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=(len(rows),224,224,3))
    began=time.time();pixel_hash=hashlib.sha256()
    with ThreadPoolExecutor(8) as pool:
        for start in range(0,len(rows),512):
            images=list(pool.map(read_rgb,[base/r['path'] for r in rows[start:start+512]]))
            block=np.stack(images);bank[start:start+len(block)]=block;pixel_hash.update(block.tobytes())
            if start%8192==0:
                status=dict(state='packing',frames=start+len(block),total=len(rows),elapsed_seconds=time.time()-began)
                (output/'pack_status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    bank.flush();del bank;partial.replace(target)
    result=dict(state='complete',frames=len(rows),shape=[len(rows),224,224,3],dtype='uint8',
        manifest_sha256=digest,pixel_sha256=pixel_hash.hexdigest(),file_bytes=target.stat().st_size,
        elapsed_seconds=time.time()-began,storage='remote server; local training uses volatile memory only')
    temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(marker)
    (output/'pack_status.json').write_text(json.dumps(result,indent=2));return result


class GPUImages:
    def __init__(self, base, output, device='cuda'):
        meta=json.loads((output/'auxiliary_pack.json').read_text())
        digest=hashlib.sha256((base/'visual_train.jsonl').read_bytes()).hexdigest()
        if meta['manifest_sha256']!=digest:raise ValueError('Auxiliary pack identity mismatch')
        target=output/'auxiliary_rgb.npy'
        if target.stat().st_size!=meta['file_bytes']:raise ValueError('Truncated auxiliary pack')
        shape=tuple(meta['shape']);needed=int(np.prod(shape))
        if torch.device(device).type=='cuda':
            free,_=torch.cuda.mem_get_info(device)
            if free-needed<6*1024**3:raise RuntimeError('Not enough GPU memory for corpus plus 6 GiB training reserve')
        self.rgb=torch.empty(shape,dtype=torch.uint8,device=device)
        # Large sequential reads avoid per-image SSHFS round trips. No local
        # disk file or mmap page-fault stream is used.
        began=time.time();digest_pixels=hashlib.sha256()
        with target.open('rb',buffering=0) as stream:
            version=np.lib.format.read_magic(stream)
            header=np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream)
            if header!=(shape,False,np.dtype('uint8')):raise ValueError('Unexpected pack header')
            for start in range(0,shape[0],512):
                n=min(512,shape[0]-start);size=n*224*224*3
                buffer=bytearray(size);view=memoryview(buffer);offset=0
                while offset<size:
                    count=stream.readinto(view[offset:])
                    if not count:raise EOFError('Truncated RGB pack')
                    offset+=count
                digest_pixels.update(buffer)
                block=np.frombuffer(buffer,np.uint8).reshape(n,224,224,3)
                self.rgb[start:start+n].copy_(torch.from_numpy(block))
        if digest_pixels.hexdigest()!=meta['pixel_sha256']:raise ValueError('Packed RGB checksum mismatch')
        print(json.dumps(dict(auxiliary_GPU_cache_frames=shape[0],bytes=needed,load_seconds=time.time()-began)),flush=True)

    def __len__(self):return len(self.rgb)

    def batches(self,batch_size):
        order=torch.randperm(len(self),device=self.rgb.device)
        for ids in order.split(batch_size):yield self.rgb[ids].permute(0,3,1,2).float()/255


class ShardMemory:
    def __init__(self,max_gib=6):
        self.limit=int(max_gib*1024**3);self.used=0;self.cache=OrderedDict();self.loads=0;self.hits=0

    def get(self,path):
        if path in self.cache:
            self.hits+=1;self.cache.move_to_end(path);return self.cache[path]
        with np.load(path) as z:row={k:z[k] for k in ('rgb','t','a','x','quality','y')}
        size=sum(v.nbytes for v in row.values());self.loads+=1
        while self.cache and self.used+size>self.limit:
            _,old=self.cache.popitem(last=False);self.used-=sum(v.nbytes for v in old.values())
        if size<=self.limit:self.cache[path]=row;self.used+=size
        return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(pack(a.base,a.output)),flush=True)


if __name__=='__main__':main()
