"""Frozen DINOv2 ViT-S/14 descriptors, computed once per cached image.

Selection labels, training and runtime must all use the same descriptor, so the name travels
with the checkpoint. thumbnail384 exists only for CPU smoke tests and is never called DINO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .memory_graph import thumbnail_descriptor
from .utility_contracts import load_contract, write_json, write_status

DINO, THUMBNAIL = 'dinov2_vits14', 'thumbnail384'


class Descriptors:
    """Lookup with an explicit fallback rule: a missing DINO entry is an error, not a thumbnail."""

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder) if folder else None
        self.name = DINO if self.folder else THUMBNAIL
        self.clips: dict = {}

    def get(self, clip: str, index: int, rgb: np.ndarray) -> np.ndarray:
        if self.folder is None:
            return thumbnail_descriptor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if clip not in self.clips:
            with np.load(self.folder / f'{clip}.npz') as data:
                self.clips[clip] = dict(zip(data['frames'].tolist(), data['features']))
        return self.clips[clip][index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contract = load_contract(args.config)
    root = args.root.resolve()
    batches = sorted(p for p in (root / contract['data']['cache_root']).iterdir() if (p / 'selected_infos.json').exists())
    if args.dry_run:
        print(json.dumps({'clips': len(batches), 'device': args.device}))
        return
    import torch
    from .graph_selector import frozen_encoder, image_feature
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    encoder = frozen_encoder(device)
    args.output.mkdir(parents=True, exist_ok=True)
    started, total = time.monotonic(), 0
    for batch in batches:
        target = args.output / f'{batch.name}.npz'
        if target.exists():
            continue
        frames, features = [], []
        for file in sorted((batch / 'reconstruction/multiview/depth_cache').glob('frame_*.jpg.npz')):
            with np.load(file) as cached:
                rgb = cached['rgb']
            frames.append(int(file.name.split('_')[1].split('.')[0]))
            features.append(image_feature(encoder, rgb, device))
        temporary = target.with_name(target.stem + '.partial.npz')
        np.savez_compressed(temporary, frames=np.asarray(frames), features=np.float32(features))
        temporary.replace(target)
        total += len(frames)
    write_json(args.output / 'manifest.json', {'descriptor': DINO, 'dimensions': 384, 'images': total, 'device': str(device),
                                               'seconds': time.monotonic() - started})
    write_status(args.output, 'embeddings', 'complete', 'PASS', 'stage3_memory')
    print(json.dumps({'images': total}))


if __name__ == '__main__':
    main()
