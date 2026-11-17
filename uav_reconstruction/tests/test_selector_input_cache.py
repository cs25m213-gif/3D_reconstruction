"""Check loader optimization preserves pixels, labels and split boundaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from uav_reconstruction.adaptive_uav.selector_input_cache import pack,GPUImages,ShardMemory,read_rgb
from uav_reconstruction.adaptive_uav.train_complete_selector import Decisions


class SelectorInputCacheTests(unittest.TestCase):
    def test_packed_pixels_equal_original_JPEG_decoder_and_cover_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();rows=[]
            rng=np.random.default_rng(42)
            for i in range(5):
                cv2.imwrite(str(root/f'{i}.jpg'),rng.integers(0,256,(224,224,3),dtype=np.uint8))
                rows.append(dict(path=f'{i}.jpg',role='train'))
            manifest=root/'visual_train.jsonl';manifest.write_text('\n'.join(map(json.dumps,rows)))
            pack(root,out);cache=GPUImages(root,out,'cpu')
            for i in range(5):np.testing.assert_array_equal(cache.rgb[i].numpy(),read_rgb(root/f'{i}.jpg'))
            self.assertEqual(sum(len(b) for b in cache.batches(2)),5)
            manifest.write_text(manifest.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):GPUImages(root,out,'cpu')

    def test_pack_refuses_protected_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'visual_train.jsonl').write_text(json.dumps(dict(path='missing.jpg',role='test')))
            with self.assertRaisesRegex(ValueError,'Held-out'):pack(root,root)

    def test_resident_utility_matches_original_samples_and_respects_memory_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'data.npz'
            rng=np.random.default_rng(11)
            np.savez_compressed(p,rgb=rng.integers(0,256,(3,224,224,3),dtype=np.uint8),t=[0,1,2],a=[-1,0,1],
                x=rng.normal(size=(3,24)).astype(np.float32),quality=rng.normal(size=(3,6)).astype(np.float32),y=np.array([.1,-.2,.3],np.float32))
            p.with_suffix('.json').write_text(json.dumps(dict(dataset='test',rows=3)))
            memory=ShardMemory(.001);old=Decisions([p]);new=Decisions([p],memory)
            for i in range(3):
                for a,b in zip(old[i],new[i]):
                    if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
                    else:self.assertEqual(a,b)
            self.assertEqual(memory.loads,1);self.assertGreater(memory.hits,0);self.assertLessEqual(memory.used,memory.limit)


if __name__=='__main__':unittest.main()
