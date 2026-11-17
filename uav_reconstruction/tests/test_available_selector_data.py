import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

from uav_reconstruction.adaptive_uav.prepare_available_selector_data import (
    admit, role_for_flight, seal,
)
from uav_reconstruction.adaptive_uav.prepare_available_selector_geometry import geometry
from uav_reconstruction.adaptive_uav.lidar_depth_supervision import extrinsics_for_scene
from uav_reconstruction.adaptive_uav.supervise_available_selector_data import link_original_utilities


class AdmissionTests(unittest.TestCase):
    def test_flight_boundaries(self):
        cases = {'HKairport_GNSS01':'train', 'HKairport_GNSS02':'train',
                 'HKairport_GNSS03':'holdout', 'HKairport03':'holdout',
                 'AMtown03':'holdout', 'AMvalley02':'validation',
                 'HKisland_GNSS_Evening':'test', 'HKairport_GNSS_Evening':'train',
                 'HKairport_GNSS04':'unknown', 'AMtown01_extra':'unknown'}
        for flight, expected in cases.items():
            self.assertEqual(role_for_flight(flight), expected, flight)

    def test_protected_frames_indexed_without_decode_and_masks_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);base=root/'base';out=base/'expansion'
            base.mkdir();(base/'visual_train.jsonl').write_text('')
            archive=root/'flight.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('interval1_AMtown03/interval1_CAM/1.0.jpg', b'invalid JPEG')
                z.writestr('interval1_AMtown03/labels/1.0.png', b'not RGB')
            task=dict(id='protected',path=str(archive),kind='uav_zip',dataset='uavscenes',
                      flight='AMtown03',image_directory='interval1_CAM',role='holdout')
            result=admit((task,str(out),str(base)))
            self.assertEqual(result['counts'],{'holdout':1})
            self.assertFalse(list(out.rglob('*.jpg')))

    def test_dedup_union_paths_and_original_manifest_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';out=base/'expansion';base.mkdir()
            image=np.full((224,224,3),80,np.uint8)
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            (base/'old.jpg').write_bytes(b'placeholder')
            original=json.dumps(dict(path='old.jpg',sha256_rgb224=digest,role='train',dataset='old'))+'\n'
            (base/'visual_train.jsonl').write_text(original)
            original_hash=hashlib.sha256(original.encode()).hexdigest()
            archive=root/'flight.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for index,value in enumerate((80,120)):
                    ok,jpg=cv2.imencode('.jpg',np.full_like(image,value));self.assertTrue(ok)
                    z.writestr(f'interval1_AMtown01/interval1_CAM/{index}.0.jpg',jpg.tobytes())
            task=dict(id='new',path=str(archive),kind='uav_zip',dataset='uavscenes',
                      flight='AMtown01',image_directory='interval1_CAM',role='train')
            result=admit((task,str(out),str(base)))
            self.assertEqual((result['admitted'],result['duplicates']),(1,1))
            final=seal(out,base,[task],original_hash)
            self.assertEqual(final['unique_training_frames'],2)
            self.assertEqual((base/'visual_train.jsonl').read_text(),original)
            for line in (out/'corpus/visual_train.jsonl').read_text().splitlines():
                row=json.loads(line);self.assertTrue((out/'corpus'/row['path']).is_file())
            (base/'visual_train.jsonl').write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'changed'):
                seal(out,base,[task],original_hash)

    def test_native_lidar_transform_and_reference_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'output';(output/'geometry').mkdir(parents=True)
            archive=root/'native.zip';K=np.array([[150.,0,128],[0,150.,96],[0,0,1]])
            yy,xx=np.mgrid[30:160:6,30:230:6]
            world=np.column_stack(((xx.ravel()-128)/150*20,(yy.ravel()-96)/150*20,np.full(xx.size,20.)))
            r,t=extrinsics_for_scene('AMtown01');infos=[]
            with zipfile.ZipFile(archive,'w') as z:
                for index,stamp in enumerate(('0.0','0.1','0.2')):
                    pose=np.eye(4);pose[0,3]=index*.1
                    camera=world-pose[:3,3]
                    lidar=np.linalg.solve(r,(camera-t).T).T
                    lines='\n'.join(' '.join(map(str,p)) for p in lidar)
                    z.writestr(f'interval1_AMtown01/interval1_LIDAR/image{stamp}_lidar{stamp}.txt',lines)
                    ok,jpg=cv2.imencode('.jpg',np.full((192,256,3),100,np.uint8));self.assertTrue(ok)
                    z.writestr(f'interval1_AMtown01/interval1_CAM/{stamp}.jpg',jpg.tobytes())
                    infos.append(dict(OriginalImageName=stamp+'.jpg',T4x4=pose.tolist(),
                                      P3x3=K.tolist(),Width=256,Height=192,
                                      K1=0,K2=0,K3=0,P1=0,P2=0))
                z.writestr('interval1_AMtown01/sampleinfos_interpolated.json',json.dumps(infos))
            result=geometry((dict(path=str(archive),flight='AMtown01',role='train'),str(output)))
            self.assertEqual(result['episodes'],1)
            with np.load(next((output/'geometry').glob('*.npz'))) as data:
                self.assertEqual(len(data['rgb']),3)
                np.testing.assert_allclose(data['reference_points'][:,2],20,atol=1e-5)
                np.testing.assert_allclose(data['P'][2,0,3],-.2)
                np.testing.assert_allclose(data['K'][0],K)

    def test_union_utility_links_keep_both_original_namespaces(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);corpus=base/'expansion/corpus';corpus.mkdir(parents=True)
            for folder in (base/'utility',base/'complete_20261115/utility'):
                folder.mkdir(parents=True);(folder/'same.npz').write_bytes(b'original')
                (folder/'same.json').write_text('{"split":"train"}')
            self.assertEqual(link_original_utilities(base,corpus),2)
            self.assertEqual(link_original_utilities(base,corpus),2)
            self.assertEqual(len(list((corpus/'utility').glob('*.npz'))),2)
            for path in (corpus/'utility').iterdir():
                self.assertTrue(path.is_symlink());self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
