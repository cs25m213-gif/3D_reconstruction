"""Analytic calibrated cameras/landmarks; no estimated real-world ground truth."""
import tempfile,unittest
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np,pycolmap
from adaptive_uav.colmap_workspace import copy_tracked_workspace,align_sparse_depth
from adaptive_uav.runtime_types import DepthFrame


class TrackedWorkspaceTests(unittest.TestCase):
    def fixture(self,root):
        sparse=root/'source/sparse';images=root/'source/images';sparse.mkdir(parents=True);images.mkdir()
        k=np.array([[60.,0,32],[0,60.,24],[0,0,1.]])
        pixels=np.array([(x,y) for x in range(16,49,4) for y in range(12,37,4)],float)
        xyz=np.column_stack(((pixels[:,0]-32)/60*2,(pixels[:,1]-24)/60*2,np.full(len(pixels),2.)))
        (sparse/'cameras.txt').write_text('1 PINHOLE 64 48 60 60 32 24\n')
        lines=[]
        for ident,center in enumerate([0.,.2,-.2],1):
            name=f'frame_{ident:06d}.jpg';cv2.imwrite(str(images/name),np.full((48,64,3),130,np.uint8))
            uv=pixels.copy();uv[:,0]-=center*30
            observations=' '.join(f'{u} {v} {point+1}' for point,(u,v) in enumerate(uv))
            lines.append(f'{ident} 1 0 0 0 {-center} 0 0 1 {name}\n{observations}\n')
        (sparse/'images.txt').write_text(''.join(lines))
        (sparse/'points3D.txt').write_text(''.join(f'{i+1} {x} {y} {z} 100 120 140 0 1 {i} 2 {i} 3 {i}\n' for i,(x,y,z) in enumerate(xyz)))
        return pycolmap.Reconstruction(sparse),images,k

    def test_original_triangulated_tracks_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,images,k=self.fixture(root)
            copy_tracked_workspace(root/'dense',model,images)
            copied=pycolmap.Reconstruction(root/'dense/sparse')
            self.assertEqual(copied.num_points3D(),63);self.assertEqual(copied.num_reg_images(),3)
            self.assertEqual(copied.points3D[1].track.length(),3)

    def test_sparse_alignment_keeps_dense_predictions_labelled_learned(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,images,k=self.fixture(root)
            f=DepthFrame('frame_000001.jpg',0,np.full((48,64),6.,np.float32),np.full((48,64,3),130,np.uint8),k,np.eye(4),'test')
            aligned=align_sparse_depth(f,root/'source',model)
            np.testing.assert_allclose(aligned.depth,2.)
            self.assertTrue((aligned.evidence==3).all())
            self.assertEqual(aligned.validation['anchor_source'],'triangulated_sfm_tracks')
            self.assertLess(aligned.validation['median'],1e-5)

    def test_excluded_features_do_not_establish_depth_scale(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,images,k=self.fixture(root)
            f=DepthFrame('frame_000001.jpg',0,np.full((48,64),6.,np.float32),np.full((48,64,3),130,np.uint8),k,np.eye(4),'test',excluded=np.ones((48,64),bool))
            aligned=align_sparse_depth(f,root/'source',model)
            self.assertFalse(aligned.validation['sparse_anchor_check']['scale_fit_valid'])
            self.assertIsNone(aligned.validation['median'])


if __name__=='__main__':unittest.main()
