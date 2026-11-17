import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_uav.robust_bundle import project, refine, load_observations
from adaptive_uav.quality_audit import audit, quality_gate


class RobustBundleTests(unittest.TestCase):
    def test_radial_projection_and_cheirality(self):
        uv,depth=project(np.zeros((1,6)),np.array([[1.,2.,10.]]),
                         np.array([0]),np.array([0]),np.array([100.,50.,40.,.1]))
        np.testing.assert_allclose(uv,[[60.05,60.1]])
        np.testing.assert_allclose(depth,[10.])

    def test_synthetic_bundle_reduces_error_and_preserves_consistent_tracks(self):
        rng=np.random.default_rng(12)
        points=rng.uniform([-1,-1,5],[1,1,8],(100,3))
        cameras=np.zeros((5,6)); cameras[:,3]=-np.linspace(0,2,5)
        initial_points=points+rng.normal(0,.015,points.shape)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source'; source.mkdir()
            (source/'cameras.txt').write_text('1 SIMPLE_RADIAL 1280 720 800 640 360 0\n')
            images=[]
            for i,cp in enumerate(cameras):
                uv,_=project(cameras,points,np.full(100,i),np.arange(100),[800,640,360,0])
                images.append(f'{i+1} 1 0 0 0 {cp[3]} 0 0 1 frame_{i}.jpg')
                images.append(' '.join(str(v) for j,p in enumerate(uv) for v in (*p,j+1)))
            (source/'images.txt').write_text('\n'.join(images)+'\n')
            tracks=[v for i in range(5) for v in (i+1,0)]
            lines=[]
            for j,p in enumerate(initial_points):
                tracks=[v for i in range(5) for v in (i+1,j)]
                lines.append(' '.join(map(str,[j+1,*p,100,150,200,1,*tracks])))
            (source/'points3D.txt').write_text('\n'.join(lines)+'\n')
            output=Path(directory)/'result'
            result=refine(source,output,max_points=60,max_nfev=50,refine_intrinsics=False)
            # Small synthetic fixtures intentionally cannot meet the production
            # 150-landmark minimum, but the optimization itself must converge.
            self.assertTrue(result['solver_success'])
            self.assertLess(result['final_robust_cost'],result['initial_robust_cost']*.1)
            images,points=load_observations(output)
            self.assertEqual(len(images),5)
            self.assertEqual(len(points),100)
            self.assertEqual(result['optimized_points'],60)
            self.assertEqual(result['retriangulated_or_revalidated_points'],40)
            lookup={p['id']:p for p in images}
            for p in points:
                for view,key in p['track']:
                    self.assertEqual(int(lookup[int(view)]['observations'][key,2]),p['id'])
            self.assertTrue(quality_gate(audit(output),{})[0])
            with self.assertRaises(ValueError):
                refine(source,source)


if __name__=='__main__':
    unittest.main()
