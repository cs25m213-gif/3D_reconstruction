import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from adaptive_uav.open3d_cpu import cpu_bindings
from adaptive_uav.surface_audit import audit_surface


class SurfaceAuditTests(unittest.TestCase):
    def test_cpu_ray_coverage_detects_missing_surfaces(self):
        try:
            o3d=cpu_bindings()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'dense').mkdir()
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector(np.array([[-10,-10,5],[10,-10,5],
                                                             [10,10,5],[-10,10,5]],dtype=float))
            mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
            o3d.io.write_triangle_mesh(str(root/'dense'/'scene_mesh.ply'),mesh)
            data={'cameras':[{'id':1,'model':'SIMPLE_RADIAL','width':1280,'height':720,
                              'parameters':[800,640,360,0]}],
                  'poses':[{'image':'frame.jpg','camera_id':1,'quaternion_wxyz':[1,0,0,0],
                            'translation_world_to_camera':[0,0,0]}]}
            path=root/'camera_poses.json'; path.write_text(json.dumps(data))
            report=audit_surface(root,threads=1)
            self.assertTrue(report['coverage_gate_passed'])
            self.assertEqual(report['sampled_views'][0]['ray_hit_fraction'],1.)
            self.assertFalse(report['geometric_accuracy_validated'])
            data['poses'][0]['translation_world_to_camera']=[-20,0,0]
            path.write_text(json.dumps(data))
            report=audit_surface(root,threads=1)
            self.assertFalse(report['coverage_gate_passed'])
            self.assertEqual(report['sampled_views'][0]['ray_hit_fraction'],0.)


if __name__=='__main__':
    unittest.main()
