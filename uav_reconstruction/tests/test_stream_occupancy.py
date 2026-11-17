import tempfile,unittest
from pathlib import Path
import numpy as np
from adaptive_uav.occupancy import OccupancyMap,traverse
from adaptive_uav.sensor_coordinates import gps_to_enu


class OccupancyTests(unittest.TestCase):
    def test_rays_distinguish_observed_free_occupied_unknown(self):
        m=OccupancyMap(1.)
        try:
            m.integrate([.1,.1,.1],[3.1,.1,.1],'ray')
            self.assertEqual(m.state((1,0,0)),'free');self.assertEqual(m.state((3,0,0)),'occupied')
            self.assertEqual(m.state((4,0,0)),'unknown')
        finally:m.close()

    def test_invalid_and_dynamic_depth_does_not_carve_space(self):
        m=OccupancyMap(1.)
        try:
            self.assertFalse(m.integrate([0,0,0],[0,0,10],'x',valid=False))
            self.assertFalse(m.integrate([0,0,0],[0,0,10],'x',excluded=True))
            self.assertEqual(len(m.cells),0)
        finally:m.close()

    def test_repetition_and_reload_preserve_evidence_identity(self):
        with tempfile.TemporaryDirectory() as d:
            m=OccupancyMap(1.,max_cells=4);m.integrate([.1,0,0],[3.1,0,0],'a')
            original=dict(m.cells);m.integrate([.1,0,0],[3.1,0,0],'a');self.assertEqual(dict(m.cells),original)
            path=Path(d)/'map.json';m.save(path);n=OccupancyMap.load(path)
            try:self.assertFalse(n.integrate([.1,0,0],[3.1,0,0],'a'));self.assertEqual(dict(n.cells),original)
            finally:m.close();n.close()

    def test_negative_coordinates_corner_ties_and_budget(self):
        self.assertEqual(list(traverse([-.1,-.1,-.1],[1.1,1.1,1.1],1.)),[(-1,-1,-1),(0,0,0),(1,1,1)])
        m=OccupancyMap(1.,max_cells=2)
        try:m.integrate([0,0,0],[5,0,0],'a');self.assertLessEqual(len(m.cells),2)
        finally:m.close()

    def test_gps_local_origin_and_axes(self):
        origin=[0,0,0];values=gps_to_enu([[0,0,0],[0,.00001,0],[.00001,0,0],[0,0,1]],origin)
        np.testing.assert_allclose(values[0],0,atol=1e-8)
        self.assertGreater(values[1,0],1.);self.assertGreater(values[2,1],1.);self.assertAlmostEqual(values[3,2],1.)


if __name__=='__main__':unittest.main()
