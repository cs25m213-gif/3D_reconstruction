"""Explicit exclusion of sensor-calibration clips from task training/validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from adaptive_uav.student_dataset import discover


class CohortTests(unittest.TestCase):
    def test_configured_calibration_exclusion_preserves_default_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'cache').mkdir()
            scales={}
            for name,scene in [('AMtown01_0','AMtown01'),('AMtown01_1','AMtown01'),
                               ('AMvalley01_0','AMvalley01'),('AMvalley01_1','AMvalley01')]:
                p=root/'cache'/name;p.mkdir();(p/'selected_infos.json').write_text('[]')
                scales[name]={'usable':True,'scene':scene,'scale_m_per_visual_unit':1.}
            (root/'scales.json').write_text(json.dumps(scales))
            (root/'gate.json').write_text(json.dumps({'flights':{'AMtown01':{'status':'QUALIFIED'},'AMvalley01':{'status':'QUALIFIED'}}}))
            contract={'data':{'cache_root':'cache','scale_calibration':'scales.json','sidecars':'targets'},
                      'geometry_gate':'gate.json','geometry':{'calibration_batch_suffix':'_0'},
                      'splits':{'train':['AMtown'],'validation':['AMvalley']}}
            with mock.patch('adaptive_uav.student_dataset.index_clip',side_effect=lambda x:x):
                old=discover(root,contract)
                self.assertEqual([len(old[r]) for r in ['train','validation']],[2,2])
                contract['student']={'exclude_calibration_clips':True}
                new=discover(root,contract)
            self.assertEqual([x.name for x in new['train']],['AMtown01_1'])
            self.assertEqual([x.name for x in new['validation']],['AMvalley01_1'])


if __name__=='__main__':unittest.main()
