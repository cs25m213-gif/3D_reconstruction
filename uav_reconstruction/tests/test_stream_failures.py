import json,queue,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from adaptive_uav.runtime import StreamRunner
from adaptive_uav.runtime_metrics import Metrics
from adaptive_uav.runtime_types import FramePacket,PoseEstimate


class FailureTests(unittest.TestCase):
    def test_live_mapping_queue_replaces_stale_work_without_growth(self):
        runner=StreamRunner.__new__(StreamRunner);runner.args=SimpleNamespace(live_url='http://local/stream')
        runner.queue=queue.Queue(maxsize=1);runner.queue.put(('old',0));runner.stop=threading.Event();runner.metrics=Mock()
        runner.enqueue('new');self.assertEqual(runner.queue.qsize(),1);self.assertEqual(runner.queue.get()[0],'new')
        runner.metrics.event.assert_called_once()

    def test_offline_queue_applies_backpressure(self):
        runner=StreamRunner.__new__(StreamRunner);runner.args=SimpleNamespace(live_url=None)
        runner.queue=queue.Queue(maxsize=1);runner.queue.put(('old',0));runner.stop=threading.Event();runner.metrics=Mock()
        worker=threading.Thread(target=runner.enqueue,args=('new',));worker.start();time.sleep(.03)
        self.assertTrue(worker.is_alive());self.assertEqual(runner.queue.get()[0],'old');worker.join(timeout=1)
        self.assertFalse(worker.is_alive());self.assertEqual(runner.queue.get()[0],'new')

    def test_changed_cached_rgb_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            runner=StreamRunner.__new__(StreamRunner);runner.out=Path(d)
            frame=FramePacket('frame_000000.jpg',0,0,np.zeros((32,48,3),np.uint8),np.eye(3))
            pose=PoseEstimate(frame.frame_id,0,np.eye(4),'test')
            record=runner.record(frame,pose);Path(record['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Cached RGB differs'):runner.record(frame,pose)

    def test_invalid_resume_preserves_previous_result_and_timing(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);session=d/'session';session.mkdir();(session/'video.mp4').write_bytes(b'fixture source identity')
            out=d/'run';out.mkdir();prior={'status':'complete','result_ready':True}
            (out/'run_report.json').write_text(json.dumps(prior));(out/'timing_summary.json').write_text('original timing')
            (out/'pipeline_manifest.json').write_text('{"different":true}')
            args=SimpleNamespace(session=session,output=out,resume=True,fresh=False,mode='preview',live_url=None)
            config={'mapping_queue':1,'voxel_fraction':.02,'max_active_submaps':1,'submap_frames':4,'device':'cpu'}
            inputs={'video':{'width':640,'height':480,'duration_seconds':1},'input_hashes':{}}
            with patch('adaptive_uav.session_inputs.inspect_session',return_value=(inputs,{'focal_ratio':.75,'camera_intrinsics':None})),patch('adaptive_uav.model_versions.model_identities',return_value=({},{})):
                runner=StreamRunner(args,config)
                with self.assertRaisesRegex(ValueError,'Inputs, models or policy changed'):runner.run()
            self.assertEqual(json.loads((out/'run_report.json').read_text()),prior)
            self.assertEqual((out/'timing_summary.json').read_text(),'original timing')

    def test_failed_stage_is_present_in_event_and_timing_report(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(Metrics,'_sample',return_value=None):metrics=Metrics(d)
            with self.assertRaises(ValueError):
                with metrics.scope('failing_stage'):raise ValueError('expected failure')
            report=metrics.close()
            self.assertEqual(report['stages']['failing_stage']['count'],1)
            events=[json.loads(line) for line in (Path(d)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['error'],'expected failure')


if __name__=='__main__':unittest.main()
