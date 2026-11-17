"""Small CPU correctness tests; no datasets, downloads, GPU or training required."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import shutil
import subprocess
from unittest import mock

import numpy as np
import torch

from adaptive_uav import sequential_replay as seq
from adaptive_uav.counterfactual_replay import View
from adaptive_uav.embedding_cache import Descriptors
from adaptive_uav.student_inference import anchor_scale, predict
from adaptive_uav.build_utility_dataset import build
from adaptive_uav.evaluate_vnext import promotion
from adaptive_uav.promotion_gate import allowed
from adaptive_uav.reconstruction_quality import DEFAULT_WEIGHTS
from adaptive_uav.stage_cache import matches


def frame(i):
    rgb=np.random.default_rng(i).integers(0,255,(16,16,3),dtype=np.uint8)
    pose=np.eye(4);pose[0,3]=float(i)
    k=np.array([[20.,0,8],[0,20.,8],[0,0,1]])
    depth=np.full((16,16),9999.,np.float32)  # Deliberately unusable teacher scale.
    return View(f'c/{i}',i,float(i),rgb,depth,k,pose,pose,k,.5,rgb,depth,k,
                aligned_pose=pose,anchors_m=np.full((16,16),10.,np.float32))


class Provider:
    def __init__(self):self.calls=[]
    def __call__(self,v,p,history):
        ids=tuple(x.frame_id for x,_ in history[-2:])
        self.calls.append((v.frame_id,ids))
        return seq.replace(v,depth_m=np.full((16,16),10.,np.float32),
                           fusion_depth=np.full((16,16),10.,np.float32),source_ids=ids),.001,False


def found(v,history,minimum):
    return {'ok':True,'pose':v.feature_pose.copy(),'matches':1,'inliers':40}


class ReplayTests(unittest.TestCase):
    def initial(self,calls=4):
        p=Provider();prefix=[]
        for v in [frame(0),frame(1)]:
            dense,_,_=p(v,v.feature_pose,prefix);prefix.append((dense,v.feature_pose))
        return seq.Branch(prefix,dict(seconds=20.,depth_calls=calls,matches=64),Descriptors(),'c'),p

    def test_forks_have_private_map_budget_hidden_and_arrays(self):
        start,_=self.initial();start.hidden=np.ones(3)
        with mock.patch.object(seq.MapState,'commit',side_effect=AssertionError('fork must not rematch')):
            a,b=start.fork(),start.fork()
        a.budget.charge(depth_calls=1);a.hidden[0]=99
        a.history[0][0].depth_m[0,0]=99
        a.memory.coverage.counts.clear()
        self.assertEqual(b.budget.spent['depth_calls'],0)
        self.assertEqual(b.hidden[0],1)
        self.assertEqual(b.history[0][0].depth_m[0,0],10)
        self.assertTrue(b.memory.coverage.counts)

    def test_three_actions_share_tracking_but_not_optional_depth(self):
        start,p=self.initial();v=frame(2);probe=found(v,start.history,30)
        branches={a:start.fork() for a in seq.ACTIONS}
        with mock.patch.object(seq,'integration_seconds',return_value=0.):
            for action,b in branches.items():b.act(v,action,probe,.001,p)
        self.assertEqual([len(branches[a].observations) for a in seq.ACTIONS],[2,2,3])
        self.assertEqual([len(branches[a].history) for a in seq.ACTIONS],[2,3,3])
        self.assertTrue(all(2 in b.poses for b in branches.values()))
        self.assertEqual(branches['TRACK_ONLY'].budget.spent['depth_calls'],0)
        self.assertEqual(branches['NEW_KEYFRAME'].budget.spent['depth_calls'],1)

    def test_future_depth_sources_follow_each_branch(self):
        start,p=self.initial();candidate=frame(2)
        skip,keep=start.fork(),start.fork()
        with mock.patch.object(seq,'integration_seconds',return_value=0.),mock.patch.object(seq,'register',side_effect=found):
            skip.act(candidate,'SKIP',found(candidate,[],30),.001,p)
            keep.act(candidate,'NEW_KEYFRAME',found(candidate,[],30),.001,p)
            for b in [skip,keep]:b.continue_with([frame(3)],p,30,.01)
        self.assertNotIn('c/2',skip.depth_contexts[-1]['sources'])
        self.assertIn('c/2',keep.depth_contexts[-1]['sources'])
        self.assertEqual(start.depth_contexts,[])

    def test_exhausted_depth_budget_does_not_infer(self):
        start,p=self.initial(calls=0);n=len(p.calls);v=frame(2)
        start.act(v,'NEW_KEYFRAME',found(v,[],30),.001,p)
        self.assertEqual(len(p.calls),n)
        self.assertEqual(start.status,'over_budget')

    def test_records_are_paired_and_explicitly_offline(self):
        start,p=self.initial();views=[frame(i) for i in range(5)]
        clip=SimpleNamespace(name='c',scene='AMtown01')
        settings={'budgets':{'tight':dict(seconds=20.,depth_calls=3,matches=64)},
                  'registration_min_inliers':30,'keyframe_baseline_ratio':.01,'voxel_m':.5,'_role':'train'}
        stream=io.StringIO()
        recorder=SimpleNamespace(backend={'prefix_pose_source':'full-clip SfM'},weights=DEFAULT_WEIGHTS,
                                 status='QUALIFIED',done=set(),stream=stream,written=0)
        reference=SimpleNamespace(identity=lambda:'fixed',protocol='p')
        def scores(branches,*args):
            return {name:{'metrics':{'recall':len(b.observations)/10},'quality':len(b.observations)/10,'pose':None} for name,b in branches.items()}
        with tempfile.TemporaryDirectory() as folder, mock.patch('adaptive_uav.counterfactual_replay.plan',
                return_value=iter([(views[:2],views[2],views[3:])])),mock.patch.object(seq,'register',side_effect=found),\
                mock.patch.object(seq,'integration_seconds',return_value=0.),mock.patch.object(seq,'score_branches',side_effect=scores):
            output=Path(folder);(output/'features').mkdir()
            result=seq.run_sequential_clip(clip,settings,reference,recorder,Descriptors(),output,1,p)
            records=[json.loads(s) for s in stream.getvalue().splitlines()]
            groups,_=build(records,.0001)
            self.assertEqual(len(records),3);self.assertEqual(len(groups),1)
            self.assertEqual({r['action'] for r in records},set(seq.ACTIONS))
            self.assertEqual(len({r['reference_identity'] for r in records}),1)
            self.assertTrue(all(not r['detail']['causal_online_evidence'] for r in records))
            self.assertEqual(result['branches'],3)
            self.assertTrue(list((output/'snapshots').glob('*.json')))


class ScaleAndGateTests(unittest.TestCase):
    def test_scale_rejects_missing_or_insufficient_anchors(self):
        for anchors in [None,np.zeros((8,8)),np.ones((2,2))]:
            with self.assertRaises(ValueError):anchor_scale(anchors)

    def test_prediction_uses_anchor_units_and_postfit(self):
        class Doubled(torch.nn.Module):
            def forward(self,rgb,k,p):
                self.pose=p
                return torch.full((1,1,192,256),2.),torch.ones(1,1,192,256)
        model=Doubled();v=frame(5)
        depth,_=predict(model,[(v.rgb,v.K,v.feature_pose)],10.,'cpu',anchors=v.anchors_m)
        np.testing.assert_allclose(depth,10.)
        self.assertAlmostEqual(float(model.pose[0,0,0,3]),-.5)

    def test_sparse_anchor_grid_does_not_become_dense_supervision(self):
        sparse=np.zeros((192,256),np.float32);sparse.flat[:30]=10.
        self.assertEqual(anchor_scale(sparse),10.)
        self.assertEqual(np.count_nonzero(sparse),30)

    def test_offline_success_is_not_streaming_promotion(self):
        rows=[{'clip':str(i),'scene':str(i),'policy':p,'quality':.7 if p=='utility' else .5,
               'reference_status':'QUALIFIED','keyframes':3} for i in range(3) for p in ['rule','coverage_gain','utility']]
        result=promotion(rows,['rule','coverage_gain','utility'],.005)
        self.assertTrue(result['development_candidate'])
        self.assertFalse(result['promote'])

    def test_bare_or_stale_promotion_does_not_activate_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'best.pt';p.write_bytes(b'weights')
            self.assertFalse(allowed({'promote':True},p,'selector'))
            self.assertFalse(allowed({'beat_teacher':True},p,'student'))

    def test_empty_legacy_done_marker_is_stale(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x.done';p.touch()
            self.assertFalse(matches(p,'new'))
            p.write_text(json.dumps({'signature':'new'}))
            self.assertTrue(matches(p,'new'))
            self.assertFalse(matches(p,'changed'))

    def test_changed_output_invalidates_a_marker(self):
        from adaptive_uav.utility_contracts import sha256_file
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'best.pt';data.write_bytes(b'old')
            marker=Path(folder)/'x.done'
            marker.write_text(json.dumps({'signature':'s','artifacts':{str(data):sha256_file(data)}}))
            self.assertTrue(matches(marker,'s'))
            data.write_bytes(b'new')
            self.assertFalse(matches(marker,'s'))

    def test_old_five_question_checkpoints_remain_loadable(self):
        from adaptive_uav.graph_selector import MemoryUtilitySelector,load_utility_checkpoint
        model=MemoryUtilitySelector(questions=5)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'old.pt'
            torch.save({'architecture_id':model.architecture_id,'model_config':{},'model':model.state_dict()},p)
            loaded,_=load_utility_checkpoint(p,'cpu')
            self.assertEqual(tuple(loaded.question.weight.shape),(5,128))


class LauncherSafetyTests(unittest.TestCase):
    def exercise(self,wait):
        original=Path(__file__).resolve().parents[2]/'run_complete_training.sh'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shutil.copy2(original,root/'run_complete_training.sh')
            (root/'.venv/bin').mkdir(parents=True);(root/'bin').mkdir()
            python=root/'.venv/bin/python'
            python.write_text("#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\n"
                "if sys.argv[1]=='-':\n sys.stdin.read();print(json.dumps({'steps':['student']}));sys.exit(0)\n"
                "if 'stage_cache' in ' '.join(sys.argv):sys.exit(1 if 'check' in sys.argv else 0)\n"
                "Path('TRAINING_STARTED').touch()\n")
            python.chmod(0o755)
            gpu=root/'bin/nvidia-smi'
            gpu.write_text('#!/bin/sh\n[ -f GPU_CHECKED ] && exit 0\ntouch GPU_CHECKED\necho 12345\n');gpu.chmod(0o755)
            sleeper=root/'bin/sleep';sleeper.write_text('#!/bin/sh\nexit 0\n');sleeper.chmod(0o755)
            # If the launcher ever tries to kill a process, make the test fail visibly.
            for name in ['kill','pkill','killall']:
                p=root/'bin'/name;p.write_text('#!/bin/sh\ntouch SIGNAL_ATTEMPTED\nexit 99\n');p.chmod(0o755)
            env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH']}
            result=subprocess.run(['bash','run_complete_training.sh','--from','student','--to','student','--threads','1']+
                                  (['--wait-for-gpu'] if wait else []),
                                  cwd=root,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0 if wait else 3,result.stdout+result.stderr)
            self.assertEqual((root/'TRAINING_STARTED').exists(),wait)  # Dummy executable, not ML training.
            self.assertFalse((root/'SIGNAL_ATTEMPTED').exists())
            self.assertFalse((root/'runs/vnext_logs/student.done').exists())

    def test_busy_gpu_guard_never_starts_work(self):
        self.exercise(False)

    def test_wait_option_rechecks_before_starting_dummy_work(self):
        self.exercise(True)


if __name__=='__main__':unittest.main()
