import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch
import cv2,numpy as np,torch
from adaptive_uav.selector_corruptions import CASES,corrupt,frame_seed,quality_vector,usable_rgb
from adaptive_uav.robust_selector_policy import QualityAdapter,RobustPolicy
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('robust_training',ROOT/'scripts/train_robust_selector.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def image():
 rng=np.random.default_rng(9);return rng.integers(0,256,(64,96,3),dtype=np.uint8)
def view(index=0):
 rgb=image();rgb[0,0,0]=index;k=np.array([[60.,0,48.],[0,60.,32.],[0,0,1.]])
 return SimpleNamespace(index=index,rgb=rgb,K=k,feature_k=k.copy(),feature_pose=np.eye(4),reference_pose=np.eye(4),anchors_m=np.ones((64,96),np.float32),frame_id=f'c/{index}',timestamp=index/2,depth_m=np.ones((64,96)),fusion_depth=np.ones((64,96)),fusion_rgb=rgb,fusion_k=k,confidence=1.)
class DummyState:
 def features(self,v,p,b,d):return {'visual':d,'query':np.zeros(17,np.float32),'tokens':np.zeros((1,11),np.float32),'token_visual':np.zeros((1,1),np.float32)}
 def commit(self,*args):pass
 def observe(self,*args):pass
class Model(torch.nn.Module):
 def forward(self,visual,*args):
  score=visual[:,:,0];return {'utility':torch.stack([score*0,score*0,score*0,score],-1)}
class Descriptors:
 def get(self,name,index,rgb):return np.float32([index])
class RobustnessTests(unittest.TestCase):
 def test_repeated_rgb_does_not_consume_another_depth_selection(self):
  views=[view(i) for i in range(4)];views[1].rgb=views[0].rgb.copy();views[2].rgb=views[0].rgb.copy()
  policy=RobustPolicy(Model(),{'decision_head':0},'cpu',QualityAdapter())
  with patch('adaptive_uav.robust_selector_policy.MapState',DummyState):selected,_,_=policy.run(SimpleNamespace(name='c',views=views),4,Descriptors())
  self.assertEqual([v.index for v in selected],[0,3]);self.assertEqual([r['action'] for r in policy.last_trace],['SELECT','SKIP','SKIP','SELECT'])
 def test_teacher_student_use_same_pixels_and_missing_predictions_are_penalized(self):
  target=np.array([[10.,10.,10.]])
  result=runner.paired_depth_metrics(np.array([[10.,0.,np.nan]]),np.array([[12.,10.,10.]]),target,np.ones_like(target,bool))
  self.assertEqual(result['shared_valid_pixels'],1)
  self.assertEqual(result['student']['common_valid_metrics']['pixels'],result['teacher']['common_valid_metrics']['pixels'])
  self.assertEqual(result['student']['invalid_prediction_pixels'],2)
  self.assertAlmostEqual(result['student']['full_population_absrel'],2/3)
 def test_freeze_repeats_one_image_for_entire_burst(self):
  views=[view(i) for i in range(5)]
  for i,v in enumerate(views):v.rgb[:]=20+i*20
  source=SimpleNamespace(name='c',views=views,points=np.ones((10,3)),raw_frame_count=5)
  def fault(case,seed,name,index,count,severity):return ('frozen',severity) if 1<=index<=3 else ('clean',0.)
  with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'frame_fault',fault),patch.object(runner,'image_feature',lambda *a:np.zeros(384,np.float32)),patch.object(runner,'predict',lambda *a,**k:(np.ones((192,256),np.float32),np.ones((192,256),np.float32))):
   clip=runner.prepare(source,'frozen',.9,194,None,None,'cpu',Path(tmp))
  for i in (1,2,3):np.testing.assert_array_equal(clip.views[i].rgb,views[0].rgb)
  np.testing.assert_array_equal(clip.views[4].rgb,views[4].rgb)
 def test_batched_rewards_reuse_equal_selections_and_keep_independent_contexts(self):
  clip=SimpleNamespace(views=[view(i) for i in range(3)],points=np.ones((10,3)),reward_cache={})
  contexts=[]
  def depth(v,history,*args):
   contexts.append((v.index,tuple(h.index for h in history)));out=copy.copy(v);out.depth_failure=None;out.depth_inference_called=True;return out
  metric={'f_score':.5,'supported_coverage':.4,'accuracy_m':.1}
  choices={'a':clip.views[:2],'same':clip.views[:2],'different':clip.views[1:]}
  with patch.object(runner,'depth_view',depth),patch.object(runner,'geometry_metrics_batch',return_value=[metric,metric]) as remote:
   result=runner.reward_batch(clip,choices,None,'cpu',{'utility':{'tolerance_m':.1,'voxel_m':.5}})
  self.assertEqual(contexts,[(0,()),(1,(0,)),(1,()),(2,(1,))])
  self.assertEqual(result['same']['current_depth_inference_calls'],0)
  self.assertEqual(result['a']['current_depth_inference_calls'],2)
  self.assertEqual(len(remote.call_args.args[0]),2)
 def test_all_corruptions_preserve_geometry_shape_and_original(self):
  original=image();saved=original.copy()
  for case in CASES:
   result,mask=corrupt(original,case,.8,17);self.assertEqual(result.shape,original.shape);self.assertEqual(result.dtype,np.uint8);self.assertEqual(mask.shape,original.shape[:2]);np.testing.assert_array_equal(original,saved)
 def test_faults_are_reproducible_and_split_seeds_differ(self):
  a,_=corrupt(image(),'dark_noise',.8,frame_seed(194,'c',1));b,_=corrupt(image(),'dark_noise',.8,frame_seed(194,'c',1));c,_=corrupt(image(),'dark_noise',.8,frame_seed(9137,'c',1));np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,c))
 def test_blur_reduces_high_frequency_detail(self):
  original=image();blur,_=corrupt(original,'motion_blur',1.,17);self.assertLess(quality_vector(blur)[0],quality_vector(original)[0])
 def test_occlusion_masks_anchor_support(self):
  rgb,mask=corrupt(image(),'occlusion',1.,17);self.assertGreater((~mask).mean(),.4);self.assertGreater(mask.mean(),.3)
 def test_weak_texture_is_not_a_hard_rejection(self):
  self.assertTrue(usable_rgb(np.full((24,24,3),120,np.uint8)));self.assertFalse(usable_rgb(np.zeros((24,24,3),np.uint8)))
 def test_features_remain_finite_on_blank(self):
  self.assertTrue(np.isfinite(quality_vector(np.zeros((24,24,3),np.uint8))).all())
 def test_first_window_is_ranked(self):
  clip=SimpleNamespace(name='c',views=[view(0),view(1),view(2)]);policy=RobustPolicy(Model(),{'decision_head':0},'cpu',QualityAdapter())
  with patch('adaptive_uav.robust_selector_policy.MapState',DummyState):selected,_,_=policy.run(clip,1,Descriptors())
  self.assertEqual([v.index for v in selected],[2])
 def test_all_bad_window_abstains(self):
  v=view();v.rgb.fill(0);clip=SimpleNamespace(name='c',views=[v]);policy=RobustPolicy(Model(),{'decision_head':0},'cpu',QualityAdapter())
  with patch('adaptive_uav.robust_selector_policy.MapState',DummyState):selected,_,_=policy.run(clip,1,Descriptors())
  self.assertFalse(selected);self.assertEqual(policy.last_trace[0]['action'],'ABSTAIN')
 def test_adapter_receives_reward_gradient(self):
  adapter=QualityAdapter();features=torch.randn(3,12);loss=torch.distributions.Categorical(logits=adapter(features)).log_prob(torch.tensor(1));loss.backward();self.assertGreater(float(adapter.net[-1].weight.grad.abs().sum()),0)
 def test_invalid_tracking_requests_relocalization(self):
  v=view();v.tracking_state='lost';policy=RobustPolicy(Model(),{'decision_head':0},'cpu',QualityAdapter())
  with patch('adaptive_uav.robust_selector_policy.MapState',DummyState):selected,_,_=policy.run(SimpleNamespace(name='c',views=[v]),1,Descriptors())
  self.assertFalse(selected);self.assertEqual(policy.last_trace[0]['action'],'RELOCALIZE')
 def test_frozen_frame_loses_current_camera_anchor_support(self):
  source=SimpleNamespace(name='c',views=[view(i) for i in range(4)],points=np.ones((10,3)),raw_frame_count=4)
  with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'image_feature',lambda *a:np.zeros(384,np.float32)),patch.object(runner,'predict',lambda *a,**k:(np.ones((192,256),np.float32),np.ones((192,256),np.float32))):
   clip=runner.prepare(source,'frozen',.9,194,None,None,'cpu',Path(tmp))
  damaged=[v for v in clip.views if v.depth_failure is not None]
  self.assertTrue(damaged);self.assertTrue(all(not v.anchors_m.any() and not v.depth_inference_called for v in damaged))
 def test_original_loader_rejects_adapter_checkpoint(self):
  from adaptive_uav.graph_selector import load_utility_checkpoint
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'adapter.pt';torch.save({'architecture_id':'robust_memory_utility_v1'},path)
   with self.assertRaises(ValueError):load_utility_checkpoint(path,'cpu')
 def test_corrupted_rgb_reaches_student_and_descriptor_and_source_is_preserved(self):
  v=view();original=v.rgb.copy();source=SimpleNamespace(name='c',views=[v],points=np.ones((10,3)),raw_frame_count=1);seen=[]
  def predict(model,views,*args,**kwargs):seen.append(views[0][0].copy());return np.ones((192,256),np.float32),np.ones((192,256),np.float32)
  def descriptor(model,rgb,device):seen.append(rgb.copy());return np.zeros(384,np.float32)
  with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'predict',predict),patch.object(runner,'image_feature',descriptor):
   clip=runner.prepare(source,'motion_blur',.9,194,None,None,'cpu',Path(tmp))
  self.assertEqual(len(seen),2);self.assertFalse(np.array_equal(seen[0],original));np.testing.assert_array_equal(seen[0],seen[1]);np.testing.assert_array_equal(v.rgb,original);np.testing.assert_array_equal(clip.views[0].reference_pose,v.reference_pose)
 def test_temporal_gaps_keep_original_reference_population_and_indices(self):
  source=SimpleNamespace(name='c',views=[view(i) for i in range(7)],points=np.ones((10,3)),raw_frame_count=7)
  with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'predict',lambda *a,**k:(np.ones((192,256),np.float32),np.ones((192,256),np.float32))),patch.object(runner,'image_feature',lambda *a:np.zeros(384,np.float32)):
   clip=runner.prepare(source,'frame_gap',.9,194,None,None,'cpu',Path(tmp))
  self.assertEqual([v.index for v in clip.views],[0,3,6]);self.assertIs(clip.points,source.points)
if __name__=='__main__':unittest.main()
