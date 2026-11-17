"""One bounded real-cache CPU replay decision. No training, GPU or video reconstruction."""
import argparse
import io
import json
from pathlib import Path
import resource
import time

import torch

from adaptive_uav.counterfactual_replay import Clip, Recorder
from adaptive_uav.embedding_cache import Descriptors
from adaptive_uav.reconstruction_quality import DEFAULT_WEIGHTS, ReferencePopulation
from adaptive_uav.sequential_replay import StudentDepthProvider, run_sequential_clip
from adaptive_uav.student_inference import load_student
from adaptive_uav.utility_contracts import load_contract, sha256_file, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('uav_reconstruction/configs/vnext_contract_v3.json'))
    p.add_argument('--clip',default='AMtown01_1')
    p.add_argument('--checkpoint',type=Path,default=Path('runs/depth_student_lidar_v2_augmented/best.pt'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    started=time.monotonic();contract=load_contract(a.config)
    clip=Clip(Path('.').resolve(),contract,a.clip)
    clip.choose_keyframes(.02)
    settings={**contract['utility'],'prefix_keyframes':[2],'candidate_baseline_ratios':[.02],
              'rollout_frames':2,'voxel_m':1.,'_role':'train',
              'budgets':{'smoke':dict(seconds=120.,depth_calls=2,matches=16)}}
    gate=json.loads(Path(contract['geometry_gate']).read_text())
    status=gate['flights'][clip.scene]['status']
    if status not in ('VERIFIED','QUALIFIED'):
        raise SystemExit('Smoke clip has not qualified geometry')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'features').mkdir()
    reference=ReferencePopulation(clip.points,1.,status)
    provider=StudentDepthProvider(load_student(a.checkpoint,'cpu'),'cpu')
    backend={'depth_backend':'student_branch_context','device':'cpu','checkpoint_sha256':sha256_file(a.checkpoint),
             'prefix_pose_source':'full-clip SfM aligned to reference; offline development only',
             'causal_online_evidence':False}
    with (a.output/'labels.jsonl').open('w') as stream:
        recorder=Recorder(stream,clip,reference,status,settings,DEFAULT_WEIGHTS,backend,set())
        result=run_sequential_clip(clip,settings,reference,recorder,Descriptors(),a.output,1,provider)
    records=[json.loads(s) for s in (a.output/'labels.jsonl').read_text().splitlines()]
    if len(records)!=3 or any(r['execution_status']=='censored' for r in records):
        raise SystemExit('Smoke did not produce three uncensored branches: '+json.dumps(result))
    report={'clip':a.clip,'device':'cpu','training_steps':0,'model_forward_calls':provider.calls,
            'records':len(records),'actions':{r['action']:r['execution_status'] for r in records},
            'candidate_registration_failed':records[0]['detail']['candidate_registration_failed'],
            'depth_contexts':{r['action']:r['detail']['depth_contexts'] for r in records},
            'seconds':time.monotonic()-started,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'scope':'Bounded real-cache CPU integration smoke; not geometry improvement, a full-video test or production promotion',
            'replay':result,'settings':settings}
    write_json(a.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
