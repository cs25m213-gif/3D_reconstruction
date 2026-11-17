"""Choose a retrained student using validation flights only; leave test untouched."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .depth_student import DepthStudent
from .train_depth_student import TeacherCacheDataset,discover,evaluate,split_flights


def measure(path,paths,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=DepthStudent(**checkpoint.get('architecture',{'channels':48,'planes':24})).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    loader=DataLoader(TeacherCacheDataset(paths),batch_size=2,num_workers=2)
    return evaluate(model,loader,device)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path('runs/uavscenes_30_batch'))
    parser.add_argument('--original',type=Path,default=Path('runs/depth_student_20261028/best.pt'))
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths,flights=split_flights(discover(args.data_root),42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original=measure(args.original,paths['validation'],device)
    candidate=measure(args.candidate,paths['validation'],device)
    improved=(candidate['anchor_absrel']<.95*original['anchor_absrel'] and
              candidate['teacher_absrel']<=1.05*original['teacher_absrel'])
    chosen=args.candidate if improved else args.original
    report={'selected':str(chosen.resolve()),'candidate_promoted':improved,
        'original_validation':original,'candidate_validation':candidate,
        'validation_flights':flights['validation'],
        'decision':'At least 5% better sparse-anchor AbsRel without >5% worse teacher AbsRel',
        'limitation':'Sparse anchors share estimated poses; independent geometry not yet tested.'}
    (args.output/'selected_depth_checkpoint.txt').write_text(str(chosen.resolve())+'\n')
    (args.output/'checkpoint_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
