"""Bounded stereo requests, separate from capture-wide evaluation selection."""
from dataclasses import dataclass,field
import numpy as np


@dataclass
class StereoPolicy:
    max_references: int=12
    budget_seconds: float=60.
    requested: set=field(default_factory=set)
    elapsed_seconds: float=0.

    def request(self,name,*,parallax_degrees,scale_uncertain=False,depth_error=None,new_area=False):
        if name in self.requested:return False,'already_requested'
        if len(self.requested)>=self.max_references or self.elapsed_seconds>=self.budget_seconds:return False,'budget'
        if parallax_degrees<3:return False,'insufficient_parallax'
        if not (scale_uncertain or new_area or depth_error is not None and depth_error>.1):return False,'no_refinement_needed'
        self.requested.add(name);return True,'uncertain_geometry'

    def record(self,seconds):
        if not np.isfinite(seconds) or seconds<0:raise ValueError('Invalid stereo time')
        self.elapsed_seconds+=seconds


def evaluation_frames(names,count=16):
    if not names:return []
    return [names[i] for i in np.unique(np.linspace(0,len(names)-1,min(count,len(names)),dtype=int))]


def plan_pose_stereo(workspace,frames,poses,depth_frames,max_references=12,source_views=3):
    """Use predicted surface points only for pair proposal, not fake SfM tracks."""
    from pathlib import Path
    from .runtime_metrics import save_json
    from .dense_views import pair_quality
    workspace=Path(workspace);selected=evaluation_frames(sorted(depth_frames),64)
    centers={n:poses[n][:3,3] for n in selected};plan=[];ranges=[]
    for name in selected:
        frame=depth_frames[name];yy,xx=np.where(frame.valid)
        if len(xx)<100:continue
        stride=max(1,len(xx)//128);yy,xx=yy[::stride],xx[::stride]
        z=frame.depth[yy,xx];k=frame.K
        camera=np.column_stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z))
        world=camera@poses[name][:3,:3].T+poses[name][:3,3];sources=[]
        ranges.extend(np.percentile(z,[2,98]).tolist())
        for other in selected:
            if name==other:continue
            p=np.linalg.inv(poses[other]);xyz=world@p[:3,:3].T+p[:3,3];ik=np.asarray(frames[other]['K'])
            uv=xyz@ik.T;uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)
            valid=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<frames[other]['width'])&(uv[:,1]>=0)&(uv[:,1]<frames[other]['height'])
            if valid.mean()<.2:continue
            angle=pair_quality(centers[name],centers[other],world[valid])
            if 3<=angle<=45:sources.append({'name':other,'median_angle_deg':angle,'score':float(valid.mean()*min(angle/5,1))})
        if len(sources)>=2:plan.append({'reference':name,'sources':sorted(sources,key=lambda x:x['score'],reverse=True)})
    # Select connected triplets, rather than isolated temporal samples. The
    # independent evaluation grid still spans the whole recording.
    rows={row['reference']:row for row in plan}
    neighbors={name:{s['name'] for s in row['sources']} for name,row in rows.items()}
    chosen=set()
    for group in np.array_split(sorted(rows),max(1,max_references//3)):
        found=False
        for a in sorted(group,key=lambda n:len(neighbors[n]),reverse=True):
            possible=[s['name'] for s in rows[a]['sources'] if s['name'] in rows]
            for i,b in enumerate(possible):
                for c in possible[i+1:]:
                    if {a,c}<=neighbors[b] and {a,b}<=neighbors[c] and len(chosen|{a,b,c})<=max_references:
                        chosen|={a,b,c};found=True;break
                if found:break
            if found:break
    for a in sorted(rows,key=lambda n:len(neighbors[n]),reverse=True):
        if len(chosen)>=max_references:break
        if len(neighbors[a]&chosen)>=2:chosen.add(a)
    plan=[row for row in plan if row['reference'] in chosen]
    for row in plan:row['sources']=[s for s in row['sources'] if s['name'] in chosen][:source_views]
    # Geometric PatchMatch needs source depth maps; retain a connected active set.
    active={row['reference'] for row in plan}
    while True:
        keep={row['reference'] for row in plan if sum(s['name'] in active for s in row['sources'])>=2}
        if keep==active:break
        active=keep;plan=[row for row in plan if row['reference'] in active]
    for row in plan:row['sources']=[s for s in row['sources'] if s['name'] in active]
    report={'plan':plan,'reference_views':len(plan),'max_reference_views':max_references,'candidate_views':len(selected),
        'pair_proposal_evidence':'learned surface projection; no sparse tracks fabricated',
        'depth_min':max(1e-4,min(ranges)*.25) if ranges else None,'depth_max':max(ranges)*3 if ranges else None}
    save_json(workspace/'dense_plan.json',report)
    (workspace/'stereo').mkdir(exist_ok=True)
    (workspace/'stereo/patch-match.cfg').write_text(''.join(row['reference']+'\n'+', '.join(s['name'] for s in row['sources'])+'\n' for row in plan))
    return report


def execute_stereo(workspace,max_size=640,iterations=2):
    import json
    from pathlib import Path
    import pycolmap
    workspace=Path(workspace);plan=json.loads((workspace/'dense_plan.json').read_text())
    if plan['reference_views']<3:raise ValueError('Insufficient connected stereo references')
    options=pycolmap.PatchMatchOptions(max_image_size=max_size,num_iterations=iterations,
        depth_min=plan['depth_min'],depth_max=plan['depth_max'],gpu_index='0',cache_size=1.,
        geom_consistency=True,filter_min_triangulation_angle=3.,num_threads=4)
    pycolmap.patch_match_stereo(workspace,options=options)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--max-size',type=int,default=640);p.add_argument('--iterations',type=int,default=2)
    args=p.parse_args();execute_stereo(args.workspace,args.max_size,args.iterations)
