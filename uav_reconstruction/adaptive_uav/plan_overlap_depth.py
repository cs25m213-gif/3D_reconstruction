"""Create overlapping image groups, then keep only posed parallax pairs for depth."""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

from .dense_views import pair_quality
from .overlap_attention import make_bins
from .template_overlap import score_history


def make_plan(candidate_file,output,overlap_threshold=.09,source_views=2):
    data=json.loads(Path(candidate_file).read_text());rows=data['frames']
    scores,_=score_history([row['image'] for row in rows])
    groups,membership=make_bins(range(len(rows)),scores,threshold=overlap_threshold)
    reconstruction=pycolmap.Reconstruction(data['report']['sparse'])
    tracks=[set(row.get('pnp_inlier_point_ids',())) if row['pnp_verified'] else set() for row in rows]
    centers=[]
    for row in rows:
        if row['pnp_verified']:
            pose=np.asarray(row['world_to_camera'],float)
            centers.append(-pose[:3,:3].T@pose[:3,3])
        else:centers.append(None)
    plans=[]
    for i,row in enumerate(rows):
        if centers[i] is None:continue
        neighbors=set(j for group in membership[i] for j in groups[group].members if j!=i)
        options=[]
        for j in neighbors:
            if centers[j] is None:continue
            score=scores.get((max(i,j),min(i,j)),0.)
            if score<overlap_threshold:continue
            common=sorted(tracks[i]&tracks[j])
            if len(common)<30:continue
            sampled=common[::max(1,len(common)//64)][:64]
            xyz=np.asarray([reconstruction.points3D[point].xyz for point in sampled])
            angle=pair_quality(centers[i],centers[j],xyz)
            if not .8<=angle<=45.:continue
            options.append((len(common)*min(angle/3.,1.)*score,j,angle,len(common),score))
        options.sort(reverse=True)
        if len(options)<source_views:continue
        chosen=options[:source_views]
        plans.append({'target_frame':int(row['frame_index']),'target_image':Path(row['image']).name,
            'bin_ids':membership[i],
            'sources':[{'frame':int(rows[j]['frame_index']),'image':Path(rows[j]['image']).name,
                'shared_verified_points':count,'median_angle_deg':angle,'image_overlap_score':overlap}
                for _,j,angle,count,overlap in chosen]})
    valid_frames=[row['frame_index'] for row in rows if row['pnp_verified']]
    planned=[plan['target_frame'] for plan in plans]
    result={'scene':Path(candidate_file).parent.name,'candidate_frames':len(rows),
        'verified_poses':len(valid_frames),'image_bins':len(groups),
        'multi_bin_frames':sum(len(value)>1 for value in membership.values()),
        'depth_targets':len(plans),'depth_target_fraction_of_verified':len(plans)/max(1,len(valid_frames)),
        'depth_time_span_fraction':(max(planned)-min(planned))/(max(valid_frames)-min(valid_frames)) if len(planned)>1 else 0.,
        'overlap_threshold':overlap_threshold,'source_views':source_views,
        'camera_convention':'world_to_camera as recorded in verified PnP candidates',
        'depth_scale':'must be derived from sparse anchors at depth inference',
        'groups':[{'anchor_frame':int(rows[group.members[0]]['frame_index']),
            'member_frames':[int(rows[j]['frame_index']) for j in group.members]}
            for group in groups],
        'plans':plans,
        'limitation':'Only image overlap proposes groups. Verified PnP tracks and camera parallax gate depth pairs; no teacher depth or mesh has been computed by this command.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return {key:result[key] for key in ('scene','candidate_frames','verified_poses','image_bins','multi_bin_frames','depth_targets','depth_time_span_fraction')}


def workspace_groups(workspace,names,by_name,points,source_views=2,overlap_threshold=.09):
    """Propose image bins, then require verified tracks and parallax for depth.

    Returns one target per group, so a target may use sources from overlapping
    bins without estimating its depth more than once.
    """
    from .multiview_dense import source_groups
    workspace=Path(workspace);names=list(names)
    scores,_=score_history([workspace/'images'/name for name in names])
    bins,membership=make_bins(range(len(names)),scores,threshold=overlap_threshold)
    tracks={name:{point.point3D_id for point in by_name[name].points2D if point.has_point3D()}
        for name in names}
    centers={name:np.asarray(by_name[name].cam_from_world().inverse().translation,float) for name in names}
    old=source_groups(names,by_name,points,max(3,source_views+1))
    fallback={target:[target]+[other for other in group if other!=target][:source_views]
        for group,targets in old for target in targets}
    groups=[];details=[];proposed=0
    for i,name in enumerate(names):
        neighbors={j for number in membership[i] for j in bins[number].members if j!=i}
        options=[]
        for j in neighbors:
            other=names[j];visual=scores.get((max(i,j),min(i,j)),0.)
            if visual<overlap_threshold:continue
            shared=sorted(tracks[name]&tracks[other])
            if len(shared)<30:continue
            sampled=shared[::max(1,len(shared)//64)][:64]
            angle=pair_quality(centers[name],centers[other],
                np.asarray([points[point].xyz for point in sampled]))
            if not .8<=angle<=45.:continue
            options.append((len(shared)*min(angle/3.,1.)*visual,other,angle,len(shared),visual))
        options.sort(reverse=True)
        if len(options)>=source_views:
            group=[name]+[option[1] for option in options[:source_views]];proposed+=1
            method='overlap_bin_verified'
        else:
            group=fallback[name];method='existing_geometric_fallback'
        groups.append((group,[name]))
        details.append({'target':name,'sources':group[1:],'bin_ids':membership[i],
            'selection':method,'verified_options':len(options)})
    report={'workspace':str(workspace),'views':len(names),'bins':len(bins),
        'bin_sizes':[len(b.members) for b in bins],
        'multi_bin_frames':sum(len(value)>1 for value in membership.values()),
        'verified_bin_targets':proposed,'geometric_fallback_targets':len(names)-proposed,
        'groups':details,'overlap_threshold':overlap_threshold,
        'note':'SIFT proposes co-visibility; shared SfM tracks and parallax verify depth pairs.'}
    return groups,report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=Path,default=Path('runs/selector_candidates'))
    parser.add_argument('--output',type=Path,default=Path('runs/overlap_depth_plans'))
    args=parser.parse_args()
    for source in sorted(args.candidates.glob('*/candidates.json')):
        print(json.dumps(make_plan(source,args.output/(source.parent.name+'.json'))),flush=True)


if __name__=='__main__':main()
