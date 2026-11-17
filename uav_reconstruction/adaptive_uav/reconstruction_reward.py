"""Counterfactual frame utility from verified PnP tracks and camera geometry.

All candidate observations are existing SfM points confirmed by PnP inliers.
Labels assess a budgeted camera subset. They are graph/triangulation proxies,
not a rerun of COLMAP or a claim of measured mesh improvement.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pycolmap


def supported_points(nodes,tracks,centers,xyz,min_angle=2.):
    observations=defaultdict(list)
    for i in nodes:
        for point in tracks[i]:
            if point in xyz:observations[point].append(centers[i])
    count=0
    for point,locations in observations.items():
        if len(locations)<2:continue
        vectors=np.asarray(locations)-xyz[point]
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-9)
        similarity=np.clip(vectors@vectors.T,-1,1)
        if np.degrees(np.arccos(similarity.min()))>=min_angle:count+=1
    return count


def graph(nodes,tracks,threshold):
    result=nx.Graph();result.add_nodes_from(nodes)
    for pos,i in enumerate(nodes):
        for j in nodes[:pos]:
            shared=len(tracks[i]&tracks[j])
            if shared>=threshold:result.add_edge(i,j,shared=shared)
    return result


def connected_pairs(g,nodes):
    count=0
    for component in nx.connected_components(g):
        members=len(set(component)&set(nodes))
        count+=members*(members-1)//2
    return count


def build(scene_file,feature_file,output):
    data=json.loads(Path(scene_file).read_text());rows=data['frames']
    model=pycolmap.Reconstruction(data['report']['sparse'])
    tracks=[set(row.get('pnp_inlier_point_ids',())) if row['pnp_verified'] else set() for row in rows]
    centers=[]
    for row in rows:
        if row['pnp_verified']:
            pose=np.asarray(row['world_to_camera'],float)
            centers.append(-pose[:3,:3].T@pose[:3,3])
        else:centers.append(np.zeros(3))
    xyz={point:np.asarray(model.points3D[point].xyz,float)
         for track in tracks for point in track if point in model.points3D}
    with np.load(feature_file) as features:
        visual=features['visual'].copy()
    if len(visual)!=len(rows):raise ValueError('Visual features do not match candidate frames')
    valid=[i for i,track in enumerate(tracks) if len(track)>=12]
    adjacent=[len(tracks[a]&tracks[b]) for a,b in zip(valid,valid[1:])]
    threshold=max(15,int(.15*np.median(adjacent)))
    depth_scale=np.median([np.linalg.norm(xyz[p]-centers[i])
        for i in valid[::max(1,len(valid)//20)] for p in list(tracks[i])[:20] if p in xyz])
    if not np.isfinite(depth_scale) or depth_scale<=0:raise ValueError('Invalid scene depth scale')
    output_rows=[]
    for stride in (2,3,4):
        for i,row in enumerate(rows):
            if i not in valid:continue
            context=[j for j in range(i-1,-1,-stride) if j in valid][:6][::-1]
            if len(context)<2:continue
            future=[j for j in range(i+1,min(len(rows),i+1+stride*4),stride) if j in valid][:3]
            old=context+future
            before=graph(old,tracks,threshold)
            after=graph(old+[i],tracks,threshold)
            delta_connectivity=connected_pairs(after,old)-connected_pairs(before,old)
            if delta_connectivity<0:raise RuntimeError('Adding a frame reduced graph connectivity')
            before_support=supported_points(context,tracks,centers,xyz)
            after_support=supported_points(context+[i],tracks,centers,xyz)
            delta_geometry=max(0,after_support-before_support)
            previous=context[-1]
            motion=np.linalg.norm(centers[i]-centers[previous])/depth_scale
            pose_i=np.asarray(row['world_to_camera'],float)
            pose_p=np.asarray(rows[previous]['world_to_camera'],float)
            relative=pose_p[:3,:3].T@pose_i[:3,:3]
            rotation=np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))
            overlap=len(tracks[i]&tracks[previous])/max(1,len(tracks[i]))
            state=[min((i-previous)*.5,3.)/3.,np.log1p(motion),min(rotation,1.5)/1.5,
                   overlap,min(row['sharpness'],300.)/300.,
                   min(nx.number_connected_components(graph(context,tracks,threshold)),6)/6.,
                   min(before_support/1000.,1.),1.]
            gain=delta_geometry/max(1,before_support)
            if delta_connectivity>0 or delta_geometry>=10 and gain>=.05:action=0
            elif delta_geometry<=2 and delta_connectivity==0 and overlap>=.6:action=2
            else:action=1
            output_rows.append((i,stride,state,[float(overlap),float(gain),float(delta_connectivity)],
                                action,delta_connectivity>0,delta_geometry))
    if not output_rows:raise ValueError('No verified candidate contexts')
    np.savez_compressed(output,visual=np.stack([visual[i] for i,*_ in output_rows]),
        state=np.asarray([row[2] for row in output_rows],np.float32),
        utility=np.asarray([row[3] for row in output_rows],np.float32),
        label=np.asarray([row[4] for row in output_rows],np.int64),
        bridge=np.asarray([row[5] for row in output_rows],bool),
        delta_geometry=np.asarray([row[6] for row in output_rows],np.int32),
        frame_index=np.asarray([rows[row[0]]['frame_index'] for row in output_rows],np.int32),
        stride=np.asarray([row[1] for row in output_rows],np.int8))
    return {'scene':Path(scene_file).parent.name,'examples':len(output_rows),
        'verified_bridge_examples':sum(row[5] for row in output_rows),
        'actions':np.bincount([row[4] for row in output_rows],minlength=4).tolist(),
        'shared_track_threshold':threshold,'depth_scale_visual_units':float(depth_scale),
        'reward':'connected camera pairs after insertion; points supported by >=2 views with >=2 degree parallax; fixed one-frame compute cost',
        'depth_improvement':'not measured for these candidate windows'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-root',type=Path,default=Path('runs/selector_candidates'))
    p.add_argument('--feature-root',type=Path,default=Path('runs/cnn_gru_selector_candidates/features'))
    p.add_argument('--output',type=Path,default=Path('runs/reconstruction_reward'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for source in sorted(args.candidate_root.glob('*/candidates.json')):
        scene=source.parent.name
        report=build(source,args.feature_root/(scene+'.npz'),args.output/(scene+'.npz'))
        reports.append(report);print(json.dumps(report),flush=True)
    (args.output/'report.json').write_text(json.dumps({'scenes':reports,
        'limitations':'Counterfactuals use verified existing SfM landmarks. They are not fresh COLMAP registration, dense depth, or mesh outcomes.'},indent=2)+'\n')


if __name__=='__main__':main()
