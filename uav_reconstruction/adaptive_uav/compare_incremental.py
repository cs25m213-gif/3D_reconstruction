"""Compare meshes on identical baseline rays after one held-out-tested Sim(3).

The existing reconstruction is a comparison reference, never ground truth.
Alignment is global, never a per-frame scale fit that could hide drift.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from .runtime_metrics import save_json
from .geometry import weighted_umeyama
from .hybrid_dense import pose4
from .multiview_dense import ray_audit


def actual_timestamps(video,indices):
    indices=set(indices);result={};cap=cv2.VideoCapture(str(video));index=0
    try:
        while indices:
            if not cap.grab():break
            if index in indices:
                result[index]=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;indices.remove(index)
            index+=1
    finally:cap.release()
    return result


def trajectory(candidate):
    if not (candidate/'poses/trajectory.jsonl').exists():
        import pycolmap
        report=json.loads((candidate/'run_report.json').read_text())
        workspace=candidate/'dense_recovery'
        if not (workspace/'sparse').exists():workspace=candidate/'dense'
        reconstruction=pycolmap.Reconstruction(workspace/'sparse')
        images=sorted((im for im in reconstruction.images.values() if im.has_pose),key=lambda im:im.name)
        indices={im.name:int(Path(im.name).stem.split('_')[-1]) for im in images}
        stamps=actual_timestamps(Path(report['source'])/'video.mp4',indices.values())
        images=[im for im in images if indices[im.name] in stamps]
        return np.array([stamps[indices[im.name]] for im in images]),np.stack([np.linalg.inv(pose4(im))[:3,3] for im in images])
    rows=[json.loads(line) for line in (candidate/'poses/trajectory.jsonl').open()]
    poses={r['frame_id']:np.asarray(r['T_world_from_camera']) for r in rows if r['state']=='tracking'}
    path=candidate/'poses/corrections.jsonl'
    if path.exists():
        for line in path.open():
            for name,p in json.loads(line)['poses'].items():
                if name in poses:poses[name]=np.asarray(p)
    selected=[r for r in rows if r['frame_id'] in poses]
    return np.array([r['timestamp'] for r in selected]),np.stack([poses[r['frame_id']][:3,3] for r in selected])


def compare(baseline,candidate,output):
    import pycolmap,open3d as o3d
    baseline,candidate,output=map(Path,(baseline,candidate,output));output.mkdir(parents=True,exist_ok=True)
    old=json.loads((baseline/'run_report.json').read_text());new=json.loads((candidate/'run_report.json').read_text())
    if Path(old['source']).resolve()!=Path(new['source']).resolve():raise ValueError('Comparison inputs belong to different sessions')
    ws=baseline/'dense_recovery'
    if not (ws/'sparse').exists():ws=baseline/'dense'
    model=pycolmap.Reconstruction(ws/'sparse');images=sorted((im for im in model.images.values() if im.has_pose),key=lambda im:im.name)
    indices={im.name:int(Path(im.name).stem.split('_')[-1]) for im in images}
    timestamps=actual_timestamps(Path(old['source'])/'video.mp4',indices.values());times,centers=trajectory(candidate)
    matched=[]
    for image in images:
        stamp=timestamps.get(indices[image.name])
        if stamp is None or stamp<times[0] or stamp>times[-1]:continue
        right=min(len(times)-1,np.searchsorted(times,stamp));left=max(0,right-1)
        if times[right]-times[left]>2:continue
        center=np.array([np.interp(stamp,times,centers[:,i]) for i in range(3)])
        matched.append((image.name,center,np.linalg.inv(pose4(image))[:3,3]))
    if len(matched)<8:raise ValueError('Insufficient overlapping tracked time for a global mesh comparison')
    train=matched[::2];hold=matched[1::2]
    scale,rotation,translation=weighted_umeyama(np.stack([r[1] for r in train]),np.stack([r[2] for r in train]))
    error=np.array([np.linalg.norm(scale*rotation@r[1]+translation-r[2]) for r in hold])
    meshes={}
    for name,folder in [('baseline',baseline),('candidate',candidate)]:
        meshes[name]=o3d.io.read_triangle_mesh(str(folder/'multiview/scene_mesh.ply'))
        if not len(meshes[name].triangles):raise ValueError('Mesh unavailable: '+name)
    affine=np.eye(4);affine[:3,:3]=scale*rotation;affine[:3,3]=translation;meshes['candidate'].transform(affine)
    frames=[]
    for image in images:
        path=baseline/'multiview/depth_cache'/(image.name+'.npz')
        if not path.exists():continue
        with np.load(path) as data:
            frames.append({'name':image.name,'prior':data['depth'],'measured':data['measured'],
                'rgb':data['rgb'],'K':data['K'],'P':data['P']})
    audit_dir=output/'common_audit';audit_dir.mkdir(exist_ok=True)
    masks=baseline/'semantics'
    audit=ray_audit(meshes,frames,audit_dir,max_views=len(frames),semantic_directory=masks if masks.exists() else None)
    metrics={}
    for name in meshes:
        rows=[r['meshes'][name] for r in audit['sampled_views']]
        measured=[r for r in rows if r['measured_pixels']>=100 and r['measured_depth_under_5pct'] is not None]
        metrics[name]={'static_ray_coverage':audit['mean_ray_hit'][name],
            'mean_depth_agreement_within_5pct':float(np.mean([r['measured_depth_under_5pct'] for r in measured])) if measured else None,
            'checked_views':len(measured),'overlapping_measured_pixels':sum(r['measured_pixels'] for r in rows)}
    report={'status':'complete','metrics':metrics,'alignment':{'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist(),
        'training_camera_centers':len(train),'held_out_camera_centers':len(hold),'held_out_median_position_disagreement_visual_units':float(np.median(error)),
        'note':'Reference is the old reconstruction, not physical ground truth. One global similarity cannot correct local trajectory drift.'},
        'baseline_seconds':old.get('elapsed_seconds'),'candidate_seconds':new.get('elapsed_seconds'),
        'candidate_capture':new.get('capture_coverage'),'candidate_geometry_status':new.get('geometry_status'),
        'evaluation':'Identical original RGB, camera rays, baseline stereo and semantic masks after one global alignment',
        'geometry_accuracy_validated':False,'audit':audit}
    save_json(output/'comparison.json',report)
    rows=''.join(f'<tr><td>{name}</td><td>{r["static_ray_coverage"]:.2%}</td><td>{r["mean_depth_agreement_within_5pct"]:.2%}</td><td>{r["checked_views"]}</td></tr>' if r['mean_depth_agreement_within_5pct'] is not None else f'<tr><td>{name}</td><td>{r["static_ray_coverage"]:.2%}</td><td>unknown</td><td>{r["checked_views"]}</td></tr>' for name,r in metrics.items())
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Incremental reconstruction comparison</title><style>body{font:16px system-ui;background:#14202a;color:#eee;margin:32px}a{color:#8df}td,th{padding:12px}</style><h1>Same-view reconstruction comparison</h1><p>Reference geometry is an earlier estimate, not ground truth. Candidate mesh aligned once using alternating camera samples; other cameras check trajectory disagreement.</p><table><tr><th>Result</th><th>Static coverage</th><th>Stereo agreement within 5% depth</th><th>Checked views</th></tr>'+rows+'</table><p><a href="common_audit/coverage.html">Matching camera renders</a> · <a href="comparison.json">Full evidence and alignment</a></p>')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();report=compare(args.baseline,args.candidate,args.output)
    print(json.dumps({k:report[k] for k in ('metrics','alignment','baseline_seconds','candidate_seconds')},indent=2))


if __name__=='__main__':main()
