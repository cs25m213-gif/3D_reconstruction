"""CPU camera repair + filtered multi-view reconstruction in a separate folder.

Reuse existing registered tracks; do not treat numerical completion as proof of
accuracy. Supports one shared SIMPLE_RADIAL camera. No new packages required.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from .quality_audit import audit, quality_gate
from .session_reconstruct import SessionReconstructor, read_model, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--source',type=Path,help='Existing reconstruction directory with best_model.json and images/')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent/'configs/cpu_quality.json')
    parser.add_argument('--colmap',type=Path,default=Path('/home/vishnu/miniconda3/envs/drone3d/bin/colmap'))
    parser.add_argument('--openmvs',type=Path,default=Path('/home/vishnu/eswar/apps/openmvs/bin/OpenMVS'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--sparse-only',action='store_true')
    args=parser.parse_args()
    session=args.session.resolve()
    source=(args.source or session/'reconstruction_online_cpu').resolve()
    output=(args.output or session/'reconstruction_refined_cpu').resolve()
    if source==output:
        raise ValueError('Use a separate output directory to preserve the original reconstruction')
    config=json.loads(args.config.read_text())
    config['threads']=min(config['threads'],len(os.sched_getaffinity(0)))
    config['source_reconstruction']=str(source)
    runner=SessionReconstructor(session,output,config,args.colmap,args.openmvs,args.resume)
    started=time.monotonic()
    save_json(output/'quality_run.json',{'status':'running','started_unix_seconds':time.time(),
        'dense_export_requested':not args.sparse_only,'surface_quality_validated':False})
    try:
        def repair():
            best=json.loads((source/'best_model.json').read_text())
            text=Path(best['text'])
            if not text.is_absolute():
                text=source/text
            # Each retry is isolated; never overwrite an earlier failed candidate.
            index=0
            while (output/f'robust_candidate_{index:03d}').exists():
                index+=1
            refined=output/f'robust_candidate_{index:03d}'
            runner.run_command('robust_bundle',[os.sys.executable,'-m','adaptive_uav.robust_bundle',
                text,refined,'--max-points',config['bundle_max_points'],
                '--max-nfev',config['bundle_max_nfev'],'--timeout',config['bundle_timeout_seconds']],
                cwd=Path(__file__).resolve().parent.parent,timeout=config['bundle_timeout_seconds']+30)
            report=json.loads((refined/'optimization_report.json').read_text())
            if not report['accepted']:
                raise RuntimeError('Robust optimization candidate was not accepted')
            conditioning=audit(refined)
            valid,reason=quality_gate(conditioning,config)
            conditioning.update(quality_gate_passed=valid,quality_gate_reason=reason,
                                optimization_report=report)
            save_json(output/'quality_audit.json',conditioning)
            if not valid:
                raise RuntimeError(reason)
            _,poses,points=read_model(refined)
            images=output/'images'; images.mkdir(exist_ok=True)
            for pose in poses:
                name=pose['image']
                target=(images/name).resolve()
                if not target.is_relative_to(images.resolve()):
                    raise ValueError('Unsafe image path in source model')
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/'images'/name,target)
            binary=output/'refined_sparse'; binary.mkdir(exist_ok=True)
            runner.run_command('refined_to_binary',[args.colmap,'model_converter',
                '--input_path',refined,'--output_path',binary,'--output_type','BIN'])
            save_json(output/'best_model.json',{'binary':str(binary),'text':str(refined)})
            save_json(output/'keyframes.json',[p['image'] for p in poses])
            save_json(output/'camera_poses.json',{'convention':'COLMAP world-to-camera; arbitrary scale',
                'cameras':conditioning['cameras'],'poses':poses})
            save_json(output/'reconstruction_metrics.json',{'mode':'robust_cpu_camera_repair_then_dense',
                'registered_frames':len(poses),'sparse_points':len(points),'scale':'arbitrary',
                'ground_truth_available':False,'surface_quality_validated':False,'optimization':report})
        runner.stage('robust_camera_repair',repair)
        best=json.loads((output/'best_model.json').read_text())
        cached_report=json.loads((Path(best['text'])/'optimization_report.json').read_text())
        if 'retriangulated_or_revalidated_points' not in cached_report:
            raise RuntimeError('Checkpoint predates full-track preservation. Use a fresh --output directory; '
                               'the old candidate and original reconstruction are preserved.')
        for stage in ('decode','selection','retrieval','features','matching','mapping'):
            runner.state['stages'][stage]={'status':'complete','seconds':0.,'provided_by':'validated_robust_camera_repair'}
        save_json(runner.state_path,runner.state)
        if not args.sparse_only:
            runner.reconstruct()
            def check_surface():
                from .surface_audit import audit_surface
                report=audit_surface(output,threads=config['threads'],
                    min_view_coverage=config.get('min_view_surface_coverage',.35),
                    min_supported_fraction=config.get('min_supported_view_fraction',.6))
                if not report['coverage_gate_passed']:
                    raise RuntimeError('Dense surface coverage failed. Exported GLB is a candidate only; '
                                       'see surface_audit.json. Original reconstruction preserved.')
            runner.stage('dense_surface_check',check_surface)
        save_json(output/'quality_run.json',{'status':'complete','seconds':time.monotonic()-started,
            'dense_export_requested':not args.sparse_only,'surface_quality_validated':False})
    except Exception as error:
        save_json(output/'quality_run.json',{'status':'failed','seconds':time.monotonic()-started,'error':str(error)})
        raise


if __name__=='__main__':
    main()
