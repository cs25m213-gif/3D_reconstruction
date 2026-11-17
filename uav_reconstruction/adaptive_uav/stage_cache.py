"""Content/config-aware stage markers. Old empty .done files are stale by design."""
import argparse
import json
from pathlib import Path

from .utility_contracts import sha256_file, state_identity, write_json

DEPENDENCIES={'freeze':[], 'clips':['freeze'], 'newclips':['clips'], 'scales':['newclips'],
              'geometry':['freeze'], 'calibrate':['geometry','scales'], 'dedome':['calibrate'],
              'targets':['dedome'], 'records':['targets'], 'controls':['freeze'], 'student':['records','controls'],
              'temporal':['student'], 'embed':['newclips'], 'memory':['records','embed'],
              'labels':['memory','student'], 'sources':['labels'], 'dataset':['labels','sources'],
              'selector':['dataset'], 'finetune':['selector'], 'evaluate':['finetune']}
ARTIFACTS={'freeze':['runs/vnext_baseline_v2/frozen_manifest.json'],
           'student':['runs/vnext_student_update_v1/best.pt','runs/vnext_student_update_v1/report.json'],
           'temporal':['runs/vnext_temporal_student_v1/best.pt','runs/vnext_temporal_student_v1/report.json'],
           'embed':['runs/vnext_memory_v1/embeddings/manifest.json'],
           'labels':['runs/vnext_utility_seq_v1/generation_report.json'],
           'sources':['runs/vnext_utility_seq_v1/source_labels_report.json'],
           'dataset':['runs/vnext_utility_seq_v1/quality_audit.json','runs/vnext_utility_seq_v1/ranking_groups.jsonl'],
           'selector':['runs/vnext_selector_seq_v1/best.pt','runs/vnext_selector_seq_v1/validation.json'],
           'finetune':['runs/vnext_selector_seq_finetuned_v1/best.pt','runs/vnext_selector_seq_finetuned_v1/report.json'],
           'evaluate':['runs/vnext_video_evaluation_seq_v1/promotion.json']}


def signature(stage, config, logs, options):
    config=Path(config)
    contract=json.loads(config.read_text())
    sources={str(p):sha256_file(p) for p in sorted(Path('uav_reconstruction/adaptive_uav').glob('*.py'))}
    for name in ('run_complete_training.sh','reconstruct_final.sh','reconstruct_best.sh'):
        if Path(name).exists():sources[name]=sha256_file(name)
    inputs={}
    fields=['depth_checkpoint']
    if stage in ('records','student','memory','labels','sources','dataset','selector','finetune','evaluate'):
        fields+=['dataset_manifest','geometry_gate']
    for name in fields:
        p=Path(contract.get(name,''))
        if p.is_file():inputs[str(p)]=sha256_file(p)
    cache=Path(contract['data']['cache_root'])
    if stage in ('clips','newclips','scales','dedome','targets','records','embed') and cache.is_dir():
        for p in sorted(cache.glob('*/selected_infos.json')):
            inputs[str(p)]=sha256_file(p)
    # Every changed predecessor invalidates downstream work, even if a stage name is unchanged.
    for prior in DEPENDENCIES.get(stage,[]):
        p=Path(logs)/(prior+'.done')
        if p.exists():inputs[str(p)]=sha256_file(p)
    relevant={'diagnostic':options.get('diagnostic')}
    if stage=='temporal':relevant['temporal']=options.get('temporal')
    if stage=='student':relevant['own_sessions']=options.get('own_sessions')
    if stage=='embed':relevant['embedding_device']=options.get('embedding_device')
    if stage=='newclips':
        relevant['reconstruction_jobs']=options.get('reconstruction_jobs','1')
        relevant['threads']=options.get('threads')
    return state_identity(stage=stage,config=sha256_file(config),sources=sources,inputs=inputs,options=relevant)


def matches(marker, expected):
    try:
        saved=json.loads(Path(marker).read_text())
        return (saved.get('signature')==expected and all(Path(p).is_file() and sha256_file(p)==digest
                                                       for p,digest in saved.get('artifacts',{}).items()))
    except (OSError,ValueError):
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','write'))
    p.add_argument('--stage',required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True)
    p.add_argument('--options',required=True,help='JSON containing resolved options and stage order')
    a=p.parse_args();options=json.loads(a.options)
    key=signature(a.stage,a.config,a.logs,options)
    marker=a.logs/(a.stage+'.done')
    if a.action=='check':raise SystemExit(0 if matches(marker,key) else 1)
    artifacts={p:sha256_file(Path(p)) for p in ARTIFACTS.get(a.stage,[])}
    write_json(marker,{'stage':a.stage,'signature':key,'options':options,'schema_version':1,'artifacts':artifacts})


if __name__=='__main__':
    main()
