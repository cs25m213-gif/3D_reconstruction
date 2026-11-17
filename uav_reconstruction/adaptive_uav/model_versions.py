"""Pin cached model snapshots and hash checkpoint/config content for cache identity."""
import hashlib,json
from pathlib import Path


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def snapshot_identity(repository,revision=None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:path=Path(snapshot_download(repository,revision=revision,local_files_only=True))
    except LocalEntryNotFoundError:
        path=Path(snapshot_download(repository,revision=revision,allow_patterns=['*.json','*.safetensors','*.bin']))
    files={str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in ('.json','.safetensors','.bin')}
    if not any(name.endswith(('.safetensors','.bin')) for name in files):raise ValueError('No cached model checkpoint: '+repository)
    return {'repository':repository,'revision':path.name,'files_sha256':files},str(path)


def model_identities(root,semantic=True,orientation=False):
    root=Path(root);depth,path=snapshot_identity('depth-anything/DA3-BASE')
    result={'depth':depth};paths={'depth':path}
    if semantic or orientation:
        from .semantic_scene import SKY_MODEL,SKY_REVISION
        result['sky'],paths['sky']=snapshot_identity(SKY_MODEL,SKY_REVISION)
    if semantic:
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        checkpoint=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
        if not checkpoint.exists():
            state=weights.get_state_dict(progress=True,check_hash=True);del state
        result['instances']={'weights':'MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1','checkpoint_sha256':sha256(checkpoint)}
    checkpoint=root/'tools/DPVO/dpvo.pth'
    if checkpoint.exists():result['pose']={'checkpoint_sha256':sha256(checkpoint),'repository_commit':'0ac95b656d1fda91c271d2a106460d19ad966fc7'}
    return result,paths
