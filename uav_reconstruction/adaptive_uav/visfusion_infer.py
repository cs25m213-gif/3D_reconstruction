"""Run the released VisFusion checkpoint on adapted COLMAP cameras.

Normalization without a measured scale is explicitly experimental. This module
runs in the isolated TorchSparse environment; it does not perform dense stereo.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def normalization(poses, anchor_depth, metres_per_unit=None):
    """Return visual-world to network-world similarity and its provenance."""
    from scipy.spatial.transform import Rotation
    if not np.isfinite(anchor_depth) or anchor_depth <= 0:
        raise ValueError('Positive finite anchor depth is required')
    scale = float(metres_per_unit) if metres_per_unit is not None else 2./anchor_depth
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive and finite')
    up = poses[0, :3, :3] @ np.array([0., -1., 0.])
    rotation = Rotation.align_vectors(np.array([[0., 0., 1.]]), up[None])[0].as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = scale*rotation
    transform[:3, 3] = np.array([0., 0., 1.5])-scale*rotation@poses[0, :3, 3]
    return transform, {'scale': scale, 'scale_source': 'user_supplied' if metres_per_unit is not None else
                       'assumed 2 nominal units median landmark depth',
                       'metric_scale_validated': False, 'gravity_alignment': 'first-camera upright assumption'}


def fragment_indices(count, window=9, stride=6):
    if count < window:
        raise ValueError('At least nine posed images are required')
    starts = list(range(0, count-window+1, stride))
    if starts[-1] != count-window:
        starts.append(count-window)
    return [list(range(i, i+window)) for i in starts]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Adapter manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repository', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repository.resolve()))
    import torch
    from torch.utils.data._utils.collate import default_collate
    from PIL import Image
    from skimage.measure import marching_cubes
    import trimesh
    from config import cfg
    from datasets import transforms
    from models import VisFusion

    data = json.loads(args.input.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'starting', 'pretrained_visfusion_used': True, 'normalization': data['normalization'],
              'units': 'COLMAP visual units', 'geometry_accuracy_validated': False, 'fragments': []}
    started = time.monotonic()

    def save(status):
        report.update(status=status, elapsed_seconds=time.monotonic()-started)
        temporary = out/'run_report.tmp.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(out/'run_report.json')
        print(status, flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the released VisFusion network')
        torch.set_num_threads(8)
        cfg.merge_from_file(str(args.repository/'config/test_scene.yaml'))
        cfg.MODEL.SINGLE_LAYER_MESH = False
        cfg.REDUCE_GPU_MEM = False
        cfg.freeze()
        save('loading_checkpoint')
        model = VisFusion(cfg).cuda().eval()
        checkpoint = args.repository/'checkpoints/pretrained/model_000049.ckpt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        weights = {k.removeprefix('module.'):v for k,v in state['model'].items()}
        model.load_state_dict(weights, strict=True)
        report['checkpoint_tensors_loaded'] = len(weights)
        converter = transforms.Compose([
            transforms.ResizeImage((640, 480)), transforms.ToTensor(),
            transforms.RandomTransformSpace(cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE,
                random_rotation=False, random_translation=False, paddingXY=0, paddingZ=0,
                max_epoch=cfg.TRAIN.EPOCHS), transforms.IntrinsicsPoseToProjection(9, 4)])
        views = data['views']
        groups = fragment_indices(len(views))
        save('gpu_visfusion_running')
        with torch.inference_mode():
            for i, indices in enumerate(groups):
                tick = time.monotonic()
                sample = {'imgs':[Image.open(views[j]['image']).convert('RGB') for j in indices],
                          'intrinsics':np.array([views[j]['K'] for j in indices], np.float32),
                          'poses':np.array([views[j]['pose'] for j in indices], np.float32),
                          'vol_origin':np.zeros(3,np.float32), 'scene':'phone_capture',
                          'fragment':f'phone_capture_{i}', 'epoch':[0], 'image_ids':indices}
                sample = default_collate([converter(sample)])
                outputs, _ = model(sample, save_mesh=i == len(groups)-1)
                torch.cuda.synchronize()
                report['fragments'].append({'index': i, 'views': indices, 'seconds':time.monotonic()-tick})
                save('gpu_visfusion_running')
        if 'scene_tsdf' not in outputs or not outputs['scene_tsdf']:
            raise RuntimeError('VisFusion produced no global TSDF')
        tsdf = outputs['scene_tsdf'][0].detach().cpu().numpy()
        origin = outputs['origin'][0].detach().cpu().numpy()
        if not np.isfinite(tsdf).all() or not tsdf.min() < 0 < tsdf.max():
            raise RuntimeError('TSDF contains no finite surface crossing')
        save('mesh_export')
        vertices, faces, normals, _ = marching_cubes(tsdf, level=0, allow_degenerate=False)
        vertices = vertices*cfg.MODEL.VOXEL_SIZE+origin
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
        mesh.apply_transform(np.linalg.inv(np.asarray(data['world_to_network'])))
        mesh.export(out/'scene_visfusion.ply')
        mesh.export(out/'scene_visfusion.glb')
        np.savez_compressed(out/'predicted_tsdf.npz', tsdf=tsdf, origin=origin,
                            voxel_size=cfg.MODEL.VOXEL_SIZE, world_to_network=data['world_to_network'])
        report.update(vertices=len(mesh.vertices), faces=len(mesh.faces),
                      gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      mesh=str(out/'scene_visfusion.glb'))
        save('complete_experimental')
    except Exception as error:
        report['error'] = str(error)
        save('failed')
        raise


if __name__ == '__main__':
    main()
