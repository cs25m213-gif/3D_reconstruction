"""Additional posed LiDAR training groups from released UAVScenes flights.

Targets are measured returns. Camera poses are released metric reference poses.
LiDAR points used for scale are disjoint from points used for the depth loss.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .build_cached_lidar_targets import downsample_sparse_min


class DirectLidarDataset(Dataset):
    def __init__(self, root: Path, target_root: Path, scenes=('AMtown01', 'HKairport01'),
                 size=(192, 256), augment=True, max_normalized_baseline=.4,
                 use_neighbors=True, sparse_anchor_root=None):
        self.root, self.target_root, self.size, self.augment = root, target_root, size, augment
        self.sparse_anchor_root = Path(sparse_anchor_root) if sparse_anchor_root else None
        self.groups = []
        for scene in scenes:
            rows = json.loads((root / scene / 'manifest.json').read_text())
            neighbor_file = root / scene / 'neighbor_manifest.json'
            neighbors = json.loads(neighbor_file.read_text()) if use_neighbors and neighbor_file.exists() else {}
            available = {int(p.stem.split('_')[-1]) for p in (target_root / scene).glob('frame_*.npz')}
            poses = [np.asarray(r['info']['T4x4'], dtype=np.float64) for r in rows]
            centers = np.asarray([p[:3, 3] for p in poses])
            for index in sorted(available):
                with np.load(target_root / scene / f'frame_{index:06d}.npz') as target:
                    depth = target['measured_depth_z_m']
                    valid = depth[depth > 0]
                if len(valid) < 1000:
                    continue
                median_depth = float(np.median(valid))
                candidates = neighbors.get(str(index), [])
                if not candidates:
                    candidates = [{'info': rows[other]['info'],
                                   'rgb': f'rgb/frame_{other:06d}.jpg',
                                   'offset': other-index}
                                  for other in range(max(0, index-3), min(len(rows), index+4))
                                  if other != index]
                choices = []
                for candidate in candidates:
                    other_pose = np.asarray(candidate['info']['T4x4'], dtype=np.float64)
                    baseline = np.linalg.norm(other_pose[:3, 3] - centers[index])
                    cosine = np.clip((np.trace(poses[index][:3, :3].T @ other_pose[:3, :3]) - 1) / 2, -1, 1)
                    angle = np.degrees(np.arccos(cosine))
                    if .04 <= baseline / median_depth <= max_normalized_baseline and angle <= 25:
                        choices.append((abs(candidate['offset']), candidate))
                choices.sort(key=lambda item: item[0])
                if len(choices) >= 2:
                    self.groups.append((scene, index, [choices[0][1], choices[1][1]], median_depth))
        if not self.groups:
            raise ValueError('No geometrically valid direct LiDAR groups')
        if self.sparse_anchor_root:
            self.groups = [group for group in self.groups
                           if (self.sparse_anchor_root / group[0] / f'frame_{group[1]:06d}.npz').exists()]
            if not self.groups:
                raise ValueError('No direct groups have verified RGB sparse anchors')
        self.manifests = {scene: json.loads((root / scene / 'manifest.json').read_text()) for scene in scenes}

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        scene, reference, sources, _ = self.groups[index]
        rows = self.manifests[scene]
        target_path = self.target_root / scene / f'frame_{reference:06d}.npz'
        with np.load(target_path) as data:
            measured_native = data['measured_depth_z_m'].copy()
        # Split by deterministic checkerboard parity in the native image. The
        # scale anchor and supervised pixels never occupy the same native cell.
        yy, xx = np.nonzero(measured_native > 0)
        anchor = ((yy + xx) & 1) == 0
        anchor_depth = measured_native[yy[anchor], xx[anchor]]
        if len(anchor_depth) < 100:
            raise ValueError(f'Insufficient scale anchors: {target_path}')
        scale = float(np.median(anchor_depth))
        sfm_anchor = np.zeros(self.size, np.float32)
        if self.sparse_anchor_root:
            with np.load(self.sparse_anchor_root / scene / f'frame_{reference:06d}.npz') as data:
                sfm_anchor_m = data['anchor_depth_m'].astype(np.float32)
                scale = float(data['anchor_scale_m'])
            sfm_anchor = sfm_anchor_m / scale
        measured_native[yy[anchor], xx[anchor]] = 0
        h, w = self.size
        measured = downsample_sparse_min(measured_native, (h, w)) / scale
        images, intrinsics, poses = [], [], []
        reference_source = {'info': rows[reference]['info'], 'rgb': f'rgb/frame_{reference:06d}.jpg'}
        for source in (reference_source, *sources):
            info = source['info']
            path = self.root / scene / source['rgb']
            rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            images.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255)
            k = np.asarray(info['P3x3'], dtype=np.float32).copy()
            k[0] *= w / info['Width']
            k[1] *= h / info['Height']
            intrinsics.append(k)
            camera_to_world = np.asarray(info['T4x4'], dtype=np.float32)
            world_to_camera = np.linalg.inv(camera_to_world)
            world_to_camera[:3, 3] /= scale
            poses.append(world_to_camera)
        image_tensor = torch.from_numpy(np.stack(images))
        if self.augment:
            gain = .75 + .5 * torch.rand(())
            offset = -.08 + .16 * torch.rand(())
            gamma = .8 + .4 * torch.rand(())
            image_tensor = (image_tensor * gain + offset).clamp(0, 1).pow(gamma)
        measured_valid = (measured > .25) & (measured < 4.)
        zeros = np.zeros((1, h, w), dtype=np.float32)
        return {'rgb': image_tensor, 'K': torch.from_numpy(np.stack(intrinsics)),
                'P': torch.from_numpy(np.stack(poses)),
                'teacher': torch.from_numpy(zeros.copy()),
                'teacher_valid': torch.from_numpy(np.zeros((1, h, w), dtype=bool)),
                'anchor': torch.from_numpy(sfm_anchor[None]),
                'measured': torch.from_numpy(measured[None].astype(np.float32)),
                'measured_valid': torch.from_numpy(measured_valid[None]),
                'metric_per_normalized': scale, 'scale': scale,
                'path': str(target_path)}
