"""Pretrained static-scene masks and time-labelled object observations.

Masks live in original undistorted camera pixels. They never rotate camera
poses, assume metric scale, infer identity, or turn moving detections into
static triangulated landmarks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

SKY_MODEL = 'nvidia/segformer-b0-finetuned-ade-512-512'
SKY_REVISION = 'cc2189cbf3173593452db75308756036c990bc44'
ANIMALS = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rotate_image(image, clockwise):
    if clockwise not in (0, 90, 180, 270):
        raise ValueError('Rotation must be 0, 90, 180 or 270')
    return np.ascontiguousarray(np.rot90(image, -(clockwise // 90)))


def remap_labels(labels, source_k, target_k, shape):
    """Nearest-neighbour pinhole mapping, including depth-network crop offsets."""
    y, x = np.indices(shape, dtype=np.float32)
    x = (x - target_k[0, 2]) / target_k[0, 0] * source_k[0, 0] + source_k[0, 2]
    y = (y - target_k[1, 2]) / target_k[1, 1] * source_k[1, 1] + source_k[1, 2]
    return cv2.remap(labels, x.astype(np.float32), y.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)


class Predictor:
    def __init__(self, device='cuda'):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Semantic CUDA inference requested but CUDA is unavailable')
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(SKY_MODEL, revision=SKY_REVISION)
        self.sky = SegformerForSemanticSegmentation.from_pretrained(SKY_MODEL, revision=SKY_REVISION).eval().to(device)
        self.sky_id = next(int(i) for i, name in self.sky.config.id2label.items() if name == 'sky')
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.categories = weights.meta['categories']
        self.detector = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=640, max_size=960,
                                               box_score_thresh=.5, box_detections_per_img=100).eval().to(device)

    def sky_prediction(self, rgb):
        import torch
        with torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
            logits = self.sky(**inputs).logits.float()
            probabilities = logits.softmax(dim=1)
            confidence = float(probabilities.max(dim=1).values.mean())
            sky = torch.nn.functional.interpolate(probabilities[:, self.sky_id:self.sky_id+1],
                size=rgb.shape[:2], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        return sky, confidence

    def instances(self, rgb, threshold):
        import torch
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(self.device).float() / 255.
        with torch.inference_mode():
            result = self.detector([tensor])[0]
        selected = []
        for i, score in enumerate(result['scores'].cpu().tolist()):
            category = self.categories[int(result['labels'][i])]
            if score >= threshold and (category == 'person' or category in ANIMALS):
                selected.append((category, score, result['masks'][i, 0].cpu().numpy() >= .5))
        return selected


def predict_workspace(workspace, output, *, rotation='auto', device='cuda', sky_threshold=.65, object_threshold=.7):
    import pycolmap
    import torch
    torch.set_num_threads(4)
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if not 0 < sky_threshold < 1 or not 0 < object_threshold < 1:
        raise ValueError('Confidence thresholds must be between 0 and 1')
    model = pycolmap.Reconstruction(workspace / 'sparse')
    images = sorted((im for im in model.images.values() if im.has_pose), key=lambda im: im.name)
    if not images:
        raise ValueError('Semantic filtering needs registered cameras')
    output.mkdir(parents=True, exist_ok=True)
    policy = {'sky_model': SKY_MODEL, 'sky_revision': SKY_REVISION,
        'instance_model': 'torchvision/maskrcnn_resnet50_fpn_v2/COCO_V1',
        'sky_threshold': sky_threshold, 'object_threshold': object_threshold,
        'rotation': str(rotation), 'device': device, 'animal_categories': sorted(ANIMALS),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'images': {im.name: hashlib.sha256((workspace / 'images' / im.name).read_bytes()).hexdigest() for im in images},
        'cameras': {}}
    # Camera dictionaries contain enum values in some bindings; keep numeric K instead.
    policy['cameras'] = {str(c.camera_id): {'width': c.width, 'height': c.height, 'K': c.calibration_matrix().tolist()} for c in model.cameras.values()}
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != policy:
        raise ValueError('Semantic cache inputs/model/settings changed; use a fresh output')
    save_json(manifest, policy)
    report_path = output / 'semantic_report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get('status') == 'complete' and all((output / (im.name + '.npz')).is_file() for im in images):
            return previous
    started = time.monotonic()
    predictor = Predictor(device)
    scores = {}
    if str(rotation) == 'auto':
        for angle in (0, 90, 180, 270):
            values = []
            for i in np.unique(np.linspace(0, len(images)-1, min(4, len(images)), dtype=int)):
                rgb = cv2.cvtColor(cv2.imread(str(workspace / 'images' / images[i].name)), cv2.COLOR_BGR2RGB)
                _, score = predictor.sky_prediction(rotate_image(rgb, angle))
                values.append(score)
            scores[str(angle)] = float(np.mean(values))
        angle = int(max(scores, key=scores.get))
    else:
        angle = int(rotation)
        rotate_image(np.zeros((2, 3)), angle)
    report = {'status': 'running', 'rotation_clockwise': angle, 'orientation_confidence_scores': scores,
        'orientation_note': 'Model-confidence heuristic; masks are mapped back to camera pixels. Camera poses are unchanged.',
        'policy': policy, 'frames': [], 'elapsed_seconds': 0., 'geometry_accuracy_validated': False}
    print(f'[semantics] {len(images)} frames; orientation {angle} degrees; device {device}', flush=True)
    for index, im in enumerate(images):
        camera = model.cameras[im.camera_id]
        rgb = cv2.cvtColor(cv2.imread(str(workspace / 'images' / im.name)), cv2.COLOR_BGR2RGB)
        oriented = rotate_image(rgb, angle)
        sky, _ = predictor.sky_prediction(oriented)
        sky = rotate_image(sky, (-angle) % 360)
        classes = np.where(sky >= sky_threshold, 1, 0).astype(np.uint8)
        instance_ids = np.zeros(classes.shape, np.uint16)
        observations = []
        for label, confidence, mask in predictor.instances(oriented, object_threshold):
            mask = rotate_image(mask.astype(np.uint8), (-angle) % 360).astype(bool)
            mask &= instance_ids == 0
            if mask.sum() < 12:
                continue
            instance_id = len(observations) + 1
            instance_ids[mask] = instance_id
            classes[mask] = 2 if label == 'person' else 3
            yy, xx = np.where(mask)
            observations.append({'instance_id': instance_id, 'class': label, 'confidence': float(confidence),
                'bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)], 'mask_pixels': int(mask.sum())})
        # Small relative padding suppresses moving object boundaries while sky
        # stays conservative around rooflines and vegetation.
        dynamic = (classes >= 2).astype(np.uint8)
        radius = max(1, round(max(classes.shape) * .0015))
        padded = cv2.dilate(dynamic, np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
        classes[padded & (classes == 0)] = 4
        path = output / (im.name + '.npz')
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, classes=classes, instances=instance_ids,
                            K=camera.calibration_matrix(), observations=json.dumps(observations))
        temp.replace(path)
        row = {'frame': im.name, 'sky_fraction': float((classes == 1).mean()),
               'dynamic_fraction': float((classes >= 2).mean()), 'observations': observations}
        report['frames'].append(row)
        report['elapsed_seconds'] = time.monotonic()-started
        save_json(report_path, report)
        if index % 20 == 0 or index == len(images)-1:
            overlay = rgb.copy(); overlay[classes == 1] = (.4*overlay[classes == 1]+.6*np.array([80, 130, 255])).astype(np.uint8)
            overlay[classes >= 2] = (.3*overlay[classes >= 2]+.7*np.array([255, 100, 20])).astype(np.uint8)
            cv2.imwrite(str(output / (im.name + '.overlay.jpg')), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f'[semantics] {index+1}/{len(images)}; sky {row["sky_fraction"]:.1%}; objects {len(observations)}', flush=True)
    report.update(status='complete', elapsed_seconds=time.monotonic()-started,
                  mean_sky_fraction=float(np.mean([r['sky_fraction'] for r in report['frames']])),
                  mean_dynamic_fraction=float(np.mean([r['dynamic_fraction'] for r in report['frames']])),
                  object_observations=sum(len(r['observations']) for r in report['frames']))
    save_json(report_path, report)
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def load_frame(directory, name, intrinsic, shape):
    with np.load(Path(directory) / (name + '.npz')) as data:
        return (remap_labels(data['classes'], data['K'], intrinsic, shape),
                remap_labels(data['instances'], data['K'], intrinsic, shape),
                json.loads(str(data['observations'])))


def object_position(frame, mask):
    """Estimate visible object surface, never use background outside its mask."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if core.sum() < 8:
        core = mask
    source = 'stereo_estimate'
    depth = frame.get('object_measured', frame['measured'])
    valid = core & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 8:
        median_error = frame['validation'].get('median')
        if median_error is None or not np.isfinite(median_error) or median_error > .2:
            return {'position': None, 'position_status': 'no_validated_depth_in_object_mask'}
        depth = frame['prior']
        valid = core & np.isfinite(depth) & (depth > 0)
        source = 'learned_depth_estimate'
        # A handful of interpolated depth pixels cannot resolve a small object
        # independently of its background, even when the overall frame validates.
        if valid.sum() < 64:
            return {'position': None, 'position_status': 'object_too_small_for_learned_depth',
                    'depth_pixels': int(valid.sum()), 'minimum_learned_depth_pixels': 64}
    if valid.sum() < 8:
        return {'position': None, 'position_status': 'insufficient_object_depth_pixels'}
    z = depth[valid]
    median = float(np.median(z)); spread = float(np.median(np.abs(z-median)))
    if spread / median > .25:
        return {'position': None, 'position_status': 'ambiguous_object_depth'}
    yy, xx = np.where(valid)
    inliers = np.abs(z-median) <= max(3*spread, .05*median)
    k, p = frame['K'], frame['P']
    pixel = [float(np.median(xx[inliers])), float(np.median(yy[inliers]))]
    camera = np.array([(pixel[0]-k[0, 2])/k[0, 0]*median, (pixel[1]-k[1, 2])/k[1, 1]*median, median])
    world = (camera-p[:3, 3]) @ p[:3, :3]
    return {'position': world.tolist(), 'position_status': source, 'depth_median': median,
            'depth_mad': spread, 'depth_pixels': int(valid.sum()),
            'position_kind': 'visible_surface_centroid', 'metric_scale_validated': False}


def filter_frame(frame, directory, fps=None):
    classes, ids, detections = load_frame(directory, frame['name'], frame['K'], frame['prior'].shape)
    observations = []
    for detection in detections:
        row = dict(detection, frame=frame['name'])
        suffix = Path(frame['name']).stem.split('_')[-1]
        index = int(suffix) if suffix.isdigit() else None
        row.update(frame_index=index, timestamp_seconds=index/fps if fps and index is not None else None)
        row.update(object_position(frame, ids == detection['instance_id']))
        observations.append(row)
    excluded = classes > 0
    frame['semantic_excluded'] = excluded
    frame['semantic_sky'] = classes == 1
    for name in ('prior', 'measured', 'anchors'):
        frame[name] = frame[name].copy()
        frame[name][excluded] = 0
    return observations


def write_observations(output, observations):
    positioned = [row for row in observations if row.get('position') is not None]
    save_json(Path(output) / 'object_observations.json', {'observations': observations,
        'positioned_observations': len(positioned), 'unlocalized_observations': len(observations)-len(positioned),
        'units': 'COLMAP visual units', 'identity_tracking': False,
        'position_policy': {'minimum_stereo_pixels': 8, 'minimum_learned_pixels': 64,
                            'maximum_frame_median_depth_error': .2, 'maximum_object_relative_mad': .25},
        'note': 'Time-labelled detections, not persistent identities or stationary landmarks. Positions are estimates of visible object surfaces.'})
    lines = ['ply', 'format ascii 1.0', f'element vertex {len(positioned)}', 'property float x',
             'property float y', 'property float z', 'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
    for row in positioned:
        color = (255, 140, 30) if row['class'] == 'person' else (90, 240, 120)
        lines.append(' '.join(map(str, [*row['position'], *color])))
    (Path(output) / 'object_markers.ply').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rotation', choices=('auto', '0', '90', '180', '270'), default='auto')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    predict_workspace(args.workspace, args.output, rotation=args.rotation, device=args.device)


if __name__ == '__main__':
    main()
