"""Add wider image matches and rebuild one saved session's camera map."""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pycolmap


def refine(online_dir: Path, output: Path, threads: int = 8, window: int = 24,
           retrieval: int = 8, timeout: int = 900):
    online_dir = online_dir.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_db = online_dir / 'database.db'
    db = output / 'database.db'
    if not db.exists():
        with sqlite3.connect(source_db) as source, sqlite3.connect(db) as target:
            source.backup(target)
    names = json.loads((online_dir / 'keyframes.json').read_text())
    descriptors = np.load(online_dir / 'descriptors.npy')
    if len(names) != len(descriptors):
        raise ValueError('Keyframe and descriptor counts differ')
    pairs = set()
    for i, name in enumerate(names):
        for j in range(i + 1, min(len(names), i + window + 1)):
            pairs.add((name, names[j]))
        similarities = descriptors @ descriptors[i]
        for j in np.argsort(-similarities)[:retrieval + 1]:
            if i != j:
                pairs.add(tuple(sorted((name, names[int(j)]))))
    pair_file = output / 'pairs.txt'
    pair_file.write_text(''.join(f'{a} {b}\n' for a, b in sorted(pairs)))
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    matching.use_gpu = False
    matching.guided_matching = True
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = pair_file
    began = time.monotonic()
    print(f'Matching {len(pairs)} candidate pairs', flush=True)
    pycolmap.match_image_pairs(db, matching_options=matching, pairing_options=pairing,
                              device=pycolmap.Device.cpu)
    matched_seconds = time.monotonic() - began
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = threads
    options.mapper.num_threads = threads
    options.multiple_models = True
    options.init_num_trials = 100
    options.min_model_size = 5
    options.mapper.init_min_tri_angle = 2.0
    options.ba_refine_focal_length = False
    options.ba_refine_extra_params = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.ba_use_gpu = False
    options.max_runtime_seconds = timeout
    print('Building global camera map', flush=True)
    models = pycolmap.incremental_mapping(db, online_dir / 'images', output / 'models',
                                          options=options)
    if not models:
        raise RuntimeError('No camera map was produced')
    best_id, best = max(models.items(), key=lambda item:
                        (item[1].num_reg_images(), item[1].num_points3D()))
    model_path = output / 'best_model'
    model_path.mkdir(exist_ok=True)
    best.write(model_path)
    best.export_PLY(output / 'sparse_map.ply')
    report = {'selected_images': len(names), 'registered_images': best.num_reg_images(),
              'sparse_points': best.num_points3D(), 'model_id': best_id,
              'matched_seconds': matched_seconds, 'total_seconds': time.monotonic() - began,
              'candidate_pairs': len(pairs), 'source_online_dir': str(online_dir)}
    (output / 'refinement_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('online_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--window', type=int, default=24)
    parser.add_argument('--retrieval', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    refine(args.online_dir, args.output, args.threads, args.window,
           args.retrieval, args.timeout)


if __name__ == '__main__':
    main()
