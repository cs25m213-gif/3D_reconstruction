"""Measure capture coverage and continue mapping from an immutable pose cache."""
import json
from pathlib import Path
import shutil
import time

import numpy as np


def capture_coverage(camera, duration, minimum=.9):
    if not np.isfinite(duration) or duration <= 0 or not 0 < minimum <= 1:
        raise ValueError('Positive duration and a coverage threshold in (0, 1] are required')
    span = max(0., float(camera['end_seconds']) - float(camera['start_seconds']))
    temporal = min(1., span / duration)
    selected = int(camera['full_run_selected_frames'])
    registered = int(camera['registered_frames'])
    registration = registered / max(1, selected)
    return {'passed': temporal >= minimum and registration >= .8,
            'temporal_fraction': temporal, 'selected_frame_registration_fraction': registration,
            'minimum_temporal_fraction': minimum, 'minimum_registration_fraction': .8,
            'registered_frames': registered, 'selected_frames': selected,
            'note': 'Coverage of the video and selected cameras; not whole-scene surface completeness.'}


def mapping_timeout(config, guided):
    """Guidance may use an explicit budget, but never adds a hidden 30-second cap."""
    timeout = float(config.get('mapping_timeout_seconds', 120))
    if guided and 'guided_mapping_timeout_seconds' in config:
        guided_timeout = float(config['guided_mapping_timeout_seconds'])
        if not np.isfinite(guided_timeout) or guided_timeout < 0:
            raise ValueError('Guided mapping timeout must be finite and nonnegative')
        if guided_timeout > 0:
            timeout = min(timeout, guided_timeout) if timeout > 0 else guided_timeout
    if not np.isfinite(timeout) or timeout < 0:
        raise ValueError('Mapping timeout must be finite and nonnegative')
    return timeout


def result_status(coverage, quality):
    accepted = {'internally_consistent_experimental', 'high_rendered_coverage_unvalidated'}
    return 'complete' if coverage['passed'] and quality in accepted else 'incomplete'


def link_cached_candidates(source, target):
    """Keep a checked previous model even if its folder name is reserved locally."""
    source, target = Path(source), Path(target)
    for candidate in source.glob('online*'):
        if not candidate.is_dir() or not (candidate / 'current_map.json').is_file():
            continue
        link = target / candidate.name
        if link.exists() and link.resolve() == candidate.resolve():
            continue
        if link.exists():
            link = target / ('online_cached_' + candidate.name)
        suffix = 1
        while link.exists() and link.resolve() != candidate.resolve():
            link = target / ('online_cached_' + candidate.name + '_' + str(suffix))
            suffix += 1
        if not link.exists():
            link.symlink_to(candidate.resolve(), target_is_directory=True)


def continue_mapping(camera, source, output, seconds, threads=8, ba_backend='auto',refine_intrinsics=False):
    """Reuse extracted features/matches and extend the existing connected model.

    Copy the database and pose files into the new run. Original runs remain
    untouched, and continuation undergoes the usual gyro/reprojection gates.
    """
    import pycolmap
    from .gpu_sfm import configure_mapping_ba
    from .combined_reconstruct import save_json

    started = time.monotonic()
    source, output = Path(source), Path(output)
    previous = Path(camera['source']).parents[3]
    database, images = previous / 'database.db', previous / 'images'
    if not database.is_file() or not images.is_dir():
        raise ValueError('Camera continuation requires its saved database and RGB images')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Continuation requires a positive mapping budget')
    folder = output / 'sparse_run' / 'online_coverage'
    if folder.exists():
        raise ValueError('Camera continuation already exists; use a fresh output directory')
    folder.mkdir(parents=True)
    shutil.copy2(database, folder / 'database.db')
    (folder / 'images').symlink_to(images.resolve(), target_is_directory=True)
    # Other checked candidates stay available to camera selection without
    # copying their databases or replacing their accepted model files.
    link_cached_candidates(source, folder.parent)
    save_json(folder.parent / 'pipeline_manifest.json',
              json.loads((source / 'pipeline_manifest.json').read_text()))
    seed = folder / 'input_model'
    seed.mkdir()
    reconstruction = pycolmap.Reconstruction(camera['model'])
    reconstruction.write(seed)
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = options.mapper.num_threads = threads
    options.multiple_models = False
    options.min_model_size = 5
    options.max_runtime_seconds = max(1, int(np.ceil(seconds)))
    options.ba_refine_focal_length = options.ba_refine_extra_params = bool(refine_intrinsics)
    options.ba_refine_sensor_from_rig = False
    options.mapper.abs_pose_refine_focal_length = options.mapper.abs_pose_refine_extra_params = bool(refine_intrinsics)
    options.ba_local_max_num_iterations = 12
    options.ba_global_max_num_iterations = 25
    options.ba_global_max_refinements = 2
    compute = configure_mapping_ba(options, {'sfm_use_gpu': True, 'sfm_ba_backend': ba_backend}, pycolmap,
                                   camera_models=[c.model_name for c in reconstruction.cameras.values()])
    save_json(folder / 'compute_backend.json', {'features': 'cached', 'matching': 'cached',
                                               'bundle_adjustment': compute})
    report = {'input_model': str(seed), 'input_registered_frames': reconstruction.num_reg_images(),
              'intrinsics_refinement_enabled':bool(refine_intrinsics),
              'mapping_budget_seconds': seconds, 'features_reused': True, 'matches_reused': True,
              'source': str(previous), 'status': 'mapping', 'compute_backend': compute}
    save_json(folder / 'continuation_report.json', report)
    models_dir = folder / 'blocks' / '0000' / 'sparse'
    models_dir.mkdir(parents=True)
    print(f"[coverage] Continuing {reconstruction.num_reg_images()} cameras for up to {seconds:g}s using saved matches", flush=True)
    models = pycolmap.incremental_mapping(folder / 'database.db', folder / 'images', models_dir,
                                         input_path=seed, options=options)
    if not models:
        raise RuntimeError('Continuation returned no camera model')
    model_id, reconstruction = max(models.items(), key=lambda row: row[1].num_reg_images())
    binary = models_dir / str(model_id)
    binary.mkdir(exist_ok=True)
    reconstruction.write(binary)
    save_json(folder / 'current_map.json', {'binary_model': str(binary.resolve()),
              'update_index': 0, 'continued_from': str(previous)})
    report.update(status='complete', registered_frames=reconstruction.num_reg_images(),
                  elapsed_seconds=time.monotonic() - started, output_model=str(binary.resolve()))
    save_json(folder / 'continuation_report.json', report)
    print(f"[coverage] Continued model has {reconstruction.num_reg_images()} cameras", flush=True)
    return folder.parent, report
