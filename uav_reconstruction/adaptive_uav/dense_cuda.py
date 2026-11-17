"""Resume CUDA COLMAP depth estimation, then fuse and texture the saved map."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import pycolmap

from .mesh_viewer import write_mesh_viewer


def ply_vertices(path):
    with path.open('rb') as stream:
        header = stream.read(4096).split(b'end_header', 1)[0].decode('ascii', 'replace')
    match = re.search(r'^element vertex (\d+)$', header, re.M)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--openmvs', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--three-assets', type=Path, required=True)
    parser.add_argument('--max-image-size', type=int, default=1280)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-views', type=int, default=0,
                        help='Bound dense reference views using coverage and parallax; 0 keeps the existing plan')
    parser.add_argument('--source-views', type=int, default=6)
    parser.add_argument('--spatial-plan', action='store_true',
                        help='Allocate source cameras by spatial support at the depth filter angle')
    parser.add_argument('--stable-normals', action='store_true',
                        help='Fit reliable local depth normals and automatically check cross-view agreement')
    parser.add_argument('--quality-report', action='store_true',
                        help='Automatically audit coverage and report missing evidence')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--fusion-min-pixels', type=int, choices=range(3, 11), default=5,
                        help='Minimum agreeing depth observations per fused point (default: 5)')
    parser.add_argument('--skip-texture', action='store_true',
                        help='Export RGB vertex PLY/GLB without computing an image texture atlas')
    parser.add_argument('--depth-only', action='store_true',
                        help='Stop after PatchMatch depth maps for hybrid fusion')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    metrics_path = workspace / 'dense_metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({'pipeline': 'CUDA COLMAP PatchMatch, COLMAP fusion, colored mesh',
                    'cuda_available': bool(pycolmap.has_cuda), 'max_image_size': args.max_image_size,
                    'source_model': str(workspace / 'sparse')})

    def save(stage):
        metrics['status'] = stage
        metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
        print(f'[{stage}]', flush=True)

    if not pycolmap.has_cuda:
        raise RuntimeError('Installed PyCOLMAP lacks CUDA support')
    if not (workspace / 'images').exists() or not (workspace / 'sparse/cameras.bin').exists():
        raise RuntimeError('Missing undistorted images or sparse model')
    # Thin recovery workspaces may contain only poses/images and a view plan.
    # Native COLMAP aborts instead of raising when its output parents are absent.
    for folder in ('depth_maps', 'normal_maps', 'consistency_graphs'):
        (workspace / 'stereo' / folder).mkdir(parents=True, exist_ok=True)

    if args.max_views:
        from .dense_views import plan_dense
        if not (workspace / 'dense_plan.json').exists():
            if list((workspace / 'stereo/depth_maps').glob('*.bin')):
                raise RuntimeError('Use a fresh workspace when changing the dense view plan')
            if args.spatial_plan:
                from .dense_coverage import plan_coverage
                plan_coverage(workspace, args.max_views, args.source_views)
            else:
                plan_dense(workspace, args.max_views, args.source_views)
        plan = json.loads((workspace / 'dense_plan.json').read_text())
        metrics['dense_reference_views'] = plan['reference_views']
        metrics['max_source_views'] = plan['max_source_views']
        metrics['dense_planner_version'] = plan.get('planner_version', 1)
    config_lines = (workspace / 'stereo/patch-match.cfg').read_text().splitlines()
    reference_names = [line.strip() for line in config_lines if line.strip() and not line.startswith('#')][::2]
    settings = {'max_image_size': args.max_image_size, 'iterations': args.iterations,
                'references': reference_names,
                'configuration': (workspace / 'stereo/patch-match.cfg').read_text()}
    filter_angle = 3.
    if (workspace / 'dense_plan.json').exists():
        plan = json.loads((workspace / 'dense_plan.json').read_text())
        if plan.get('planner_version', 1) >= 2:
            filter_angle = float(plan['filter_min_triangulation_angle'])
            settings['filter_min_triangulation_angle'] = filter_angle
    settings_path = workspace / 'depth_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError('Depth settings changed; choose a new workspace to avoid mixing cached maps')
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')

    if args.depth_only:
        missing_depth = [name for name in reference_names
                         if not (workspace / 'stereo/depth_maps' / (name + '.geometric.bin')).is_file()]
        if missing_depth:
            save('gpu_depth_running')
            tick = time.monotonic()
            options = pycolmap.PatchMatchOptions(max_image_size=args.max_image_size,
                                                 gpu_index='0', cache_size=4,
                                                 num_threads=args.threads,
                                                 geom_consistency=True,
                                                 filter_min_triangulation_angle=filter_angle,
                                                 num_iterations=args.iterations)
            pycolmap.patch_match_stereo(workspace, options=options)
            metrics['gpu_depth_seconds'] = time.monotonic() - tick
        missing_depth = [name for name in reference_names
                         if not (workspace / 'stereo/depth_maps' / (name + '.geometric.bin')).is_file()]
        if missing_depth:
            raise RuntimeError(f'PatchMatch left {len(missing_depth)} reference maps missing')
        save('depth_only_complete')
        return

    fused = workspace / 'scene_dense.ply'
    fusion = pycolmap.StereoFusionOptions(max_image_size=args.max_image_size,
                                        num_threads=args.threads, cache_size=4,
                                        min_num_pixels=args.fusion_min_pixels)
    fusion_settings = {'min_num_pixels': args.fusion_min_pixels,
                       'max_image_size': args.max_image_size,
                       'max_reproj_error': fusion.max_reproj_error,
                       'max_depth_error': fusion.max_depth_error,
                       'max_normal_error': fusion.max_normal_error,
                       'input_type': 'geometric'}
    if args.stable_normals:
        from .depth_normals import NORMAL_POLICY
        fusion_settings['normal_policy'] = NORMAL_POLICY
    fusion_manifest = workspace / 'fusion_settings.json'
    if fusion_manifest.exists():
        if json.loads(fusion_manifest.read_text()) != fusion_settings:
            raise RuntimeError('Fusion settings changed; use a separate workspace for comparison')
    elif fused.is_file() and args.fusion_min_pixels != metrics.get('fusion_min_pixels', 5):
        raise RuntimeError('Cached cloud uses different fusion support; use a separate workspace')
    fusion_manifest.write_text(json.dumps(fusion_settings, indent=2) + '\n')
    metrics['fusion_settings'] = fusion_settings
    metrics['fusion_min_pixels'] = args.fusion_min_pixels
    if not fused.is_file():
        missing_depth = [name for name in reference_names
                         if not (workspace / 'stereo/depth_maps' / (name + '.geometric.bin')).is_file()]
        if missing_depth:
            save('gpu_depth_running')
            tick = time.monotonic()
            options = pycolmap.PatchMatchOptions(max_image_size=args.max_image_size,
                                                 gpu_index='0', cache_size=4,
                                                 num_threads=args.threads,
                                                 geom_consistency=True,
                                                 filter_min_triangulation_angle=filter_angle,
                                                 num_iterations=args.iterations)
            pycolmap.patch_match_stereo(workspace, options=options)
            metrics['gpu_depth_seconds'] = time.monotonic() - tick
            save('gpu_depth_complete')
        if args.stable_normals:
            from .depth_normals import stabilize_workspace
            save('normal_estimation_running')
            metrics['normal_estimation'] = stabilize_workspace(workspace)
            save('normal_estimation_complete')
        pycolmap.stereo_fusion(fused, workspace, input_type='geometric',
                               options=fusion, output_type='PLY')
    if not fused.is_file():
        raise RuntimeError('GPU depth fusion produced no point cloud')
    metrics['dense_points'] = ply_vertices(fused)
    if metrics['dense_points'] < 1000:
        raise RuntimeError(f"Only {metrics['dense_points']} fused points")
    save('dense_point_cloud_complete')

    runtime_lib = args.runtime.resolve() / 'usr/lib/x86_64-linux-gnu'
    linker = runtime_lib / 'ld-linux-x86-64.so.2'

    def openmvs(stage, binary, cmd, expected):
        target = workspace / expected
        if target.is_file() and target.stat().st_size > 0:
            print(f'[{stage}] reusing {target}', flush=True)
            return
        save(stage + '_running')
        logfile = workspace / f'{stage}.log'
        executable = args.openmvs.resolve() / binary
        launch = ([str(linker), '--library-path', str(runtime_lib), str(executable)]
                  if linker.is_file() else [str(executable)])
        with logfile.open('w') as stream:
            subprocess.run(launch + cmd, cwd=workspace, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f'{stage} produced no {target}; see {logfile}')
        save(stage + '_complete')

    # ReconstructMesh needs per-point camera visibility. A plain COLMAP PLY
    # omits those lists: import its companion .vis before passing it to meshing.
    visibility = workspace / (fused.name + '.vis')
    if not visibility.is_file():
        raise RuntimeError(f'Missing fused visibility {visibility}; rerun fusion')
    importer = workspace / 'mvs_import'
    importer.mkdir(exist_ok=True)
    for name, target in {'images': workspace / 'images', 'sparse': workspace / 'sparse',
                         'fused.ply': fused, 'fused.ply.vis': visibility}.items():
        link = importer / name
        if not link.exists():
            link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    openmvs('visibility_import', 'InterfaceCOLMAP',
            ['-i', str(importer), '-o', 'scene_dense_visibility.mvs'],
            'scene_dense_visibility.ply')
    imported = workspace / 'scene_dense_visibility.ply'
    with imported.open('rb') as stream:
        header = stream.read(4096).split(b'end_header', 1)[0]
    if b'view_indices' not in header:
        raise RuntimeError('Imported cloud has no camera visibility; refusing unsupported meshing')
    if (workspace / 'scene_mesh.ply').exists() and metrics.get('mesher') == 'COLMAP Poisson':
        raise RuntimeError('Existing mesh is a Poisson fallback; use a separate output for corrected meshing')
    openmvs('mesh', 'ReconstructMesh',
            ['-i', 'scene_dense_visibility.mvs', '-p', imported.name,
             '-o', 'scene_mesh.mvs', '--max-threads', str(args.threads)], 'scene_mesh.ply')
    metrics['mesher'] = 'OpenMVS visibility graph cut'
    if not (workspace / 'mesh_support.json').exists():
        from .supported_mesh import trim_mesh
        metrics['mesh_support'] = trim_mesh(workspace)
    else:
        metrics['mesh_support'] = json.loads((workspace / 'mesh_support.json').read_text())
        if not metrics['mesh_support'].get('passed', True):
            raise RuntimeError('Previously generated mesh failed the dense support gate')
    # Native cleanup repairs topology after support trimming, without filling
    # missing surfaces or smoothing the measured geometry.
    openmvs('mesh_cleanup', 'ReconstructMesh',
            ['-i', 'scene_dense_visibility.mvs', '--mesh-file', 'scene_mesh.ply',
             '-o', 'scene_mesh_clean.mvs', '--close-holes', '0', '--smooth', '0',
             '--max-threads', str(args.threads)], 'scene_mesh_clean.ply')
    shutil.copy2(workspace / 'scene_mesh_clean.ply', workspace / 'scene_mesh.ply')
    metrics['mesh_vertices'] = ply_vertices(workspace / 'scene_mesh.ply')
    from .supported_mesh import color_supported_mesh
    metrics['coloring'] = color_supported_mesh(workspace)
    metrics['mesh_faces'] = metrics['coloring']['colored_faces']
    save('mesh_complete')
    if metrics['mesher'] == 'OpenMVS visibility graph cut' and not args.skip_texture:
        openmvs('texture', 'TextureMesh',
                ['-i', 'scene_dense_visibility.mvs', '-m', 'scene_mesh.ply', '-o', 'scene_textured_glb.mvs',
                 '--export-type', 'glb', '--max-threads', str(args.threads)],
                'scene_textured_glb.glb')
        metrics['textured_glb_bytes'] = (workspace / 'scene_textured_glb.glb').stat().st_size
        save('textured_mesh_complete')
    else:
        save('colored_mesh_complete')
    if args.quality_report:
        from .dense_quality import audit_dense
        from .planar_completion import run as complete_planes
        save('planar_completion_running')
        try:
            metrics['planar_completion'] = complete_planes(workspace)
        except Exception as error:
            metrics['planar_completion'] = {'status': 'failed', 'error': str(error),
                                            'measured_mesh_modified': False}
        save('planar_completion_complete')
        save('quality_audit_running')
        quality = audit_dense(workspace)
        metrics['quality_status'] = quality['status']
        metrics['quality_report'] = 'quality_report.json'
        metrics['quality_warning'] = ('Incomplete surface coverage; see the coverage report.'
                                      if quality['status'] == 'incomplete' else
                                      'Broad visible coverage; geometric accuracy remains unverified.')
        save('complete')
    print(write_mesh_viewer(workspace, args.three_assets), flush=True)


if __name__ == '__main__':
    main()
