"""Verify native CUDA dense bundle adjustment without trusting a CUDA flag."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pycolmap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(12)
    xyz=rng.uniform([-1,-.6,3],[1,.6,6],(100,3))
    cameras=['# synthetic SIMPLE_PINHOLE calibration','1 SIMPLE_PINHOLE 1280 720 940 640 360']
    images=[]
    centers=[]
    for i in range(20):
        center=np.array([.03*i,.1*np.sin(i),0.]);centers.append(center)
        images.append(f'{i+1} 1 0 0 0 {-center[0]} {-center[1]} 0 1 frame_{i:06d}.jpg')
        pc=xyz-center;uv=940*pc[:,:2]/pc[:,2,None]+[640,360]
        uv+=rng.normal(0,.05,uv.shape)
        images.append(' '.join(f'{u} {v} {j+1}' for j,(u,v) in enumerate(uv)))
    points=[]
    for j,point in enumerate(xyz):
        perturbed=point+rng.normal(0,.003,3)
        tracks=' '.join(f'{i+1} {j}' for i in range(20))
        points.append(f'{j+1} {perturbed[0]} {perturbed[1]} {perturbed[2]} 128 128 128 0 {tracks}')
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)
        (p/'cameras.txt').write_text('\n'.join(cameras)+'\n')
        (p/'images.txt').write_text('\n'.join(images)+'\n')
        (p/'points3D.txt').write_text('\n'.join(points)+'\n')
        reconstruction=pycolmap.Reconstruction();reconstruction.read_text(p)
        options=pycolmap.BundleAdjustmentOptions()
        options.ceres.use_gpu=True;options.ceres.gpu_index='0'
        options.ceres.min_num_images_gpu_solver=0;options.ceres.auto_select_solver_type=False
        solver=options.ceres.solver_options
        solver.linear_solver_type=type(solver.linear_solver_type).DENSE_SCHUR
        solver.dense_linear_algebra_library_type=type(solver.dense_linear_algebra_library_type).CUDA
        solver.max_num_iterations=5
        config=pycolmap.BundleAdjustmentConfig()
        for image_id in reconstruction.reg_image_ids():config.add_image(image_id)
        summary=pycolmap.create_default_bundle_adjuster(options,config,reconstruction).solve().todict()['ceres_summary']
        valid=(summary['dense_linear_algebra_library_type'].name=='CUDA' and
               summary['termination_type'].name in ('CONVERGENCE','NO_CONVERGENCE') and
               np.isfinite(summary['final_cost']) and summary['final_cost']<summary['initial_cost'])
        if not valid:raise RuntimeError('CUDA solver did not pass the synthetic geometry test')
        result={'pycolmap_version':pycolmap.__version__,'pycolmap_module':pycolmap.__file__,
                'ceres_gpu_verified':True,'ceres_gpu_dense_verified':True,'ceres_gpu_sparse_verified':False,
                'preferred_backend':'CERES','caspar_verified':False,
                'ceres_test_evidence':{k:(v.name if hasattr(v,'name') else v) for k,v in summary.items()
                    if k in ('linear_solver_type_used','dense_linear_algebra_library_type','initial_cost','final_cost','total_time_in_seconds')},
                'note':'Native support test only; this is not a reconstruction accuracy guarantee.'}
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
