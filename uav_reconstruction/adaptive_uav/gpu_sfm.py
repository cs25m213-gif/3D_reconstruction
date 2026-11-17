"""Select GPU bundle adjustment only from a tested native COLMAP build."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=4)
def verified_capabilities(module_path, version):
    path = Path(module_path).resolve().parent.parent/'gpu_ba_verified.json'
    if not path.is_file():
        return {'caspar_verified': False, 'ceres_gpu_verified': False}
    data = json.loads(path.read_text())
    if data.get('pycolmap_version') != version or Path(data.get('pycolmap_module','')).resolve() != Path(module_path).resolve():
        return {'caspar_verified': False, 'ceres_gpu_verified': False}
    return data


def configure_mapping_ba(options, config, pycolmap, *, camera_models=None):
    capabilities = verified_capabilities(pycolmap.__file__, pycolmap.__version__)
    requested = str(config.get('sfm_ba_backend', 'auto')).lower()
    if requested not in ('auto', 'caspar', 'ceres'):
        raise ValueError('sfm_ba_backend must be auto, caspar or ceres')
    gpu = bool(config.get('sfm_use_gpu', False))
    verified_caspar = gpu and bool(capabilities.get('caspar_verified', False))
    caspar = verified_caspar and (requested == 'caspar' or
             requested == 'auto' and bool(config.get('prefer_caspar_auto', False)))
    if requested == 'caspar' and not verified_caspar:
        raise RuntimeError('CASPAR requested but this native build has not passed GPU validation')
    fallback_reason = None
    if caspar:
        if not camera_models:
            if requested == 'caspar':
                raise ValueError('CASPAR requires camera model names for compatibility checking')
            caspar = False
            fallback_reason = 'CASPAR camera compatibility cannot be checked without camera model names'
        # The verified build implements adapters for these two models only.
        # Other models can be silently skipped by its native optimizer.
        supported = set(capabilities.get('caspar_camera_models', ['SIMPLE_RADIAL', 'PINHOLE']))
        unsupported = set(camera_models or []) - supported
        if unsupported:
            caspar = False
            fallback_reason = 'CASPAR does not support camera models: ' + ', '.join(sorted(unsupported))
    backend = pycolmap.BundleAdjustmentBackend.CASPAR if caspar else pycolmap.BundleAdjustmentBackend.CERES
    options.ba_local_backend = backend
    options.ba_global_backend = backend
    options.ba_use_gpu = bool(caspar or gpu and capabilities.get('ceres_gpu_verified', False))
    options.ba_gpu_index = '0' if options.ba_use_gpu else '-1'
    return {'solver_policy': ('CUDA dense solves for eligible global models; small/local and sparse solves use CPU'
                if options.ba_use_gpu and not caspar else 'CASPAR CUDA' if caspar else 'CPU'),
            'local_backend':backend.name,'global_backend':backend.name,'gpu_bundle_adjustment':options.ba_use_gpu,
            'gpu_index':options.ba_gpu_index,'camera_registration_device':'CPU',
            'requested_backend':requested,'fallback_reason':fallback_reason,
            'camera_models':sorted(set(camera_models or [])),
            'native_module':pycolmap.__file__,'capability_test':capabilities}
