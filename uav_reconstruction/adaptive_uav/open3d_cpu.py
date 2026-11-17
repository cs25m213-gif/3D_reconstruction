"""Load only Open3D's CPU extension, bypassing its GPU-probing initializer."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
import sys


@lru_cache(maxsize=1)
def cpu_bindings():
    # Fusion may already have imported the normal package. Loading the same
    # native extension under another name re-registers pybind types and fails.
    canonical = 'open3d.cpu.pybind'
    if canonical in sys.modules:
        return sys.modules[canonical]
    package = importlib.util.find_spec('open3d')
    if package is None:
        raise RuntimeError('Open3D CPU bindings are unavailable')
    directory = Path(package.origin).parent / 'cpu'
    paths = list(directory.glob('pybind*.so'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected a single Open3D CPU extension in {directory}')
    dependencies = subprocess.check_output(['ldd', str(paths[0])], text=True)
    if any(name in dependencies for name in ('libcuda.', 'libcudart.', 'libcublas.')):
        raise RuntimeError('The supplied Open3D CPU binding links CUDA')
    # Native extension entry point is PyInit_pybind. Loading the extension
    # directly never executes open3d/__init__.py or imports open3d.cuda.
    spec = importlib.util.spec_from_file_location(canonical, paths[0])
    module = importlib.util.module_from_spec(spec)
    # A subsequent normal import, or this helper imported via a second project
    # namespace, must reuse the same extension rather than initialize it again.
    sys.modules[canonical] = module
    spec.loader.exec_module(module)
    return module
