# Runtime and optional dependencies

The source export excludes installed environments, binary wheels, external model repositories and model weights. Set `GEOSTREAM_TOOLS` when your tool installations live outside this checkout. Root launch scripts expect `.venv/bin/python`; `scripts/run_module.sh` also supports `GEOSTREAM_PYTHON`.

- Core mapping: NumPy, SciPy, OpenCV and PyCOLMAP. CPU PyCOLMAP can be installed through `requirements.txt`; CUDA COLMAP/PyCOLMAP needs a compatible build. The original custom GPU wheel is not included.
- Dense reconstruction: COLMAP PatchMatch and OpenMVS command-line programs are external installations. Some paths expect `tools/OpenMVS_2.4.0` and `tools/noble_runtime`; inspect CLI flags/environment configuration for your machine.
- Depth Anything 3: `setup_multiview.sh` obtains the source at the recorded revision and installs the multiview requirements. Additional requirements files reflect the original workstation, rather than a freshly resolved lockfile. Model weights are obtained separately through the backend.
- DPVO: `tools/setup_streaming.py` installs the pinned optional repository and verifies the released checkpoint hash. It creates an isolated CUDA environment and performs compiler-dependent extension builds; inspect before running it.
- VisFusion: optional upstream source and its checkpoints must be placed in `VisFusion/`; `requirements_visfusion.txt` describes the isolated environment additions.
- SLAM3R: optional upstream source belongs in `tools/SLAM3R/`; its checkpoint manifest and checkpoints are external artifacts.
- Viewer assets: the small Three.js runtime files are included with the MIT license. The OpenMVS depth-map reader is retained with the upstream license in `uav_reconstruction/apps/openMVS-source/`.

Use the CLI help to configure paths. Some historical modules/configurations contain original server or experiment defaults; these do not grant access to data and must be set for a new installation. Production reconstruction, GPU extension builds and training have not been rerun as part of this source-only publication.
