# GeoStreamFusion

Reconstruction-aware frame selection and compact depth learning for efficient UAV video-to-3D reconstruction. Developed as a Smart India Hackathon (SIH) problem project at IIT Tirupati / SEVA Lab.

The pipeline recovers camera poses, evaluates candidate frames, estimates multiview depth, checks geometric consistency and exports triangle meshes. The learned selector predicts whether to **ADD** a candidate, **SWAP** it for a retained occupant or **KEEP** the current frame set.

This repository contains the implementation, configuration templates, operational scripts and tests. Experimental outputs, datasets, trained weights, meshes and machine environments are stored separately.

[Project website](https://cs25m213-gif.github.io/3D_reconstruction/) · [Poster (PDF)](https://cs25m213-gif.github.io/3D_reconstruction/assets/poster.pdf) · [Architecture and implementation status](docs/architecture.md) · [Source map](docs/CODE_MAP.md) · [External dependencies](docs/DEPENDENCIES.md)

## Source layout

| Location | Contents |
|---|---|
| `uav_reconstruction/adaptive_uav/` | Camera recovery, reconstruction, depth models, frame selectors, fusion, dataset preparation, training and evaluation |
| `uav_reconstruction/configs/` | Reconstruction presets and training contracts; paths refer to separately provisioned artifacts |
| `uav_reconstruction/tests/` | Geometry, data admission, streaming, selector and reconstruction tests |
| `uav_reconstruction/drone_telemetry/` | Frame extraction, telemetry alignment and export helpers |
| `scripts/` | Dataset/selector utilities and source-level diagnostic tools |
| `tools/` | Optional backend setup helpers and licensed browser-viewer dependencies |
| `docs/` | Setup, source map, architecture and implementation notes |

## Setup

Use a Python virtual environment. The original workstation used Python 3.12; CUDA backend builds require matching PyTorch, CUDA and compiler versions.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

GPU COLMAP, OpenMVS and optional model backends require additional installation; see [dependencies](docs/DEPENDENCIES.md). Training data and model checkpoints must be provided separately. No pretrained weights or calibration for a new camera are included.

## Reconstruct a saved video

```bash
./reconstruct.sh /path/to/video.mp4 --no-serve
```

For the optional pose-conditioned multiview path, install its backend first:

```bash
./setup_multiview.sh
./reconstruct.sh /path/to/video.mp4 --multiview --no-serve
```

These commands expose existing reconstruction paths; enabling a learned selector or replacing the depth backend requires its compatible checkpoint and validation configuration. The exported source does not automatically promote a trained model. Estimated calibration, sparse references and teacher predictions retain their respective limitations.

## Train and evaluate

The large visual selector and compact depth model have separate entrypoints:

```bash
./scripts/run_module.sh adaptive_uav.train_complete_selector --help
./scripts/run_module.sh adaptive_uav.train_depth_student --help
./scripts/run_module.sh adaptive_uav.prepare_available_selector_data --help
```

Supply dataset/output paths and configuration for the chosen experiment. Historical training contracts refer to `runs/` paths as runtime inputs and outputs; their files are deliberately excluded from Git. Dataset preparation preserves held-out flight/sequence roles. The selector learns signed changes in reconstruction quality with depth held fixed within a label round; refresh utility labels after changing depth.

For the imported expanded-selector reconstruction evaluation, set `GEOSTREAM_SELECTOR_DATA` to the prepared dataset root and provide output directory and worker count as positional arguments. Its checkpoint names encode that historical protocol; inspect the module before launching a different experiment.

## Checks

```bash
PYTHONPATH=.:uav_reconstruction .venv/bin/python -m unittest \
  uav_reconstruction.tests.test_available_selector_data \
  uav_reconstruction.tests.test_reconstruction_io \
  uav_reconstruction.tests.test_selector_robustness -v
```

The complete test suite includes optional GPU/backend and dataset-specific checks. Passing the source checks does not certify trained-model quality or geometry on a new video.

## Team

Venkatasai Eswar Nandu (CS25M213), Atul Tripathi (CS25M251), M Rajvardhan (CS25M107), Atri Sukul (CS25M206). Supervisor: Dr. Chalavadi Vishnu. Indian Institute of Technology Tirupati / SEVA Lab.

Third-party packages retain their own licenses and attribution; see [third-party notices](THIRD_PARTY.md).
