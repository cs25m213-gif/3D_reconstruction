# Reconstruction-driven frame selector

These scripts and modules came from the experiments on 10.23.20.60 (`/home/vishnu/geostreamfusion_attribution_20261108`). Results and architecture are in `GEOSTREAMFUSION_COMPLETE_REPORT.md`.

## Modules in `uav_reconstruction/adaptive_uav/`

| File | Purpose |
|---|---|
| `contribution_labels.py` | Shapley and leave-one-out frame contributions; `load_clip` (fixed-context student depth plus LiDAR reference) |
| `train_contribution_selector.py` | DINO ListNet scorer; `dino_bins`, `spearman` |
| `reconstruction_selector.py` | Depth-aware DAgger selector. Also provides the shared crash-safe `parallel()` helper and `load_clip_cached` (student depth and confidence cached per clip) |
| `bin_memory_selector.py` | Depth-free online bin-memory selector with ADD/SWAP gains and a sigmoid scorer (the user's design) |

## Running

```bash
ROOT=/path/to/repo PY=/path/to/python CONFIG=... STUDENT=... LABELS=... GREEDY=... \
  ./run_stage.sh bin_memory_selector runs/<work_dir> 56
```

The work directory's `tables/` and `student_depth/` caches are reused across runs.

Prerequisites:

- `LABELS` is the output of `contribution_labels`.
- `GREEDY` is the output of `greedy_selector.py`.

## Diagnostic experiments (kept as run)

- `drop_test.py`, `consistency_test.py`, `greedy_selector.py`, `analyse_picks.py`, `depth_error_selector.py`
- `run_greedy.sh`, `run_depth_error.sh`, `run_conf.sh`

These still contain the 10.23.20.60 paths (`/home/vishnu/...`, `runs/contract_vishnu.json`). Edit `M`, `CONFIG` and `cd` before running them here.

`depth_error_selector.py` early-stops on validation, so its numbers are leaky. It is superseded by the two selectors.
