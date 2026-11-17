# Implementation map

The maintained package lives in `uav_reconstruction/adaptive_uav/`. Modules retain their original names to preserve imports and training/checkpoint contracts.

| Area | Main modules |
|---|---|
| Saved-video reconstruction | `reconstruct`, `adaptive_sfm`, `dense_cuda`, `multiview_dense`, `combined_reconstruct`, `final_reconstruct` |
| Online capture and tracking | `phone_live`, `phone_source`, `runtime`, `pose_dpvo` |
| Depth learning | `depth_student`, `depth_backend`, `train_depth_student`, `lidar_depth_supervision` |
| Expanded visual selector | `large_selector_model`, `train_complete_selector`, `selector_input_cache` |
| Action utility and measured gains | `contribution_labels`, `reconstruction_quality`, `reconstruction_selector`, `label_multisource_selector` |
| Dataset admission and geometry | `prepare_multisource_selector`, `prepare_selector_sfm`, `prepare_available_selector_data`, `prepare_available_selector_geometry` |
| Queues and late-label coverage | `supervise_complete_selector`, `supervise_available_selector_data` |
| Causal online experiments | `bin_memory_selector`, `profile_online_cascade`, `diagnose_online_cascade` (inspect module names and CLIs for the exact experiment) |
| Expanded-selector replay | `eval_selector_reconstruction` |
| Proposed memory/action model | `reconstruction_action_selector`; its feature schema differs from the existing 36-dimensional trainer |

Some modules are alternative reconstruction backends or research experiments, rather than stages of one integrated deployment. Read `architecture.md` and each CLI before combining them.

## Source consolidation

The local source was compared with the remote attribution project's Python tree. The remote-only reconstruction evaluation was added. Remote `prepare_selector_sfm_boost_impl.py` is byte-identical to the current local `prepare_selector_sfm.py`, so the duplicate was consolidated under the maintained name. The local selector cache retains the CPU-device guard absent from the remote copy. Existing geometry-admission and worker-thread fixes were preserved.

`source_import.json` records imported-file hashes. It describes this source export, not dataset/training completion or checkpoint availability.
