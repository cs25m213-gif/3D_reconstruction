# Pose-conditioned reconstruction

Use the same pretrained model and fixed policy on another saved video:

```bash
cd /home/mtp26/Documents/3d_reconstruction
./reconstruct.sh /path/to/video.mp4 --multiview
```

A session folder containing `video.mp4` is also accepted. On another installation,
run `bash setup_multiview.sh` once in the existing reconstruction environment.
Weights download automatically on the first run. The tested model is
`depth-anything/DA3-BASE`, Hugging Face snapshot
`f4a6c9b3c95e41c82048423d3493a81ec3fa810e`. The provider checkout is
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. This is pretrained inference;
no model was trained on your videos.

## What changed and why

The previous hybrid estimates Depth Anything V2 independently for each image.
A global scale/shift fit can fail for a whole frame, and textureless regions
often fail the subsequent agreement gate. Only six temporal neighbours were
used by that gate. Mesh cleanup also discards any component smaller than 1%
of the complete mesh; its cutoff grows with the scene. The prior audits found
32.9% measured depth, 16.4% accepted learned depth, and 16.4% of raw mesh faces
removed by cleanup in the room capture. These are different quantities.

The new path in `adaptive_uav/multiview_dense.py`:

1. Uses undistorted images and registered COLMAP camera poses.
2. Selects four covisible frames by shared 3D tracks and parallax, including previous frames
   as context, and predicts their depth jointly with Depth Anything 3.
3. Fits one common scale for the fragment against measured depth. Spatial
   image blocks are held out for a consistency check. It does not fit a separate
   shift to every image. A bounded smooth local scale correction is then fitted
   from measured depth outside spatial holdout blocks; it is used only when it
   improves median holdout error by 20% without worsening the 90th percentile.
   This corrects spatial distortion that one global scale cannot fix. Scale
   remains in COLMAP units. The holdout is also used for correction selection,
   so it is an internal check rather than independent validation.
4. Caches predictions, camera calibration, confidence, source names, and checks.
   Matching inputs resume; changed inputs or inference code require fresh output.
5. Retains measured depth, checks learned depth against other camera views,
   rejects foreground contradictions with measured surfaces, and removes depth
   discontinuities. Occlusion is handled separately from contradiction.
6. Produces two coloured TSDF meshes: `scene_supported` uses measured and
   cross-view-supported depth; `scene_completed` additionally uses learned
   estimates that pass the confidence/alignment gates. Small-component cleanup
   has a fixed 50-face limit instead of a scene-size-dependent cutoff. Depth
   limits use each frame's range. A global median-derived far cutoff previously
   clipped distant surfaces in captures with a large depth range. TSDF volumes
   are split into groups spanning at most a fourfold range of frame median
   depths, with resolution scaled per group and meshes exported in the common
   COLMAP world frame. This bounds memory; seams or overlap between resolutions
   remain possible.
7. Renders both meshes from sampled registered camera poses (including cameras
   omitted from depth estimation) and reports ray coverage and
   disagreement with available measured depth, with comparison images.

The viewer opens the completed mesh and links to the supported version and
rendered comparisons. Completed surfaces are inferred; multiple predictions
agreeing is not proof of correct geometry. One visible source match is sufficient
for the supported tier when that match has measured depth, otherwise two
learned source maps must agree. Nearby cameras with negligible baseline do not
count as confirmation.

## Reuse an existing dense workspace

This avoids rerunning COLMAP and PatchMatch:

```bash
export PYTHONPATH="$PWD/.gpu_pycolmap:$PWD/uav_reconstruction:$PWD/tools/Depth-Anything-3/src"
export GEOSTREAM_TOOLS=/home/mtp26/Desktop/geostreamfusion_online/tools
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -u \
  -m adaptive_uav.multiview_dense \
  --workspace /path/to/dense_workspace \
  --output /path/to/new_output
```

The workspace needs `images/`, an undistorted pinhole `sparse/` model, and
`stereo/depth_maps/*.geometric.bin`. It operates on available measured-depth
reference views; it does not silently recover frames omitted upstream.
`--stage infer` populates the cache; `--stage fuse` reuses it. `--max-frames`
is available for small smoke tests. Default inference uses four views at 504 px
long side, tested on the 4 GB GPU. The CLI accepts a different compatible DA3
model/resolution, which changes the cache identity.

## Coverage and accuracy

Final room test, 128 depth references, with 16 poses sampled across all 207
registered cameras (including cameras not selected for depth inference):

| Mesh | Rendered coverage | Overlapping pixels within 5% of measured depth |
| --- | ---: | ---: |
| Previous hybrid | 58.1% | 95.9% |
| New supported | 79.8% | 95.6% |
| New completed | 92.4% | 93.4% |

The depth-agreement column is the mean per-view fraction over pixels with
available measured depth. The more complete mesh has a real accuracy tradeoff.
The room output is `runs/session_20261011_202810_corrected/dense_multiview_corrected`.
Open `mesh_viewer.html` and `coverage.html` there.

The identical policy was also run on the older 152-frame corridor workspace.
Rendered coverage rose from 33.5% to 94.7%, but only about 42.7% of overlapping
pixels agree within 5% of measured depth. Its output is flagged
`inconsistent_geometry` and is **not an accepted reconstruction**. This is why
coverage alone cannot be the success criterion. It needs an upstream geometry
investigation/reconstruction before that capture can be trusted. Existing
nearby depth maps agreeing does not establish globally consistent camera poses.

The quality gate reports `inconsistent_geometry` when mean measured-depth
agreement is below 80%, even when rendered coverage is high. Inference failures,
rejected views, component cleanup, and unmeasured completion remain explicit.
The tested geometric functions pass 15 unit/regression tests, including a
synthetic near/far scene that would lose its far plane under the former cutoff.

These comparisons use COLMAP depth that also anchors fusion, so they are internal
consistency checks, not independent ground truth. Retained depth-pixel coverage,
rendered ray-hit coverage, and whole-scene completeness are different measures.
100% accurate coverage cannot be guaranteed for surfaces hidden from every camera,
reflective surfaces, moving objects, or textureless regions without enough depth
evidence. This pipeline can infer plausible missing surfaces but labels them.

Sources: [official DA3 code and model descriptions](https://github.com/ByteDance-Seed/Depth-Anything-3),
[pose-conditioned inference API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md).
