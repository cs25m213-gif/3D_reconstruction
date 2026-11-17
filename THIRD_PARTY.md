# Third-party components

GeoStreamFusion uses the following external projects. Their original licenses, authorship and checkpoint terms remain applicable.

| Component | Source | Included here |
|---|---|---|
| Three.js r169 | https://github.com/mrdoob/three.js/tree/r169 | Browser modules under `tools/three_0.169.0/`; MIT license included |
| OpenMVS | https://github.com/cdcseacave/openMVS | Depth-map utility reader and upstream license under `uav_reconstruction/apps/openMVS-source/` |
| Depth Anything 3 | https://github.com/ByteDance-Seed/Depth-Anything-3 | Adapter/setup code; upstream model source and weights installed separately |
| DPVO | https://github.com/princeton-vl/DPVO | Adapter and pinned setup helper; upstream source/weights installed separately |
| VisFusion | https://github.com/huiyu-gao/VisFusion | Adapter; upstream source and weights installed separately |
| SLAM3R | https://github.com/PKU-VCL-3DV/SLAM3R | Adapter; upstream source and weights installed separately |
| COLMAP / PyCOLMAP | https://github.com/colmap/colmap | Integration code; runtime installed separately |

Recorded optional source revisions: DA3 `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`; DPVO `0ac95b656d1fda91c271d2a106460d19ad966fc7`; SLAM3R `f531d841ab743217a4464344119a350eb0556d17`; VisFusion `a3823dcc35d026402f5620ff25e514fafb6d637b`.

This source export does not relicense third-party code or model weights.
