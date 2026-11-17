# Datasets for GeoStreamFusion training

Sizes are measured from the actual files. "On storage" means `10.23.20.60:/home/vishnu/datasets/vnext_downloads/`.

## What to use

| Priority | Dataset | Used for | Size | Status |
|---|---:|---|---:|---|
| Required | **UAVScenes** full-rate flights, 12 | Main training: RGB, LiDAR, poses | 90.4 GB (117 GB unpacked) | On storage |
| Required | **Terra maps** (part of UAVScenes) | Dense depth targets and LiDAR calibration | 6.8 GB (1.7 GB of meshes needed) | On storage; meshes also on 10.23.9.49 |
| Required | **UAVScenes labels** | Masks for vehicles and water | 8.0 GB | On storage |
| Optional | **MARS-LVIG** raw bags, 12 | Per-point LiDAR timing and IMU; might help the flights that fail alignment | 201.9 GB | Blocked by Google Drive quota; the pipeline does not need them |
| Recommended | **Your drone sessions**, 7 | Your real target domain, self-supervised | Already local | On 10.23.9.49 |
| Recommended | **Your own new captures with reference geometry** | The only honest final test | You create it | See below |
| Evaluation | **UseGeo** | Independent geometry check | Not measured | Manual browser download |
| Evaluation | **MUN-FRL** | External tracking and LiDAR check | 444.1 GB | Optional in the script |
| Evaluation | **NTU VIRAL** | External pose stress test | 88.4 GB | 7.3 GB fetched |
| Geometry check | **MatrixCity** one block | Exact-truth check of tools | 1.8 GB | On storage |
| Optional | **UAVScenes GNSS flights** | Other conditions, not new places | 52.8 GB | Not fetched |
| Optional | **Dronescapes** | Appearance variety | 393 GB full; subsets on storage | Subsets only |
| Optional | **Mid-Air** | Synthetic pretraining | About 316 GB for one selection | Email request; not needed to start |

## How the splits work

| Role | Flights | Rule |
|---|---|---|
| Train | AMtown01-03 | Student with LiDAR, labels, selector |
| Train, map only | HKairport01-03 | Student through map depth only, because LiDAR alignment failed. No selector labels |
| Validation | AMvalley01-03 | Early stopping and choosing models |
| Diagnostic | HKisland01-03 | Inspected many times already, so never a final test |
| Calibration | Each flight's `_0` clip | Fits time offset, extrinsics and scale; never scored |
| Final test | **None yet** | Needs locations nothing has touched |

Split by whole flight or location. Neighbouring frames are nearly duplicates, so a random split would fake good results. UAVScenes has only four places, which limits how far any result here generalizes.

## What each source may be trusted for

The order follows your proposal: LiDAR and map first, then multiview geometry, photometric, and the teacher last.

- **LiDAR** is measured truth once calibrated, and is the only thing validation scores against.
- **Terra maps** are dense, but they count as a map, not a sensor. A frame uses its map depth only if the map agrees with that frame's LiDAR.
- **DA3 teacher depth** is an estimate, and it inherited the SfM doming. It only fills pixels nothing better covers.
- **Dronescapes and your sessions** have no reference geometry, so they train through self-supervised losses and never measure accuracy.

## More data does not automatically help

v4 to v6 got worse when direct data was added. A likely reason is now testable: cached clips carried domed poses while direct data used true poses, so identical image evidence implied depths about 3.5% apart. De-doming removes that conflict. Add data one group at a time and compare against the previous checkpoint.

## Getting data onto the GPU workstation

Datasets stay on 10.23.20.60. These working subsets are copied to 10.23.9.49:

| What | Where on 10.23.9.49 | Size |
|---|---|---|
| 71 new 30-second clips from all 12 flights | `runs/vnext_clips_v1/`, `runs/vnext_clips_v1_lidar/` | 9.0 GB |
| Their semantic masks | `runs/vnext_dataset_v2/semantic_labels/` | 130 MB |
| Calibration, corrected poses and targets for the original 24 clips | `runs/vnext_dataset_v2/` | 1.3 GB |

Each new clip needs one reconstruction, which `./run_complete_training.sh --to newclips` runs. It uses the settings of the first 30 clips, and takes about 3 minutes per clip.

The 6 labelled clips of the original set are far too few to train a selector, which is why the new clips matter: they bring the count to about 42.

To build a different set, run this on 10.23.20.60 with the code copied there, then copy `runs/vnext_clips_v1/<clip>` and `runs/vnext_clips_v1_lidar/<clip>` across. Skip each clip's `source/` folder, because only `session/` and `selected_infos.json` are needed.

```bash
PY=env/bin/python; C=uav_reconstruction/configs/vnext_contract_v3.json; M=uav_reconstruction.adaptive_uav
E=/home/vishnu/datasets/vnext_downloads/extracted/uavscenes/interval1_CAM_LIDAR
$PY -m $M.prepare_uavscenes_clips plan  --config $C --extracted $E --per-flight 6
$PY -m $M.prepare_uavscenes_clips build --config $C --extracted $E
```

`plan` mixes straight runs with turns, because near-parallel straight flight is what causes doming. It avoids every frame of the existing clips.

## Your own capture campaign

This is the most valuable data you can add, and the only way to make a real generalization claim. Starting targets:

- At least 8 sites, with at least 3 kept completely untouched for the final test.
- At least 2 flights per site, at different times or on different routes.
- Fixed lens mode, stabilization off, fixed exposure and focus, calibration file saved.
- Nadir, 45 and 60 degree views, orbits, lawnmower lines, hover, and return loops. **Include gently banked turns or tilted views**, because they prevent doming.
- Record video timestamps, IMU, GNSS or RTK, and clock offsets.
- Reference geometry: a LiDAR scan is best. Surveyed ground control points are the minimum, and a photogrammetric map must be reported as agreement with photogrammetry, not truth.

## Licences

Keep the licence register the download script wrote. UAVScenes, MARS-LVIG, Mid-Air, NTU VIRAL and UseGeo are non-commercial or academic use, and MatrixCity is CC BY-NC. MUN-FRL data is CC BY 4.0. UAVScenes' own licence metadata conflicts, so treat the stricter reading as binding, and check all of these before any commercial use.
