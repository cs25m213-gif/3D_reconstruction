# GeoStreamFusion: Complete Architecture and Work Log

*Reconstruction-driven frame selection, 2026-11-07 to 2026-11-12*

This file covers the whole conversation: the goal, your design, every architecture involved, every experiment and its result, every problem we hit and how it was fixed, the current state of the runs, and the next steps.

---

## Contents

1. [Goal](#1-goal)
2. [Your design (handwritten notes)](#2-your-design-handwritten-notes)
3. [Machines, data and paths](#3-machines-data-and-paths)
4. [Architecture A: the existing GeoStreamFusion training pipeline](#4-architecture-a-the-existing-geostreamfusion-training-pipeline)
5. [Architecture B: the reconstruction measurement (the training signal)](#5-architecture-b-the-reconstruction-measurement-the-training-signal)
6. [Architecture C: contribution labels and the DINO scorer (first experiments)](#6-architecture-c-contribution-labels-and-the-dino-scorer-first-experiments)
7. [Architecture D: depth-aware reconstruction selector (reference)](#7-architecture-d-depth-aware-reconstruction-selector-reference)
8. [Architecture E: depth-free online bin-memory selector (your design)](#8-architecture-e-depth-free-online-bin-memory-selector-your-design)
9. [Parallel execution and crash safety](#9-parallel-execution-and-crash-safety)
10. [All experiments and results](#10-all-experiments-and-results)
11. [Timeline of the conversation](#11-timeline-of-the-conversation)
12. [Problems hit and fixes](#12-problems-hit-and-fixes)
13. [Reinforcement learning assessment](#13-reinforcement-learning-assessment)
14. [Improvements for the complete training pipeline](#14-improvements-for-the-complete-training-pipeline)
15. [Current state and next steps](#15-current-state-and-next-steps)
16. [File index](#16-file-index)

---

## 1. Goal

The aim is to make the GeoStreamFusion frame selector choose the frames that give the best fused 3D reconstruction, judged against LiDAR.

Training it is hard because SfM, OpenMVS and meshing are not differentiable. So the reconstruction itself is used as the label:

1. Fuse the chosen frames' depth.
2. Measure the surface against LiDAR.
3. Train the selector to predict how much each frame adds.

---

## 2. Your design (handwritten notes)

### 2.1 How a frame helps: the score

```
Q(f | S) = F(S ∪ {f}) − F(S)              if f's bin is still empty        (ADD)
Q(f | S) = F(S ∪ {f} − {a}) − F(S)        if bin already holds frame a     (SWAP / replace)
```

- `F` is reconstruction quality (the F-score against LiDAR) and `S` is the selected set.
- A frame can fall into two bins; only the frame that is actually replaced counts.
- **Top-K rule:** keep candidates scoring at least `max − 5%·max`. The 5% is a hyperparameter.
- **Score updates:** scores are recomputed as frames are added or removed.
- **0.5 point:** a score of 0.5 means the frame gives no gain, and the note suggests 0.5 as the decision point.

### 2.2 Bin memory

- Bins come from DINO embedding similarity, using protected prototypes as in your ORTrack design.
- A new, novel frame opens a new bin.
- There is also a temporal memory.
- A missing frame creates holes, so a frame that is completely different from everything selected must be included.
- The notes link three things to each other: a bin, a region of the 3D model, and an image/frame.

### 2.3 Selector network

- Architecture: CNN → FFN → sigmoid score, recurrent with the bin-memory vectors.
- It must know the previous frame and the frames that overlap most with the candidate.
- **Depth must not be used as a parameter for selecting a frame.**

### 2.4 Forward pass and loss

```
frames ──► selector ──► depth student (distilled from Depth Anything 3 teacher) ──► reconstruction
                                                                                   │
                                                LiDAR ◄── reconstruction error ────┘
```

### 2.5 Concerns raised

- With no blur and no fast motion, selecting all frames might reconstruct better. The selector should avoid repetition without dropping useful frames.
- Reinforcement learning was raised as an idea, with the note that it is not necessary to implement.

### 2.6 Standing instruction

> Make the pipeline very effective, without issues; implement everything discussed; don't ask for permission; reconstruction must be very good; the whole pipeline must take its loss from the reconstruction and the selector must select frames by the score in the paper.

---

## 3. Machines, data and paths

| Machine | User | Role |
|---|---|---|
| 10.23.9.49 | iit-t | GPU machine with the main repo: `/home/iit-t/GeoStreamFusion_GPU_20261013/GeoStreamFusion_GPU` |
| 10.23.20.60 | vishnu | CPU machine (224 cores, 125 GB RAM, no GPU). Holds the datasets and the working copy `/home/vishnu/geostreamfusion_attribution_20261108/` |

Paths on 10.23.20.60:

- Python: `/home/vishnu/miniconda3/envs/drone3d/bin/python` (torch 2.5.1)
- Contract: `runs/contract_vishnu.json`
- Data: `/home/vishnu/geostreamfusion_more_data_20261104/data`
- Depth student: `.../data/student.pt`
- DINO embeddings: `.../data/embeddings`

**Access:** SSH uses this PC's key, which is in vishnu's `authorized_keys`. The transfer key `~/.ssh/geostreamfusion_transfer_ed25519` is not used; that access was refused twice and was not worked around.

**Splits:**

- Training: AMtown, 65 clips across flights AMtown01, AMtown02 and AMtown03. AMtown03 is held out for tuning.
- Validation: AMvalley, 20 clips across AMvalley01, AMvalley02 and AMvalley03. It is used only for the final evaluation.
- HKairport is in the contract's training split but has not been used yet.

---

## 4. Architecture A: the existing GeoStreamFusion training pipeline

This is what is on 10.23.9.49. It was read and not modified.

### 4.1 Stage chain (`run_complete_training.sh`, config `uav_reconstruction/configs/vnext_contract_v3.json`)

```
freeze → clips → newclips → scales → geometry → calibrate → dedome → targets → records → controls
       → student → temporal → embed → memory → labels → sources → dataset → selector → finetune → evaluate
```

| Stage | Module or meaning |
|---|---|
| `student` | Depth student distilled from the Depth Anything 3 (DA3-BASE) teacher |
| `embed`, `memory` | DINO embeddings and the causal memory (`memory_graph.CausalMemory`) |
| `labels` | `counterfactual_replay --replay-mode sequential` |
| `sources` | `source_view_labels` |
| `selector` | `train_utility_selector` |
| `finetune` | `finetune_selector` (reconstruction-reward fine-tuning via `selector_policy.sampling_distribution`) |
| `evaluate` | Final evaluation |

### 4.2 Selector-side modules

- `selector_features.py`: causal policy features. `CoverageMap` "needs no depth call for the candidate".
- `selector_policy.py`: `uniform`, `random_policy`, `rule`, `coverage_gain`, `UtilityPolicy`, `sampling_distribution`.
- Also: `train_reward_selector.py`, `reconstruction_reward.py`, `robust_selector_policy.py`.
- `counterfactual_replay.use_student`: each view's depth is predicted from `[view] + the two previous keyframes` (past-only sources).

### 4.3 Joint pilot (from `GEOSTREAMFUSION_ARCHITECTURE_20261108.md`)

```
 4 candidate source frames ──► SourceSelector ──► choose 1 of 6 source pairs ──► DepthStudent(48,24) ──► depth
   (visual CNN 16/32/32 +       (exact expected loss over the 6 pairs)          24-plane pose-conditioned
    relative-pose features)                                                       cost volume
                                                                                   │
                                     DA3-BASE teacher (fixed) ── distillation ─────┤
                                     LiDAR target-camera 3D point loss ────────────┘
```

**DepthStudent forward pass** (`depth_student.py`):

1. Warp source features onto 24 depth planes with `grid_sample`.
2. Build a cost volume that accumulates negative squared difference, correlation and valid count.
3. Pass it through `volume` → softmax → soft-argmin depth `low`, and compute the entropy.
4. Run `refine(cat(reference, rgb, low))` to get a residual, giving `depth = low · exp(0.5·tanh(r₀))`.
5. Confidence = `sigmoid(r₁ − entropy/planes)`.

**Pilot result:**

- It trained on 15 AMtown clips and validated on 8 AMvalley clips.
- Best point error 0.029409, against 0.030435 at the start and 0.029344 for the previous pilot, so it was **not promoted**.
- The checkpoint was chosen on the validation set, which lets validation data influence model choice.
- The neural selector is **not yet the global frame selector**. The classical route is blur, exposure and flow heuristics followed by COLMAP and OpenMVS.
- The document itself says the selector needs a reward or surrogate that "preserves this full outcome" (accuracy and coverage of the fused mesh).

---

## 5. Architecture B: the reconstruction measurement (the training signal)

```
clip frames ─► choose_keyframes(0.12) ─► use_student (fixed dense sources) ─► × depth_scale
                                                         │
selected subset S ─► TSDF fusion (hybrid_dense.make_volume / integrate, voxel 0.5 m)
                                                         │
               extract point cloud ─► surface_metrics vs LiDAR reference (tolerance 1 m) ─► F(S)
```

- **Fixed reference poses:** pose error is kept out of the label.
- **Fixed-context depth:** a frame's depth comes from a fixed dense set of source frames. It doesn't change with the selection, so `F(S)` is a well-defined set function.
- **F-score:** combines precision (the fused surface lies near the LiDAR) and recall (the LiDAR is covered).
- **Leakage control:**
  - Epochs and thresholds are chosen on AMtown03.
  - Training clips get out-of-fold (by flight) predictions.
  - Validation is used once, at the end.
  - Confidence intervals come from a bootstrap over flight clusters (`cluster_interval`).

---

## 6. Architecture C: contribution labels and the DINO scorer (first experiments)

### 6.1 Labels (`contribution_labels.py`)

- Each frame gets a Shapley value (random permutations) and a leave-one-out contribution.
- Bins come from `novelty_bins` (DINOv2 ViT-S/14, 384-d) with protected prototypes and a null slot, τ 0.80–0.95.

### 6.2 Scorer (`train_contribution_selector.py`)

- Inputs: DINO embedding plus runtime features (sharpness, brightness and similar).
- Model: MLP with a ListNet ranking loss.
- Selection: `dino_bins` change-point bins, then the best-scored frame in each bin.

### 6.3 Diagnostics built on top

| Script | What it does |
|---|---|
| `drop_test.py` | Removes low-value frames from the full set |
| `consistency_test.py` | Reprojects each frame's depth into frames i±1 and i±2 (5% relative tolerance); the agreement is a quality cue |
| `greedy_selector.py` | Exact greedy Δ(f\|S) oracle; stores states and candidate gains per step |
| `analyse_picks.py` | What explains the oracle's picks |
| `depth_error_selector.py` | Per-frame LiDAR depth error, a predictor for it, and selection by predicted error |

---

## 7. Architecture D: depth-aware reconstruction selector (reference)

File: `uav_reconstruction/adaptive_uav/reconstruction_selector.py`.

This was built before your second set of notes. It uses depth-derived features, so it serves as the **upper reference** for what depth information buys.

### 7.1 Per-frame table (cached in `<work>/tables/<clip>.npz`)

- **10 scalars:** `agree`, `agree_past`, `log_depth_median`, `depth_range_ratio`, `log_footprint`, `rel_height`, `off_nadir`, `sharp_z`, `bright_z`, `grad_mean`.
- **LiDAR depth error per frame:** the training target for `DepthNet`.
- **Coverage ("seen") bitmaps** of reference points, stored with packbits.

### 7.2 DepthNet: predicts a frame's depth error from appearance and scalars

```
DINO emb (384) ─► Linear 32 ─► SiLU ─► Dropout .2 ─┐
                                                   ├─► concat ─► Linear 64 ─► SiLU ─► Linear 1  → predicted log error
10 scalars ────────────────────────────────────────┘
```

Training predictions are out-of-fold by flight, so no clip sees its own label.

### 7.3 Scorer: greedy step scorer

- **Input features (20):** `pred_log_err` + the 10 scalars + `new_cover`, `cos_max`, `cos_mean`, `bin_count`, `null_bin`, `set_size`, `set_min_err`, `set_mean_err`, `err_minus_set_min`.
- **Network:** standardise → Linear 64 → SiLU → Linear 64 → SiLU → Linear 1. The output is the predicted gain Δ(f|S).

### 7.4 Training loop (DAgger / learning-to-search)

```
it 0: states from the exact greedy oracle ─► measure gains of all candidates ─► train Scorer
it k: roll out the current Scorer ─► at the states it visits, measure true gains (+ swap states) ─► aggregate ─► retrain
```

### 7.5 Selection at test time

- **Rollout:** add the best-scored frame per step, using the top-K band `max − band` with band 0 or 0.05.
- **Swap pass:** optionally replace frame a with frame b when `Δ(b|S−a) − Δ(a|S−a) > 0`.
- **Auto-stop:** stop when the best score is below τ (`stop_length`). Exploration is capped at 8 frames.
- **Tuning:** band, swap and τ are tuned on AMtown03 (`tune.json`).

### 7.6 Stages and outputs

- Stages: `tables → depthnet → label(it) / train(it) → tune → eval`, and `all` runs everything.
- Outputs: `eval/rows.jsonl`, `eval/summary.json`, `RESULT.md`.

---

## 8. Architecture E: depth-free online bin-memory selector (your design)

File: `uav_reconstruction/adaptive_uav/bin_memory_selector.py`.

This implements the second set of notes: depth-free for the candidate, online, with bin memory, and with add and swap scores.

### 8.1 Online loop (frames in time order)

```
for each incoming frame t:
    ├─ null slot?  (t ≥ 8 and log-sharpness < running 10% quantile) → reject (blurred/unusable)
    ├─ DINO embedding e_t → nearest protected prototype p
    │     cos(e_t, p) ≥ τ  → bin of p
    │     otherwise        → NEW bin (novel view → hole risk → strong candidate)
    ├─ bin empty?     ADD  candidate:  Q = F(S + t) − F(S)
    │  bin occupied?  SWAP candidate:  Q = F(S − a + t) − F(S)    (a = current occupant)
    ├─ BinScorer(t, a, memory state) → sigmoid score s
    └─ s ≥ θ  → take (add, or replace a);  otherwise skip
one frame per bin; S is the final selection that goes to the depth student and fusion
```

### 8.2 Inputs of BinScorer: no depth call for the candidate

- **Frame features** (all causal): `sharp_c`, `bright_c`, `move_m`, `turn_deg`, `off_nadir`, `rel_height`.
- **Memory and state features:**
  - `pred_err`: from ErrNet, which uses appearance and scalars only, no depth
  - `swap`: whether the candidate's bin is occupied
  - cosines to the bin prototype, the second-nearest prototype, the occupant and the closest selected frame: `cos_proto`, `cos_second`, `cos_occupant`, `cos_max_set`
  - `explained_set` and `explained_occupant`: how much of the candidate's view the already-selected frames' depth covers, computed by projecting their depth into the candidate camera. This is "the selector must know the overlapping frames".
  - `base_ratio`, `angle_nearest`, `since_selected`, `set_size`, `bins`
  - occupant statistics: `occ_pred_err`, `err_diff`, `occ_sharp_c`, `occ_age`, `occ_agree`

### 8.3 ErrNet: depth-free error predictor

```
DINO emb (384) ─► Linear 32 ─► SiLU ─► Dropout .3 ─┐
                                                   ├─► Linear 64 ─► SiLU ─► Linear 1
causal scalars ────────────────────────────────────┘
```

Its training predictions are out-of-fold by flight.

### 8.4 BinScorer: the sigmoid scorer

```
candidate emb e_f ─► proj: Linear(384→32) ─► SiLU ─► Dropout .3 ───────────────┐
occupant  emb e_a ─► same proj  × has_a (0 when the bin is empty) ─────────────┼─► concat
state features x  ─► standardise (μ, σ buffers) ───────────────────────────────┘
          ─► Linear(64+W → 64) ─► SiLU ─► Linear 32 ─► SiLU ─► Linear 1 ─► logit ─► sigmoid = score
```

- **Loss:** smooth-L1(logit, clamp(20·gain, ±4)) + BCE(logit, gain > 0.001).
- **Score meaning:** 0.5 is no gain, above 0.5 helps the reconstruction, below 0.5 hurts it. This is the "0.5 F" point in your notes.
- **Decision:** take the frame if score ≥ θ.

### 8.5 Training (DAgger with measured reconstruction gains)

```
prep:    causal scalars per frame + ErrNet (out-of-fold) → online/depth/
it 0:    roll out following the MEASURED gain (take if gain > 0), 15% random flips → label every decision
it k:    roll out following the CURRENT BinScorer, 15% flips → label every decision with measured add/swap gain
         train on all iterations' decisions (scorer_inner_it*.pt on inner flights, scorer_full_it*.pt on all)
tune:    τ ∈ {.85, .90, .95} × θ ∈ {.4, .5, .6, .7} on held-out flight AMtown03 → online/tune.json
eval:    validation (AMvalley, 20 clips) against baselines (below)
```

Each training clip is rolled out with all three τ values, so the scorer sees different bin granularities.

### 8.6 Evaluation baselines

| Baseline | Meaning |
|---|---|
| `online_oracle` | Same online loop, but decides with the measured gain (perfect score) |
| `greedy_oracle` | Offline exact greedy selection |
| `bin_first_frame` | First frame of each bin |
| `middle` | Middle frame of each bin, at matched count |
| `uniform` | Uniform frames, at matched count |
| `all_frames` | Every frame |

The evaluation also reports depth calls saved, decision time per frame, and a backward leave-one-out check of the selected frames.

Outputs: `online/summary.json`, `online/eval_rows.jsonl`, `online/progress.txt`.

---

## 9. Parallel execution and crash safety

Shared helper `parallel()` in `reconstruction_selector.py`, used by both selectors.

```
parallel(fn, items):
  round 1..3:  ProcessPoolExecutor (spawn context, initializer _init)
               a worker that segfaults breaks only that round → its unfinished items are retried
  round 4:     each remaining item in its own 1-worker executor (isolation)
  still lost → recorded in failures.txt; the run continues

_init (every worker):
  copy CFG · disable oneDNN · torch 1 thread (+1 interop) · faulthandler → <work>/faults/<pid>.txt
  OMP/MKL/OPENBLAS_NUM_THREADS = 1, set before spawn

load_clip_cached(name):   (replaces load_clip inside Context)
  first load:  run the student, record each predict() depth → <work>/student_depth/<student-key>/<clip>.npz
  later loads: replay the recorded depths (no student convolution runs at all)
  verified: max |depth diff| 0.0, intrinsics identical, F identical (0.7445 on AMvalley01_1)
```

The student key is the checkpoint's path hash, size and mtime, so a new student never reuses old depth.

---

## 10. All experiments and results

All results are on the 20 AMvalley validation clips unless marked otherwise.

### 10.1 Shapley labels

- The contributions sum to the clip's F.
- Median standard error is 0.007.
- About 11% of frames are more than 2 SE above zero.

### 10.2 DINO scorer vs rules (F at a budget of K frames)

| Method | K = 4 | K = 8 | K = 16 |
|---|---|---|---|
| uniform | 0.329 | 0.322 | 0.289 |
| middle of DINO bin | 0.356 | 0.345 | 0.311 |
| learned DINO scorer (Spearman 0.08) | 0.353 | 0.331 | 0.300 |
| Shapley-in-bin oracle | 0.375 | 0.354 | 0.320 |

### 10.3 Drop test

| Frames kept | F |
|---|---|
| All frames | 0.234 |
| Frames with positive contribution (70%) | 0.296 |
| Frames above 2 SE (11%) | 0.420 |

Using every frame hurts the reconstruction. This answers your concern about selecting all frames: in this data, adding everything lowers F.

### 10.4 Depth consistency

- Spearman 0.26 against contribution, AUC 0.63.
- All frames minus the least consistent 30%: F 0.277, a gain of +0.043 (CI +0.024..+0.055).
- At budget 16 the gain is +0.033 (CI +0.003..+0.060).
- Using consistency to pick the frame inside a bin does nothing.

### 10.5 Greedy Δ(f|S) oracle

| | K = 4 | K = 8 | K = 16 |
|---|---|---|---|
| greedy oracle | **0.485** | **0.489** | **0.489** |
| learned greedy (within-step Spearman 0.31) | 0.356 | 0.336 | 0.310 |

- The oracle is +0.13 to +0.18 above the middle-of-bin rule.
- Example: on AMvalley01_1, the best 2 frames give F 0.80, against 0.74 for all 35 frames.

### 10.6 Pick analysis: what explains the oracle's picks

| Factor | Correlation with gain |
|---|---|
| LiDAR depth error of the frame | ρ −0.71 (first pick), −0.67 (later picks) |
| Depth consistency | +0.44 / +0.11 |
| Median depth | −0.11 / −0.50 |
| Footprint | −0.16 / −0.49 |
| New coverage | −0.23 |
| DINO similarity | ≈ 0 |

- The first frame alone gives F 0.39, the second adds +0.06 and the third +0.025.
- After that, only 2–10% of candidates help.

**Conclusion:** a frame's value is dominated by how accurate its depth is. Appearance novelty alone doesn't predict it.

### 10.7 Runtime

- Extra features cost 1.2–1.8 ms/frame and a selection step about 0.8 ms, on CPU.
- You asked whether this increases inference time: the cost is small.

### 10.8 Depth-aware selector (Architecture D)

**Smoke test** (6 training + 6 validation clips, 5 epochs; checks function only):

| K | uniform | middle | selector | greedy oracle | selector − middle (95% CI) |
|---|---|---|---|---|---|
| 2 | 0.260 | 0.418 | 0.426 | 0.497 | +0.007 [−0.014, +0.019] |
| 3 | 0.350 | 0.433 | 0.463 | 0.524 | +0.030 [+0.000, +0.054] |
| 4 | 0.383 | 0.430 | 0.444 | 0.532 | +0.015 [−0.003, +0.040] |
| 8 | 0.375 | 0.420 | 0.436 | 0.535 | +0.017 [−0.040, +0.084] |

- Auto-stop chose 16 frames and reached F 0.405, against 0.314 for all frames.
- On the full data, DepthNet reached out-of-fold train Spearman 0.536 and validation Spearman 0.670, against 0.288 for consistency alone. The first iteration's scorer reached within-state Spearman 0.721 on AMtown03.

### 10.9 Depth-free online selector (Architecture E), smoke test

This was 6 validation clips with 1–4 training epochs, so it checks function only. Results were identical across three reruns, which shows the crash fixes didn't change any numbers.

| Method | F |
|---|---|
| greedy oracle | 0.521 |
| **online oracle (perfect add/swap score)** | **0.487** |
| middle of bin | 0.441 |
| bin first frame | 0.399 |
| learned online selector (about 2.7 frames) | 0.389 |
| uniform | 0.350 |
| all frames | 0.314 |

- The learned selector beats uniform by 0.039 (CI 0.036..0.041) and all frames by 0.075. It is 0.052 below the middle-of-bin rule, which is expected: its holdout Spearman was only about 0.17 after a few epochs.
- Decisions took 0.63 ms/frame, and 94.7% of depth calls were saved.
- **The key number is the online oracle at 0.487, close to the greedy oracle's 0.521.** One pass in time order with your add/swap score loses little compared with offline search.

### 10.10 Full runs (65 training clips, 20 validation clips; finished 2026-11-12)

The first table build lost its items to worker crashes twice, while it was filling the student-depth cache. The retry recovered all 85 clips, and there were no failed items in either run.

#### Depth-free online selector (Architecture E, your design)

**Training diagnostics:**

| Component | Result |
|---|---|
| ErrNet (depth-free error prediction) | Validation Spearman **0.22**; out-of-fold train 0.27 |
| BinScorer it 0 | AMtown03 holdout Spearman **0.548**, decision accuracy 0.66 |
| BinScorer it 1 | Spearman 0.452 |
| BinScorer it 2 | Spearman 0.441. Later DAgger iterations did not improve the holdout. |
| Tuning on AMtown03 | τ = 0.90, θ = **0.50**, giving F 0.583 with 6.2 frames. The optimum is exactly the 0.5 point in your notes. |

**Validation (20 clips):**

| Method | F | Selector minus method (95% CI) |
|---|---|---|
| greedy oracle | 0.479 | |
| **online oracle (perfect add/swap score)** | **0.451** | −0.090 [−0.135, −0.050] |
| middle of bin (matched count) | 0.375 | −0.014 [−0.063, +0.037] (no significant difference; wins on 10 clips, loses on 8) |
| **learned online selector** | **0.361** (3.3 frames of 52) | |
| bin first frame | 0.302 | +0.058 [+0.045, +0.078] |
| uniform (matched count) | 0.286 | +0.074 [+0.060, +0.093] |
| all frames | 0.234 | **+0.127 [+0.106, +0.147]** |

- **Cost:** 93.7% of depth calls are saved, decisions take 0.88 ms/frame, and there are 11.4 swaps per clip on average.
- **Backward check:** 18 of the 66 selected frames had a negative leave-one-out gain.

#### Depth-aware reference (Architecture D)

- DepthNet validation Spearman 0.670.
- The scorer's within-state holdout Spearman was 0.72, 0.71 and 0.71 across the three iterations.
- Tuning chose band 0, swap on, τ 0.035.

| K | uniform | middle | selector | no swap | with stop | greedy oracle | selector − middle (95% CI) |
|---|---|---|---|---|---|---|---|
| 2 | 0.239 | 0.364 | 0.375 | 0.376 | 0.383 | 0.452 | +0.011 [−0.022, +0.046] |
| 3 | 0.314 | 0.359 | 0.387 | 0.383 | 0.400 | 0.476 | +0.028 [−0.006, +0.067] |
| 4 | 0.329 | 0.356 | 0.394 | 0.394 | 0.408 | 0.485 | **+0.038 [+0.007, +0.074]** |
| 8 | 0.322 | 0.345 | 0.376 | 0.384 | 0.402 | 0.489 | +0.029 [−0.016, +0.079] |

- Auto-stop chose 5.7 frames and reached **F 0.402**, against 0.234 for all frames.
- Runtime: features 1.8 ms/frame, selection 1.3 ms/step.

#### What the full runs show

1. **Selection helps a lot.** Both selectors beat all frames by a wide margin: 0.361 and 0.402 against 0.234.
2. **Your online add/swap formulation is sound.** With a perfect score it reaches 0.451, only 0.028 below offline greedy search (0.479). That gap is too small to justify reinforcement learning.
3. **The bottleneck is the score's quality, not the formulation.** The learned score loses 0.09 to the online oracle. The weak link is depth-free error prediction (ErrNet Spearman 0.22, against 0.67 for DepthNet, which uses depth features). The pick analysis found that a frame's depth accuracy is the strongest factor (ρ −0.71). Without depth information the selector only ties the middle-of-bin rule; with it, it beats that rule significantly (+0.038 at K = 4, and 0.402 with auto-stop).
4. **Iteration 0 was the best scorer.** Later DAgger iterations lowered the holdout Spearman, so the next run should keep the best iteration on AMtown03 instead of always using the last one.

### 10.11 Student confidence as a selector input (2026-11-13; you allowed confidence)

**What was added:**

- The student-depth cache now also stores the student's per-pixel confidence. It was discarded by `use_student` before.
- Four per-frame features, fed into ErrNet and BinScorer: `conf_mean`, `conf_p10`, `conf_low` (fraction of pixels below 0.5) and `conf_z` (mean relative to earlier frames in the clip).
- The candidate's depth geometry is still not used.
- Tuning now searches the DAgger iteration × τ × θ grid on AMtown03.

Both full runs below use the new tuning, so they differ only in the confidence features.

| Run | ErrNet val Spearman | BinScorer holdout Spearman (it 0/1/2) | Tuned | Validation F | − middle (95% CI) |
|---|---|---|---|---|---|
| with confidence | 0.10 | 0.570 / 0.554 / 0.553 | it 2, τ 0.90, θ 0.6 | **0.349** | −0.020 [−0.043, −0.008] (worse) |
| without confidence | 0.22 | 0.548 / 0.452 / 0.441 | it 1, τ 0.90, θ 0.7 | **0.341** | −0.043 [−0.095, −0.002] (worse) |
| earlier run (last iteration, θ 0.5) | 0.22 | — | it 2, τ 0.90, θ 0.5 | 0.361 | −0.014 [−0.063, +0.037] |

**Why confidence did not help** (direct check: per-clip Spearman of confidence against LiDAR depth error):

| Split | conf_mean | conf_p10 | conf_low | Spread of conf_mean across frames |
|---|---|---|---|---|
| AMtown (train) | −0.25 | −0.24 | +0.23 | 0.02 |
| AMvalley (validation) | **+0.23** | **+0.22** | −0.05 | 0.05 |

- The current student's confidence head is **not calibrated across areas.** On AMtown, higher confidence means lower error; on AMvalley it means *higher* error.
- Confidence also barely varies between frames.
- So whatever the networks learn from it on AMtown is reversed on AMvalley. ErrNet's validation Spearman dropped from 0.22 to 0.10.
- **Confidence features are now off by default** (`SELECTOR_CONF=1` turns them on).

**Tuning noise:** the AMtown03 tuning grid is almost flat; every best configuration gives about 0.583. So the choice of iteration and θ is mostly noise, and run-to-run differences of about 0.02 on validation (0.341 / 0.349 / 0.361) are within that noise. A larger tuning set is needed, such as more AMtown03 clips or HKairport.

**Conclusion:** the useful per-candidate signal is depth accuracy. Today it is only available from depth geometry (DepthNet Spearman 0.67, selector F 0.402). Student confidence will only become usable after the confidence head is trained against LiDAR error on several areas, which is improvement #2 in Section 14. That would also give fusion-time pixel weighting.

---

## 11. Timeline of the conversation

1. **Connectivity:** 10.23.9.49 became available. Work moved to 10.23.20.60, which has the data and 224 CPU cores.
2. **Contribution labels:** Shapley and leave-one-out labels for 85 clips, and the DINO scorer. Result: DINO barely predicts value (Spearman 0.08).
3. **Drop test** (you said "Yes, test it"): using all frames hurts, and dropping low-contribution frames helps a lot.
4. **First notes (Δ(f|S), bins, top-K, mapping):** "check this as well". Built the exact greedy Δ(f|S) oracle (you said "Yes, build it" and "if you think it helps then for sure"). Result: a large oracle gap, +0.13..+0.18.
5. **Pick analysis** ("Yes, analyse picks"): depth accuracy explains the picks.
6. **Inference-time question:** measured about 1–2 ms/frame extra, so you said "okay then continue".
7. **"Make the pipeline very effective, implement everything":** built `reconstruction_selector.py`, the depth-aware DAgger selector with swap pass, auto-stop and leakage control. The smoke test passed and the full run started.
8. **Model switched** (Fable 5.2). **Second notes (CNN→FFN→sigmoid, bin memory, add/swap Q, no depth for selection, DA3 distillation):** read the full repo pipeline on 10.23.9.49 and the architecture document. The earlier selector used depth features and was offline, so I wrote `bin_memory_selector.py`, which follows your design exactly.
9. **RL idea:** assessed (Section 13) and not implemented, because DAgger already covers it and the oracle gap will decide.
10. **"Any improvement in the complete training pipeline":** answered with 7 items (Section 14).
11. **Full-run stall (about 22 h):** a fork deadlock. Fixed with a spawn pool.
12. **"Give complete context in one .md file":** `GEOSTREAMFUSION_SELECTOR_CONTEXT.md`.
13. **Hang (7.5 h):** a worker segfault left `Pool.map` waiting forever. Fixed with the crash-safe executor and fault logs.
14. **Segfault root cause:** the depth student's CPU forward pass in workers (native OpenMP thread). Tried oneDNN off, then single-threaded workers. Final fix: per-clip student depth cache, verified bit-exact. The smoke test is clean (0 faults).
15. **Model switched** (Opus 5.5). Full runs of both selectors launched with 56 workers each.
16. **This report.**

---

## 12. Problems hit and fixes

| Problem | Cause | Fix |
|---|---|---|
| Large bash heredoc failed | Apostrophes in the script | Write files locally, strip CR, `scp` |
| SSH exit 255 | The connection dropped; the job was fine | Jobs use `setsid nohup`; watchers use short separate SSH checks |
| `pkill -f` killed its own shell | The pattern matched the SSH command line | Bracket trick (`[r]econstruction...`) or kill by process group |
| Tables stage hung | `Pool(maxtasksperchild=1)` re-forking | Removed |
| `FileNotFoundError tables/...` in smoke test | `--limit` chose different clips per stage | `clips(None)` = union of the per-split lists |
| Evaluation explored 16 frames, tuning covered 8 | Inconsistent budgets | Cap at `LABEL_BUDGET` = 8 |
| Full run stalled for 22 h | Fork after torch use, deadlock | `spawn` context with an initializer |
| Runs hung for 7.5 h | A worker segfaulted, `Pool.map` waits forever | Crash-safe executor: retry, isolate, `failures.txt`, `faults/` traces |
| Segfaults in `DepthStudent.forward` | Native OpenMP thread crash in the CPU student forward pass inside workers | oneDNN off, single-threaded workers, `Context` no longer overrides threads, **student depth cache** (bit-exact) |
| Old `depth_error_selector` training hung for 30 h | Old fork deadlock | Stopped; superseded |
| Validation leakage in early experiments | Best epoch chosen on validation | AMtown03 holdout plus out-of-fold predictions in the new pipelines |
| Selector used depth features and was offline | Contradicted your second notes | New `bin_memory_selector.py`, depth-free and online |

---

## 13. Reinforcement learning assessment

- **Current approach:** selection is a sequential decision problem. The DAgger loop used here is already a learning-to-search form of imitation and RL. It rolls out the current policy, measures the true reconstruction gain at the states that policy visits, and retrains.
- **When full RL would be needed:** full RL (policy gradient or Monte Carlo returns) only pays off if one-step gains are short-sighted, that is, if taking a frame now changes the value of later frames a lot.
- **What the smoke test shows:** the online oracle (0.487) is close to the greedy oracle (0.521). Myopic add/swap gains are nearly sufficient.
- **Decision:** don't add RL yet. Revisit if the full run shows a large gap between the online and greedy oracles.
- **If RL is added:** the reward would be the final F of the episode, with the existing `selector_policy.sampling_distribution` and `finetune` stage used for policy-gradient fine-tuning, starting from the DAgger scorer.

---

## 14. Improvements for the complete training pipeline

In priority order:

1. **Train the selector on the fused result.** The `labels → sources → dataset → selector → finetune` stages should use the measured Δ(f|S) add and swap gains from fusion, not per-ray point error. The pilot's point error barely moved (0.0304 to 0.0294); the oracle gap in fused F is 0.13. *(Built in the new modules.)*
2. **Use the student's confidence at fusion.** Calibrate the confidence output against LiDAR error, then mask or down-weight pixels in the TSDF, and add distance weighting. About 30% of frames hurt the map, mostly through bad depth.
3. **Feed hard cases back into student training.** Use source dropout (zero or one source); first frames show about 3.5 m error against about 0.3 m for later ones. Oversample far-range and high-error frames.
4. **Alternate training rounds:** student → gain labels → selector → mine frames that hurt fusion → student again. Refuse stale labels by checking the student checksum (the new depth cache is already keyed by checkpoint).
5. **Fix model choice.** Choose on a held-out training flight, and keep one untouched test flight (for example AMvalley03). *(Built in the new modules.)*
6. **Use more flights**, starting with HKairport.
7. **Change the promotion gate in `evaluate`.** Require that fused precision, recall and covered area all hold up, that the test uses tracked rather than true poses, and that labels and evaluation run on the same device (CPU and GPU differ by about 0.005 F).

---

## 15. Current state and next steps

### Done

Both full runs have finished; see Section 10.10. Nothing is running now.

### Next steps

1. ~~Allow student confidence~~: done and tested (Section 10.11). It doesn't help, because the confidence is uncalibrated across areas.
2. **Calibrate the student's confidence head against LiDAR depth error** on several areas (student-side training on the GPU machine), then retest it as a selector input and as a fusion weight.
3. Enlarge the tuning set (HKairport, more holdout clips), because the current AMtown03 tuning grid is flat. Fall back to the middle-of-bin rule when the score is uncertain.
3. Deploy `reconstruction_selector.py`, `bin_memory_selector.py` and `run_stage.sh` to 10.23.9.49. The repo copy of `reconstruction_selector.py` there is older and should be overwritten.
4. Add per-pixel confidence filtering before fusion, and report precision, recall and coverage alongside F.
5. Integrate the selector as the global frame selector before the depth student, in `run_complete_training.sh`.
6. Test with tracked (aligned) poses, and add a streaming-safe replace (fuse a bin's frame only after the camera leaves that bin).

---

## 16. File index

**On 10.23.20.60 under `/home/vishnu/geostreamfusion_attribution_20261108/`**, also kept in the local scratchpad:

| Path | Content |
|---|---|
| `uav_reconstruction/adaptive_uav/contribution_labels.py` | Shapley and leave-one-out labels, `load_clip`, `novelty_bins` |
| `uav_reconstruction/adaptive_uav/train_contribution_selector.py` | DINO ListNet scorer, `dino_bins` |
| `uav_reconstruction/adaptive_uav/reconstruction_selector.py` | Depth-aware DAgger selector, crash-safe `parallel`, student depth cache |
| `uav_reconstruction/adaptive_uav/bin_memory_selector.py` | Depth-free online bin-memory selector (your design) |
| `drop_test.py`, `consistency_test.py`, `greedy_selector.py`, `analyse_picks.py`, `depth_error_selector.py` | Diagnostic experiments |
| `run_stage.sh` | Runner: `./run_stage.sh <module> <work_dir> <workers> [args]` |
| `runs/contribution_20261109/` | Labels, greedy oracle, drop and consistency tests, depth-error tables |
| `runs/selector_smoke/` | Smoke tests of both selectors |
| `runs/selector_20261111/` | Full runs (tables, depth, student_depth cache, online/, RESULT.md) |

**This folder:**

- `GEOSTREAMFUSION_SELECTOR_CONTEXT.md`: the earlier, shorter context file.
- `GEOSTREAMFUSION_COMPLETE_REPORT.md`: this report.
