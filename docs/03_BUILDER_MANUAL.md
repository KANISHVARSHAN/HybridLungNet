# HybridLungNet v2: Builder Manual

Everything a person needs to rebuild, retrain, evaluate, run and extend the system. Code paths refer to `source/` in this folder; the live copy is `C:\mini-project\v2\`.

---------------------------------------------------------------------------------------------------
## 0. The system in one picture

```
 raw CT DICOM folder (any number of series; XML NOT accepted)
        │
        ▼  hln/lidc_io.py
 load_ct ── pick CT series with most slices · sort by z · dedup · HU = pixel*slope+intercept · clip < -1024
        │
 resample_1mm ── SimpleITK linear, true slice spacing (median z-diff, NOT SliceThickness)
        │
 lung_mask ── body convex hull → air < -320 HU → ≤ 3 largest components → fill → dilate 10 mm
        │      crop to mask bbox + 4 mm
        ├────────────── to_uint8(crop)/255 ───────────────┐
        │                                                 ▼  hln/detect.py + NoduleDetector3D
        │                                    sliding window 128³, overlap 24, max-merge
        │                                    heat map (stride 2) + offset + log-size
        │                                    peaks: score ≥ thr, NMS 6 mm, top 200, inside mask
        │                                    keep score ≥ "possible" (0.2404), max 20
        ▼  normalize(crop) in [0,1] (HU -1000..400)          │ centres (z,y,x) mm
 hln/pipeline.py: 48³ patch at each centre + 5 shape features (threshold 0.25)
        │
 HybridLungNet (original fold-2) · MC dropout 20 passes (dropout on, BatchNorm frozen)
        │
        ▼
 findings: DICOM slice/row/col · detection score + level (confident ≥ 0.4504 / possible)
           malignancy p ± std · diameter estimate      →  FastAPI  →  React viewer
```

---------------------------------------------------------------------------------------------------
## 1. Environment

| Item | Version used | Notes |
|---|---|---|
| OS | Windows 11 | Kaggle runs Linux (Python 3.13, CUDA, T4 ×1 or ×2) |
| Python | 3.13.7 in `C:\mini-project\.venv` | |
| PyTorch | 2.11.0 (CPU locally; CUDA on Kaggle) | |
| numpy 2.4.3 · pandas 3.0.1 · scipy 1.17.1 · scikit-learn 1.9.1 · scikit-image 0.26.0 | | |
| SimpleITK 2.5.3 · pydicom 3.0.1 · Pillow 12.1.1 | | |
| FastAPI 0.136.0 · uvicorn 0.44.0 · python-multipart | | the app |
| Node.js 24.16 · Vite 5 · React 18 | | frontend build |
| Kaggle CLI | `C:\mini-project\.venv\Scripts\kaggle.exe` | credentials in `%USERPROFILE%\.kaggle\kaggle.json` (account `messi2011`). **Never commit or print that file.** |

Install from scratch:
```
python -m venv .venv
.venv\Scripts\pip install torch numpy pandas scipy scikit-learn scikit-image SimpleITK pydicom pillow fastapi uvicorn python-multipart kaggle requests
cd source\app\frontend && npm install && npx vite build
```

---------------------------------------------------------------------------------------------------
## 2. Data

### 2.1 Sources
| Dataset | Where | Used for |
|---|---|---|
| LIDC-IDRI DICOM + XML, patients 0001–0600 (598 patients) | Kaggle `washingtongold/lidcidri30` | everything new (detector, corrected patches, end-to-end test) |
| Original HybridLungNet patches + `final_dataset.csv` (1707 nodules, 457 patients) | Kaggle `kanishvarshan/a-raw-data-of-ct-scan` | original classifier, experiments E1 |
| Original model | private Kaggle dataset `messi2011/hln-original-model` | end-to-end kernel |
| Full LIDC-IDRI (1010 patients) | TCIA (not on Kaggle at full size) | **not used.** This limits our scope to 0001–0600. |

### 2.2 LIDC-IDRI facts you must know
* Each scan was read by up to **4 radiologists, independently (blinded)**. Each marked:
  * nodules ≥ 3 mm with a contour and a malignancy rating 1–5;
  * nodules < 3 mm with only a point and no rating;
  * non-nodules.
* **The same physical nodule appears up to 4 times** (once per reader). The marks must be grouped:
  * `cluster_annotations` links two marks if their centres are within max(0.5 × larger diameter, 5 mm);
  * it never merges two marks from the same reader.
* A patient folder may contain **several series**, including chest X-rays (CR/DX) and multiple CTs. Pick by SeriesInstanceUID (CT only). The XML names its series in `ResponseHeader/SeriesInstanceUid`.
* **Slice spacing ≠ SliceThickness in about 48% of scans.** Always use the measured z-step between `ImagePositionPatient[2]` values. This was the original project's main data bug.
* XML coordinates: `imageZposition` is in mm (world z); `xCoord`/`yCoord` are pixel column/row. Convert with `world_to_1mm`: z_mm − z_min, row × dy, col × dx.
* Padding outside the field of view is around −2000 HU; we clip to −1024.

### 2.3 Labels
* **Malignancy label = mean of the readers' ratings** (readers ≥ 3 mm only).
* Original soft labels:

  | Mean rating | Soft label | Hard label |
  |---|---|---|
  | < 1.5 | 0.05 | benign |
  | ≤ 2.5 | 0.15 | benign |
  | ≤ 3.5 | 0.5 | ambiguous (label −1): trained on, never evaluated |
  | ≤ 4.5 | 0.85 | malignant |
  | > 4.5 | 0.95 | malignant |

* **Detection ground truth (LUNA16 rule):** nodule ≥ 3 mm marked by **≥ 3 readers**. Nodules marked by 1–2 readers, and those < 3 mm, are "ignore": a detection there is neither a hit nor a false alarm.
* **Patient label** (end-to-end): 1 if any nodule has mean rating ≥ 3.5; 0 if every nodule is ≤ 2.5 or the patient has none. Otherwise excluded.
* **These are radiologist opinions, not biopsy results.** Never call them "cancer".

### 2.4 Prepared data (`kaggle/src/prep_body.py`, kernel `hln-prep`)
* Processes every CT series with ≥ 50 slices.
* Outputs:
  * `vol/{pid}_{suid[-8:]}.npz`: uint8 lung-cropped 1 mm volume, plus the bit-packed lung mask and its shape;
  * `nodules.csv`: clustered nodules in crop coordinates (z, y, x, diam_mm, n_readers, n_readers_large, mal_mean, is_small, key);
  * `series.csv`: crop offset, shape, spacing metadata;
  * `patch64/{patch_id}.npy`: correctly re-extracted 64³ float16 patches for every row of `final_dataset.csv` within 0001–0600;
  * `orig_patch_check.csv`: correlation of the original vs corrected patch (the evidence for the z bug).
* Totals: 541 scans with nodules; 3970 clusters; 796 nodules ≥ 3 mm with ≥ 3 readers.

---------------------------------------------------------------------------------------------------
## 3. Preprocessing details (`hln/lidc_io.py`)
* `load_ct(paths, series_uid=None, min_slices=20)`:
  * reads headers, groups by SeriesInstanceUID (Modality CT only) and picks the series with the most slices;
  * sorts by z, drops duplicate z values and slices with a different size;
  * computes HU = pixel × RescaleSlope + RescaleIntercept;
  * spacing = (median z-diff, PixelSpacing[0], PixelSpacing[1]).
* `resample_1mm`: SimpleITK linear interpolation; default value −1024.
* `normalize`: clip HU to [−1000, 400], scale to [0, 1]. `to_uint8`: × 255 and round. **The detector was trained on uint8 / 255, so feed it exactly that.**
* `lung_mask`: computed at 2 mm. Per slice, take the largest component > −500 HU (the body), its convex hull, eroded 2. Then:
  1. air < −320 inside the hull;
  2. keep up to 3 components ≥ 10% of the largest;
  3. fill holes;
  4. dilate 5 voxels (10 mm) so nodules touching the pleura are kept;
  5. upsample.

  Fallback: the hull, if the result has < 2000 voxels. The convex hull was added because a "drop border components" rule removed the lungs when the field of view cut the chest wall (LIDC-1003).
* `extract_patch(vol, center, size, fill)`: patch centred on the rounded centre, padded with `fill`.

---------------------------------------------------------------------------------------------------
## 4. The detector

### 4.1 Model (`hln/models.py: NoduleDetector3D`)
* 3D U-Net built from residual "DRes" blocks; widths (24, 48, 96, 160).
* Encoder down to stride 16; decoder back up to **stride 2**; 4.94 M parameters.
* Head with 5 channels:
  * heat-map logit (bias initialised to −4.6, i.e. prior 0.01);
  * 3 sub-voxel offsets;
  * log of the diameter in mm.

### 4.2 Training (`kaggle/src/det_body.py`, kernel `hln-det`, T4 GPU)
* **Samples:** random 96³ crops; 60% centred within ±30 mm of a nodule with ≥ 2 readers, 40% random inside the lung box.
* **Augmentation:** applied to image *and* coordinates *before* building targets. Random flips on 3 axes, y/x transpose, intensity × U(0.9, 1.1) + U(−0.05, 0.05).
  * Building targets first and then flipping made off-by-one regression targets. That was a real bug, and it was fixed.
* **Targets** at stride 2:
  * Gaussian heat bump with σ = max(0.8, d/8) voxels; exact 1 at floor(q); offset = q − floor(q); log(d).
  * Small or 1-reader nodules get weight 0 within radius d/4 + 2 (ignored).
* **Loss:** CenterNet focal loss (α = 2, β = 4) + 0.5 × L1 on offsets and log-size at positive voxels.
* **Optimiser:** AdamW, lr 1e-3, weight decay 1e-4, OneCycle (5% warm-up), 14 000 iterations, batch 8, AMP.
* **Validation:** every 2000 iterations, full-scan CPM on DET_VAL; the best is saved to `detector_best.pth`. Best was iteration 6000 with VAL CPM 0.825.

### 4.3 Inference (`hln/detect.py`)
* `sliding_heatmap(model, vol01, win=128, overlap=24)`: tiles the cropped volume and merges by taking the maximum.
* `extract_peaks(H, O, S, mask, thr, topk=200, nms_mm=6)`:
  * local maxima above `thr`, inside the lung mask;
  * centre = (index + offset) × 2 mm; diameter = exp(log-size);
  * greedy NMS at 6 mm.
* **Thresholds are taken from VALIDATION patients only:** confident = score at 1 FP/scan (0.4504); possible = score at 4 FP/scan (0.2404).

### 4.4 Matching rules (`match_scan`)
* A candidate is a **hit** if within max(d/2, 1.5 mm) of an unmatched ground-truth nodule. A second hit on the same nodule is dropped.
* It is **ignored** if within max(d/2, 3 mm) of an ignore nodule.
* Anything else is a **false positive**.
* FROC sensitivity is reported at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan; **CPM = their mean**.

---------------------------------------------------------------------------------------------------
## 5. The malignancy classifier

### 5.1 Model (`HybridLungNet`)
* Input: 48³ patch at 1 mm, normalised to [0, 1] (HU −1000..400), plus 5 normalised shape features.
* Stem: Conv3d(32). Then 4 residual blocks (64, 128, 256, 256), each with stride 2 and squeeze-excitation, each followed by dual (channel + spatial) attention.
* Global average pool → 256 features.
* Feature branch: Linear 5 → 64 → 32 (dropout 0.2).
* Head: concatenate (288) → 256 → dropout 0.3 → 64 → 1 logit.

### 5.2 The 5 shape features (`compute_features`, identical to the original `scripts/04_aux_features.py`)
* Threshold the patch at 0.25 and take the largest connected component.
* Features: `[area (voxels), bbox area, solidity, extent, mean intensity of the thresholded voxels]`.
* They are normalised with **the mean/std of the training patients** (stored in `deploy_config.json` → `aux_mean`/`aux_std`).
* **Using other statistics breaks the model.** The old UI used invented ones.

### 5.3 Training recipe (the original, also used for E1–E3; `kaggle/src/cls_body.py`)
* 100 epochs, AdamW lr 1e-4, weight decay 1e-4, cosine schedule, batch 16.
* Loss: BCE on soft labels. A WeightedRandomSampler balances the classes.
* Augmentation: random flips, 90° rotations in-plane, zoom U(0.85, 1.15) with re-crop, intensity jitter.
* Ambiguous nodules are trained on (target 0.5) but never evaluated.
* **Folds:** `StratifiedKFold(5, shuffle=True, random_state=42)` on *patients*, stratified by "mean label ≥ 0.5" over clean nodules. `hln/splits.py: classifier_folds` reproduces the notebook exactly (fold 2 = 77 patients, 239 nodules, 48 malignant).
* **Checkpoint selection:** v2 uses an inner 15% of the training patients (rng seed 0). The original notebook picked the best epoch on the *scored* fold, which inflates AUC by about 0.02–0.04.
* **E2:** corrected patches (`patch64`, centre-crop 48³). **E3:** E2 + random ±5-voxel centre jitter during training.

### 5.4 MC dropout, done right (`mc_dropout_predict`)
* `model.eval()` first, then set **only `nn.Dropout` modules** to train mode, and run n = 20 passes.
* Return the mean and std of the sigmoid outputs.
* **Never call `model.train()` for MC dropout.** That puts BatchNorm into batch-statistics mode; with batch size 1 it corrupts predictions (AUC 0.956 → 0.809 in the old UI).

---------------------------------------------------------------------------------------------------
## 6. Splits and leakage rules (the most important section)

| Split | Who | Used for |
|---|---|---|
| Classifier folds | 5 patient folds of `final_dataset.csv` | classifier CV; the original model = fold-2 training patients |
| **TEST** | fold-2 validation patients (in 0001–0600) **plus** a seeded 20% of the patients the classifier never used | final detector and end-to-end numbers only |
| DET_VAL | 15% of the remaining patients (seed 0) | detector checkpoint and thresholds |
| DET_TRAIN | the rest | detector training |

* `hln/splits.py: make_split` asserts that all three are disjoint. `split.json` in `results/step5_detector/` lists every patient.
* Test patients are never used for training, early stopping, thresholds or any setting.
* **Caveat:** the *original* classifier's epoch was chosen on fold-2 validation patients, who are part of TEST. Classifier numbers on TEST are therefore optimistic. Detector numbers are not affected.
* **Decision rule for upgrades** (fixed before running): adopt only if the paired patient-bootstrap 95% CI of ΔAUC vs the baseline is entirely above 0. Otherwise keep the original and say so.
* Report every metric with its CI and the exact population (how many patients/nodules, which reader rule).

---------------------------------------------------------------------------------------------------
## 7. Kaggle workflow (`source/kaggle/`)
* Kernels are single scripts. `build.sh` concatenates modules into each kernel's script:
  * `k1 prep` = lidc_io + prep_body
  * `k2 det` = lidc_io + models + splits + detect + det_body
  * `k3 cls` = models + splits + cls_body
  * `k4 e2e` = lidc_io + models + splits + detect + pipeline + e2e_body

  This is why `hln/pipeline.py` uses names it does not import. Locally, `main.py` and `run.py` inject them; see section 9.
* Each `kNN/kernel-metadata.json` sets the GPU, data sources, and `kernel_sources` (the outputs of earlier kernels appear under `/kaggle/input/<kernel-name>/`).
* Commands:
  ```
  ./build.sh
  kaggle kernels push -p k2_det
  ./waitk.sh messi2011/hln-det                         # polls until COMPLETE/ERROR
  ./fetchk.sh messi2011/hln-det out_det                # download + print log tail
  ./dl_loop.sh messi2011/hln-cls out_cls               # retry loop for big outputs
  kaggle kernels output <slug> -p <dir> --file-pattern "<regex>"   # only some files
  ```
* Measured run times (from the kernel logs): det **1.8 h** GPU; cls **3.2 h** GPU (all 4 arms × 5 folds × 100 epochs); e2e **0.9 h** GPU; prep a few hours on CPU (not timed).
* Queueing for a GPU can add hours before a kernel starts. **Kaggle kills sessions at 12 h**, and GPU time has a weekly quota.
* The API often times out from this network: retry, and download only the files you need.
* Kernel environment variables for quick tests: `ITERS`, `VAL_EVERY`, `BS`, `NW` (workers), `EPOCHS`, `ARMS`, `SMOKE` (60-patient subset).

---------------------------------------------------------------------------------------------------
## 8. Evaluation scripts and where each number comes from

| Number | Script | Output |
|---|---|---|
| Original model test (fold 2, AUC 0.956, UI path 0.809, shift, MC, calibration) | `step4_test_current_model.py` | `results/step4_original_model_test/` |
| Patch z-distortion (48%, median r 0.884) | `prep_body.py` | `results/data_prep_check/orig_patch_check.csv` |
| Detector CPM 0.821 [0.746, 0.895] | `det_body.py` | `results/step5_detector/det_test.json` |
| Classifier experiments (0.906, E2/E3, ΔAUC CIs) | `cls_body.py` | `results/classifier_experiments/cls_summary.json`, `cls_folds.csv` |
| End-to-end (CPM 0.809, 91.3% at 1.11 FP, patient AUC 0.805, FP malignancy 19%) | `e2e_body.py` | `results/step6_end_to_end/e2e_summary.json` |
| LIDC-1003 demo | `results/demo_LIDC-1003/run.py` | `out.json` |

---------------------------------------------------------------------------------------------------
## 9. The application (`source/app/`)

### 9.1 Backend: `app/backend/main.py` (FastAPI)
* On import, it adds `v2/` to `sys.path` and injects the `hln.lidc_io`, `hln.models` and `hln.detect` names into `hln.pipeline`.
* Reads `v2/models/deploy_config.json`:
  ```json
  {"detector": "detector.pth", "classifier": "original_hybrid_fold2.pth",
   "classifier_name": "...", "aux_mean": [5 floats], "aux_std": [5 floats],
   "thresholds": {"confident": 0.4504, "possible": 0.2404}, "mc_passes": 20}
  ```
  It also reads `model_card.json`: `{"metrics": [[name, value], ...], "limitations": [..]}`.
* Endpoints:

  | Method | Path | Purpose |
  |---|---|---|
  | GET | `/api/health` | model status and thresholds |
  | GET | `/api/model-card` | metrics and limitations |
  | POST | `/api/analyze` | multipart `files` (.dcm, extension-less DICOM, or .zip); ≥ 20 files; **.xml rejected (400)**. Runs the pipeline (max 20 findings) and returns series info, findings, timings and `session_id`. |
  | GET | `/api/session/{sid}/slice?axis=axial\|coronal\|sagittal&idx=&wl=&ww=` | PNG slice of the 1 mm volume with the given window; coronal/sagittal flipped so superior is up |
  | GET | `/` | serves `frontend/dist` |

* Sessions are kept in memory, at most 4; there is no database. One analysis runs at a time (lock).

### 9.2 Frontend: `app/frontend/` (React 18 + Vite)
* `src/App.jsx`:
  * **TopBar:** research-prototype badge, model status, theme toggle.
  * **LeftPanel:** drag-and-drop a folder (webkitGetAsEntry), folder/file pickers, pipeline steps with timings, series info.
  * **Viewport:** axial/coronal/sagittal; Lung, Mediastinum and Wide window presets; finding circles sized by distance to the slice (solid = confident, dashed = possible); mouse wheel or arrow keys to page; scrubber with finding ticks; slice prefetch.
  * **RightPanel:** verdict summary; finding cards with detection bar, malignancy bar with ± spread and threshold line, uncertainty level; "how to read" box; model card.
* `src/styles.css`: dark design tokens by default plus a light theme; 300 px | 1fr | 360 px grid, single column below 900 px.
* Design reference: OHIF Viewer layout. AI output shown as text + overlay + confidence bar, following RSNA reader-preference findings.
* Build: `npx vite build` creates `dist/`. Dev mode: `npx vite` (proxies `/api` to :8000).

### 9.3 Run
* `v2\app\start.bat` builds the UI if needed, starts uvicorn on 127.0.0.1:8000 and opens the browser.
* CPU time for one scan is about 1.5–2.5 min (detection about 50–100 s, MC classification about 40 s).
* To stop it, close the window or end the python process on port 8000.

---------------------------------------------------------------------------------------------------
## 10. Output fields (per finding)
| Field | Meaning |
|---|---|
| `detection_score` | detector peak height, 0–1 (not a calibrated probability) |
| `detection_level` | `confident` (≥ 0.4504; about 91% sensitivity at about 1.1 FP/scan on test) or `possible` (≥ 0.2404) |
| `malignancy_prob` | HybridLungNet mean of 20 MC passes. **Not calibrated** (ECE 0.24): use it as a ranking |
| `malignancy_mc_std` | spread over the passes; typically 0.01–0.04; weak error signal |
| `est_diameter_mm` | detector size estimate |
| `dicom.slice_index / row / col / z_mm` | location in the original series |
| `voxel_1mm` | location in the resampled 1 mm volume (uncropped) |

---------------------------------------------------------------------------------------------------
## 11. How to retrain or upgrade (recipes)

**A. Swap in a candidate classifier (E3)**
1. Copy `models/candidates_not_adopted/E3_sub_fold2.pth` into `v2/models/`.
2. In `deploy_config.json`, set `classifier` to that file name and copy `aux_mean`/`aux_std` from `E3_sub_fold2_aux.json`.
3. **Before adopting:** add `"E3_sub"` to the variants in `e2e_body.py` (it already supports this when the cls kernel is a source), run `hln-e2e`, and compare patient AUC with a paired bootstrap CI against `original`. Adopt only if the CI is above 0.
4. Update `model_card.json` with the new measured numbers.

**B. Hard-negative classifier (recommended next upgrade)**
1. Run the detector on DET_TRAIN scans only and save every candidate that is not a hit or ignore.
2. Cut 48³ patches at those candidates with label 0 ("not a nodule"). Either add them to the classifier's negatives or add a 3rd output class.
3. Retrain with the same folds and inner-split selection.
4. Evaluate on TEST with `e2e_body.py` (patient AUC, specificity, fraction of false alarms rated ≥ 0.5) and apply the CI rule.

**C. Retrain the detector**
* Edit `det_body.py` (e.g. `ITERS`, widths, augmentation), then `./build.sh`, push `k2_det`, wait, fetch.
* Thresholds come out of `det_val.json → operating_thresholds`. Put the "1" and "4" values into `deploy_config.json`.
* Keep `split.json` unchanged so results stay comparable.

**D. New data (full LIDC / LUNA25)**
* Extend `prep_body.py` for the new folder layout.
* Keep a patient-level TEST set that is never touched.
* For LUNA25, labels are outcome-based, so evaluate them separately from the LIDC reader scores.

---------------------------------------------------------------------------------------------------
## 12. Troubleshooting (all seen during this project)
| Symptom | Cause | Fix |
|---|---|---|
| `NameError: NoduleDetector3D` when using `hln.pipeline` locally | the module is written to be concatenated | inject the names as in `app/backend/main.py` lines 30–40 |
| Lung mask covers ~0% of the lung | lungs touch the image border | current convex-hull method (already in place) |
| Detector trains but offsets are wrong | augmentation applied after building targets | transform image + coordinates first (already in place) |
| MC dropout very slow or hangs on CPU | all patches in one batch × 20 passes | batches of 16 |
| Malignancy outputs shift between requests | `model.train()` for MC dropout (BatchNorm drift) | `mc_dropout_predict` (dropout only) |
| Windows: DataLoader spawn errors | multiprocessing on Windows | `num_workers=0` locally (`NW=0`) |
| Kaggle CLI "connect timeout" | flaky network | retry loops (`fetchk.sh`, `dl_loop.sh`), `--file-pattern` |
| Kernel fails at start: `cls_summary.json` None | dependent kernel not finished | the file is now optional in `e2e_body.py`; or wait for the source kernel |
| App returns 400 | < 20 DICOM files, or an XML upload | upload the whole CT series folder |
| A real nodule missing in the app but present in evaluation | finding cap was 10 | the cap is 20 (`main.py` line 100) to match evaluation |
| `UnicodeEncodeError` printing notebooks | Windows console | `set PYTHONIOENCODING=utf-8` |

---------------------------------------------------------------------------------------------------
## 13. Known problems in the ORIGINAL project (`C:\mini-project\scripts`, `ui/`, notebook)
1. `03_extract_patches.py` z index = `argmin_idx * SliceThickness` and resampling uses SliceThickness, which is wrong when spacing ≠ thickness (48% of nodules).
2. The loader walks the whole patient folder and can mix series.
3. Nodule grouping rounds centres to a grid; it can split or merge nodules.
4. Notebook: best epoch picked on the outer fold (optimism about 0.02–0.04); feature statistics computed per dataset.
5. `ui/backend/main.py`:
   * made-up feature mean/std;
   * feature threshold 0.3 and solidity = extent;
   * `model.train()` MC dropout;
   * temperature 1.35 and threshold 0.42 of unknown origin;
   * a heuristic "auto-detect" that was never evaluated.

   Together these give **AUC 0.809 instead of 0.956**.
6. `summa.txt` claims (86% accuracy, 95.8% sensitivity) hold only at threshold 0.5, with true centres, outside the UI, and with selection bias.

---------------------------------------------------------------------------------------------------
## 14. Glossary
* **AUC:** probability that a random malignant case scores higher than a random benign one (0.5 = chance).
* **CPM:** Competition Performance Metric. Mean detection sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 false positives per scan (LUNA16).
* **FROC:** sensitivity vs false positives per scan curve.
* **FP/scan:** false alarms per CT.
* **ECE:** expected calibration error; average gap between predicted probability and observed frequency.
* **MC dropout:** running the network many times with dropout on, to estimate spread.
* **Soft label:** a target between 0 and 1 that reflects reader disagreement.
* **Patient-level split:** all scans and nodules of a patient go to the same side, to prevent leakage.
* **Bootstrap CI:** resample patients with replacement 2000×; the 2.5–97.5 percentile range.
* **Hard negatives:** false alarms of the detector used as training negatives.
* **HU:** Hounsfield units (air −1000, water 0, bone +400 to +1000).
