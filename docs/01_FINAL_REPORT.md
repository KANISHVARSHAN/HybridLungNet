# HybridLungNet: review, independent test, detector, end-to-end pipeline

*Every number below was measured in this work. Scripts: `v2/step4_test_current_model.py`, Kaggle kernels `hln-prep`, `hln-det`, `hln-cls`, `hln-e2e` (sources in `v2/kaggle/src`). Raw outputs: `v2/kaggle/out_*`, `v2/results/`.*

## 1. What the project is (Step 1)

* **Task:** classify a lung nodule in a CT scan as benign or malignant. The dataset is LIDC-IDRI, where up to 4 radiologists rate malignancy from 1 to 5 per nodule.
* **Data prep** (`scripts/02–06`):
  * The XML reader annotations are grouped into nodules.
  * A 48³ patch at 1 mm is cut around each nodule centre.
  * 5 morphology features are computed (volume, bbox volume, solidity, extent, mean intensity).
  * The label is the readers' average malignancy, turned into a soft label: < 1.5 → 0.05, ≤ 2.5 → 0.15, ≤ 3.5 → ambiguous (0.5), ≤ 4.5 → 0.85, otherwise 0.95.
  * The resulting dataset has 1707 nodules from 457 patients: 755 benign, 708 ambiguous, 244 malignant.
* **Model:** HybridLungNet, a 3D ResNet with squeeze-excitation and dual attention, plus a small network for the 5 features, plus Monte-Carlo dropout.
* **Training:** 5-fold cross-validation split by patient, 100 epochs, AdamW, cosine schedule, weighted sampler, augmentation.
* **Reported results:** mean AUC 0.930, best fold (fold 2) AUC 0.956. The deployed checkpoint is `best_hybrid_fold2.pth`.

## 2. Methodology review (Step 2)

**Strengths**
* Folds are split by patient, so the same patient never appears in both training and validation.
* Soft labels reflect disagreement between readers.
* The model architecture is reasonable.
* The ambiguous nodules are kept out of the evaluation sets.

**Weaknesses I measured**
1. **The z-position and z-spacing bug.** The slice position and the resampling both use `SliceThickness` instead of the actual slice spacing.
   * In 440 of the 926 nodules I could re-extract (48%), spacing ≠ thickness.
   * For those nodules, the original patches correlate with correctly extracted patches at a median of only 0.884 (10th percentile 0.509).
   * For the 486 nodules where spacing = thickness, my re-implementation matches the original exactly (median correlation 1.000). This validates the comparison.
   * 44 patches correlate below 0.5, meaning they are largely off-target.
2. **Checkpoint chosen on the test fold.** The best epoch was picked by validation AUC on the same fold it is then scored on.
   * Fold 2: best epoch AUC 0.956, final epoch 0.924, mean of the last 10 epochs 0.915.
   * Across all folds: mean of the best epochs 0.930 vs. mean of the last 10 epochs 0.881.
   * So the published 0.930 / 0.956 are optimistic.
3. **Feature normalisation leak.** The 5 features were normalised with each set's own statistics, including the validation set's. Measured impact is negligible (0.9556 → 0.9562 with training statistics).
4. **The CT loader can mix series.** It walks the whole patient folder, which can contain several CT series.
5. **The deployed UI does not match the training pipeline.**
   * The feature means/stds are invented.
   * Features are computed with a different threshold, and solidity is set equal to extent.
   * MC dropout runs `model.train()` on a batch of one. BatchNorm then uses that single-sample statistic, and its running statistics drift with every request.
   * A temperature (1.35) and threshold (0.42) of unknown origin are applied.
   * Measured effect: **the AUC drops from 0.956 to 0.809, and sensitivity at 0.5 from 95.8% to 72.9%** (Step 4, D).
6. **The model needs to be told where the nodule is.** It requires a nodule centre taken from XML or typed in by hand (see Step 3).

**Improvements tested** (each with a pre-registered rule: adopt only if the paired patient-bootstrap 95% CI of ΔAUC is above 0)
* **E2:** train on correctly re-extracted patches. Why: removes the z-distortion. Test: same folds and recipe, paired comparison against E1 on the same patients.
* **E3:** E2 plus ±5-voxel centre jitter. Why: a detector's centres are off by a few mm, and the original model degrades quickly with shift (Step 4, G). Test: AUC with 3 mm and 6 mm shifts.
* **Epoch selection on an inner validation split** instead of the outer fold. Why: removes the optimism from point 2. Applied to every variant, including the re-trained baseline E1.
**Result** (Kaggle kernel `hln-cls`, same 5 patient folds and recipe as the notebook; the checkpoint is chosen on an inner 15% of the training patients, never on the fold being scored):

| Arm | Mean fold AUC (honest) | AUC if the best outer epoch is picked (notebook style) | AUC, centre shifted 3 mm | AUC, centre shifted 6 mm |
|---|---|---|---|---|
| E1_full: original data and recipe, all 457 patients | **0.906 ± 0.021** | 0.929 | 0.855 | 0.786 |
| E1_sub: same, only patients 0001–0600 (the ones I could re-extract) | 0.894 ± 0.053 | 0.922 | 0.878 | 0.782 |
| E2_sub: corrected patches | 0.919 ± 0.032 | 0.932 | 0.871 | 0.784 |
| E3_sub: corrected patches + jitter | 0.916 ± 0.015 | 0.942 | **0.914** | **0.911** |

* **The original 0.930 is reproduced (0.929), but only when the epoch is picked on the scored fold. Chosen honestly, the same model and data give 0.906.** That ~0.02–0.03 gap is selection optimism.
* **E2 vs E1:** ΔAUC +0.018, 95% CI −0.004 to +0.042. **E3 vs E1:** +0.016, CI −0.011 to +0.045.
* **Neither CI excludes 0, so by the pre-registered rule neither upgrade is adopted. The original classifier stays deployed.** The z-fix *may* help (4/5 folds positive for E2), but this data cannot show it.
* **Secondary finding, not used for the decision:** with jitter (E3), AUC at a 6 mm centre error stays at 0.911 vs 0.782, and E3 is higher in all 5 folds (0.88–0.92 vs 0.64–0.84).
  * This matters because in the end-to-end system the centres come from the detector, not from radiologists.
  * It was not the pre-registered criterion, so I report it as a strong candidate for a follow-up study, not as a proven improvement.

## 3. Limitations of the original system (Step 3)

* **It classifies only.** It needs a nodule centre to be supplied and cannot examine a scan by itself.
* **The XML locations come from radiologists.** They exist only for the research dataset, never for a new patient, so the original system cannot be used on a new scan as built.
* Its "auto-detect" in the UI is a hand-written heuristic that was never evaluated.
* The labels are radiologists' opinions, not pathology-confirmed diagnoses.
* All training data comes from one public dataset. There is no external validation.

## 4. Independent test of the existing model (Step 4)

Script: `v2/step4_test_current_model.py`, results in `results/step4/`.

**What I did**
* Rebuilt the notebook's exact fold-2 split: 77 patients, 239 nodules, 48 malignant. This matches the notebook.
* Those patients are the only ones this checkpoint never trained on.
* I computed confidence intervals by resampling patients (bootstrap).

| Test | Result |
|---|---|
| AUC, notebook protocol | 0.956 (95% CI 0.922–0.982) |
| AUC, training-set feature statistics | 0.956 (0.921–0.981); AP 0.837 |
| At threshold 0.5 | accuracy 85.8%, sensitivity 95.8% (46/48), specificity 83.2%, PPV 59.0% |
| At threshold 0.3 | accuracy 49.8%, sensitivity 100%, specificity 37.2%, PPV 28.6% |
| Calibration | Brier 0.134, ECE 0.236 (poorly calibrated) |
| Without the 5 features (zeroed) | AUC 0.950: the features add little |
| UI feature code, eval mode | AUC 0.947 |
| **Full UI path as deployed** | **AUC 0.809; sensitivity 72.9% at 0.5** |
| MC-dropout uncertainty | mean std 0.017 (max 0.040). Std flags the model's own errors with AUROC 0.63, worse than plain distance from 0.5 (0.78) |
| Ambiguous nodules (n = 127) | mean prediction 0.43, Spearman ρ with reader score 0.21 |
| Centre shift 0 / 2 / 4 / 6 / 8 mm | AUC 0.956 / 0.931 / 0.822 / 0.726 / 0.716 |
| Optimism from epoch selection | fold 2: 0.956 at the best epoch vs. 0.915 mean of the last 10 |

**Confidence**
* The ranking ability (AUC around 0.95 with the nodule centre given) is **reproducible**.
* It is **likely optimistic by about 0.03–0.05**, because the epoch was selected on these patients.
* The "86% accuracy / 95.8% sensitivity" claims hold only at threshold 0.5, with the true centre supplied, and outside the UI.

## 5. Nodule detector (Step 5)

* **Model:** CenterNet-style 3D U-Net (4.9M parameters), trained on Kaggle (T4 GPU) on LIDC-IDRI patients 0001–0600.
* **Patient split** (disjoint, asserted in code):
  * TEST = classifier fold-2 validation patients, plus 20% of patients the classifier never used.
  * DET_VAL = 15% of the rest.
  * DET_TRAIN = the remainder.
  * The checkpoint and the operating thresholds were chosen on DET_VAL only.
* **Ground truth** follows LUNA16: nodules ≥ 3 mm marked by ≥ 3 readers. Marks by 1–2 readers and nodules < 3 mm are ignored (neither a hit nor a false alarm).

| Held-out TEST (125 scans, 153 nodules) | Value |
|---|---|
| CPM (mean sensitivity at 0.125–8 FP/scan) | **0.821** (95% CI 0.746–0.895) |
| Sensitivity at 0.125 / 0.25 / 0.5 / 1 / 2 / 4 / 8 FP/scan | 0.41 / 0.71 / 0.84 / 0.90 / 0.95 / 0.97 / 0.97 |
| At the "confident" threshold (set on VAL for 1 FP/scan) | 140/153 found (91.5%), **1.10 false alarms per scan** |
| At the "possible" threshold (set on VAL for 4 FP/scan) | 147/153 found (96.1%), 3.2 false alarms per scan |
| Counting nodules marked by any reader (258 nodules) | CPM 0.646; 73% found at 1.10 FP/scan |

VAL CPM was 0.825 (best checkpoint at iteration 6000 of 14000). TEST CPM is 0.821, so there is no sign of overfitting the threshold choice.

## 6. End-to-end pipeline (Step 6)

* **Input:** a raw DICOM folder only. XML is rejected by the app.
* **Steps:** pick the CT series with the most slices → resample to 1 mm → lung mask → sliding-window detector → peaks → 48³ patch + 5 features → original classifier with MC dropout (20 passes, dropout only, BatchNorm frozen).
* **Output per finding:** DICOM slice/row/col, detection level (confident / possible), malignancy probability ± MC std, estimated diameter.
* **The XML was used only afterwards**, to score the output (Kaggle kernel `hln-e2e`).
* **Cap:** at most 20 findings per scan above the "possible" threshold, the same in the evaluation and the app.
* **Test set:** 122 held-out patients, the detector TEST split, started from their raw DICOM files. All 122 ran without error.

| Detection from raw DICOM | Value |
|---|---|
| CPM | **0.809** (vs 0.821 from preprocessed volumes, so the full pipeline costs little) |
| "Confident" level (threshold set on validation for 1 FP/scan) | **136/149 nodules found (91.3%), 1.11 false alarms per scan** |
| "Confident" + "possible" (4 FP/scan threshold) | 143/149 found (96.0%), 3.16 false alarms per scan |
| Nodules missed even at the "possible" level | 6 (diameters 6.5–34 mm, mean reader malignancy 1.0–4.3) |
| Mean time per scan | 24.6 s on a T4 GPU (85 s on this PC's CPU for LIDC-1003) |

| Malignancy, using detector centres | Value |
|---|---|
| AUC on correctly detected, clearly labelled nodules (134; 64 malignant) | 0.980, optimistic: about half of these patients are fold-2 patients the checkpoint was selected on |
| Detector false alarms rated ≥ 0.5 malignant | **19%** (mean probability 0.39). The classifier was never trained on non-nodules |
| **Patient level** ("does this scan contain a nodule radiologists rate ≥ 3.5?"): 91 patients, 48 positive, using the highest malignancy among confident findings | **AUC 0.805; sensitivity 87.5%, specificity 62.8% at 0.5** |
| Same patients, but classifier given the XML locations (check only) | AUC 0.999 (optimistic, see above) |

**Reading the gap:** given the right location, the classifier separates patients almost perfectly. Starting from the raw CT, AUC falls to 0.805. The main causes are visible in the candidate table:
* False alarms with high malignancy scores push benign scans up (specificity 63%).
* A few malignant nodules are detected only at the "possible" level, which this patient score ignores.

**Classifier upgrade:** none adopted (Section 2). The end-to-end numbers above are for the original classifier.

## 7. Final report (Step 7)

### What changed and why
| Change | Why | Effect (measured) |
|---|---|---|
| New loader: one CT series, true slice spacing | Old loader mixed series and used thickness as spacing | 48% of original patches were z-distorted (median correlation 0.88) |
| Nodule detector (new) | The old system needed XML or a typed-in location | 91% of nodules found at 1.1 false alarms per scan, on unseen patients |
| End-to-end pipeline + new UI | The user only provides the CT | Runs on raw DICOM; patient AUC 0.805 |
| UI inference fixed: real feature statistics, real feature code, MC dropout without BatchNorm drift, no made-up temperature or threshold | The deployed UI path scored AUC 0.809 vs 0.956 for the same model | The UI now uses exactly the evaluated path |
| Classifier retrain (E2/E3) | z-fix and shift robustness | **Not adopted**: the gain (+0.016–0.018) was not significant |

### Before / after (same evaluation)
| | Before (as published / deployed) | After (this work) |
|---|---|---|
| Classifier AUC, 5-fold mean | 0.930 (epoch picked on the scored fold) | 0.906 honest estimate for the same recipe (0.929 if the scored fold is used to pick the epoch) |
| Fold-2 checkpoint, true centres | 0.956 | 0.956 (unchanged model; optimistic) |
| Classifier inside the UI | AUC 0.809 | Same path as evaluated (eval mode, correct features) |
| Can it analyse a CT by itself? | No | Yes: CPM 0.809, 91% sensitivity at 1.1 FP/scan |
| Patient-level answer from the CT alone | Not possible | AUC 0.805 (91 test patients) |

### Limitations and failures
1. **Data scope:** only LIDC-IDRI patients 0001–0600 were usable on Kaggle (the full LIDC is not available there). All results come from one public dataset and are not externally validated.
2. **Labels:** these are radiologist suspicion scores, not pathology. "Malignant" means a reader average of at least 3.5.
3. **Leakage in the original checkpoint:** its epoch was chosen on fold-2 patients, who make up part of the end-to-end test set. Classifier numbers on those patients are optimistic. Detector numbers are not affected: the detector's checkpoint and thresholds were chosen only on DET_VAL.
4. **False alarms:** about 1.1 per scan at the confident level, and 19% of them are rated ≥ 50% malignant. This is the biggest weakness for patient-level use.
5. **Misses:** 6 of 149 nodules were not found at any level. Nodules marked by only 1–2 readers are found much less often (CPM 0.646).
6. **Calibration:** ECE 0.24. The probabilities should be read as rankings, not as true risks.
7. **Uncertainty:** MC-dropout spread is tiny (mean 0.017) and flags errors poorly (AUROC 0.63). Distance from 0.5 is a better warning sign.
8. **Shift sensitivity:** the original classifier loses accuracy when the centre is off by more than a few mm (AUC 0.73 at 6 mm). The jitter-trained model fixes this in cross-validation, but it was not adopted under the pre-set rule.
9. **Single-patient UI demo:** LIDC-1003 is one example, not evidence.
10. **Not a medical device.**

### Proven vs suspected
* **Proven (measured with confidence intervals or exact reproduction):**
  * the z-distortion of the original patches;
  * the optimism from epoch selection (0.929 → 0.906);
  * the UI degradation (0.956 → 0.809);
  * detector performance on unseen patients (CPM 0.821, CI 0.746–0.895);
  * the end-to-end detection and patient-level numbers above;
  * shift sensitivity of the original model.
* **Suspected, not proven:**
  * that the z-fix improves the classifier (+0.018, CI includes 0);
  * that jitter training would improve end-to-end patient AUC (strong cross-validation hint, not tested end-to-end);
  * that training the classifier on detector false alarms would raise patient-level specificity (not tested).
* **Unknown:** performance on other hospitals' scanners and protocols.

### Recommended next steps
1. Train the classifier with the detector's false-alarm candidates as a "not a nodule" class.
2. Adopt jitter training (E3) after a pre-registered end-to-end test.
3. Calibrate probabilities on a held-out split.
4. Validate on external data (for example LUNA25 or NLST).

## UI
The layout is modelled on the OHIF Viewer, the open-source radiology viewer: a dark theme, a study/series panel, the central viewport, and a findings panel.

The way AI results are shown follows RSNA findings that radiologists prefer text + image overlay + a confidence bar. Each finding shows:
* a marker on the image;
* a detection-confidence bar;
* a malignancy bar with its ± spread;
* a plain-language uncertainty level;
* a model card with the measured numbers and limitations above.

Sources: docs.ohif.org/configuration/ui, ohif.org, devopedia.org/ohif-viewer, atlas.rsna.org (AI-output display preferences), harrison.ai (AI interface design).

Start with `C:\mini-project\v2\app\start.bat` and open http://127.0.0.1:8000.
