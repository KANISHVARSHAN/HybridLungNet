# Models

## `deployed/`: what the system uses now

| File | What it is | Size | SHA-256 |
|---|---|---|---|
| `detector.pth` | **Nodule detector.** CenterNet-style 3D U-Net (`NoduleDetector3D`, 4.94 M parameters). Input: lung-cropped CT at 1 mm, uint8/255. Output: heat map + offset + size at stride 2. | 19.8 MB | `ad5eb22c…58b87a4` |
| `classifier_hybridlungnet_original_fold2.pth` | **Malignancy classifier.** The original HybridLungNet fold-2 checkpoint (`models/best_hybrid_fold2.pth` in the old project, md5-identical). | 29.2 MB | `eed94f17…4bdae8` |
| `deploy_config.json` | Which files to load, the feature normalisation (mean/std of the 5 features, from fold-2 training patients), and the detector thresholds. | | |
| `model_card.json` | Measured metrics and limitations shown in the app. | | |

Full hashes are printed by `python -c "import hashlib;print(hashlib.sha256(open('detector.pth','rb').read()).hexdigest())"`.

### Detector thresholds (both chosen on detector VALIDATION patients only)
* `confident` = **0.4504**: the score giving 1 false alarm/scan on validation.
  * On test: **91.3% of nodules found, 1.11 false alarms/scan** (raw DICOM, 122 patients).
* `possible` = **0.2404**: the score giving 4 false alarms/scan on validation.
  * On test: 96.0% found, 3.16 false alarms/scan.

### Why the classifier is still the original
The two retrained classifiers below scored a higher mean AUC (+0.016 to +0.018), but the 95% CI of the gain included zero. The rule fixed before the experiment was "adopt only if the CI is above zero", so the original stays.

## `candidates_not_adopted/`: higher mean metrics, not proven better

Honest 5-fold cross-validation on patients 0001–0600. The checkpoint epoch was chosen on an inner split, never on the scored fold.

| Model | Mean AUC | AUC, centre 3 mm off | AUC, centre 6 mm off | Status |
|---|---|---|---|---|
| E1_sub (original recipe, same patients) | 0.894 | 0.878 | 0.782 | baseline |
| **E2_sub**: correct z-spacing patches | **0.919** | 0.871 | 0.784 | gain +0.018, CI −0.004 to +0.042, not adopted |
| **E3_sub**: correct patches + ±5 mm centre jitter | 0.916 | **0.914** | **0.911** | gain +0.016, CI −0.011 to +0.045, not adopted. **Best candidate for the next upgrade.** |

Files:
* `E2_sub_fold{1-5}.pth` and `E3_sub_fold{1-5}.pth` are the checkpoints, if the download finished; see `DOWNLOAD_STATUS.txt`.
  * If missing, get them with `kaggle kernels output messi2011/hln-cls -p <dir> --file-pattern "^E[23]_sub_fold[1-5]\.pth$"`.
* `*_aux.json` holds the feature mean/std each checkpoint must be used with. Using a different normalisation breaks the model.

**To use a candidate in the app:**
1. Copy the `.pth` into `v2/models/`.
2. In `v2/models/deploy_config.json`, set `"classifier"` to its file name and `"aux_mean"`/`"aux_std"` to the values from its `_aux.json`.
3. Restart the app.

Do this only after testing it end-to-end on the held-out test patients (see `docs/03_BUILDER_MANUAL.md`, section 11).
