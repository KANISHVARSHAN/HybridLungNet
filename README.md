# HybridLungNet: Reference Folder

> **Research prototype, not a medical device.** Do not use it for patient care.

**Quick start (Windows):**
1. `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`
2. Install Node.js.
3. Run `source\app\start.bat` and open http://127.0.0.1:8000.

Model weights (`*.pth`) are stored with **Git LFS**: run `git lfs install` before cloning.

Snapshot of the HybridLungNet v2 system. It holds the models in use, the higher-scoring candidate models, every measured result, the full source code, and the manuals. Built November 2026.

**What the system does:** you give it a chest CT (DICOM) and nothing else. It finds lung nodules and, for each one, gives:
* its location;
* a detection confidence;
* a malignancy-suspicion score with an uncertainty spread.

## Start here
| Read | For |
|---|---|
| `docs/01_FINAL_REPORT.md` | What was done, all measured results, before/after, limitations, what is proven vs suspected |
| `docs/02_RELATED_WORK_AND_GAP_ANALYSIS.md` | Comparison with published work 2021–2026: what we miss, where we are ahead, the plan to beat them |
| `docs/03_BUILDER_MANUAL.md` | Everything needed to rebuild, retrain, evaluate and extend the system |
| `docs/04_USER_MANUAL.md` | How to use the app and read its output |
| `models/MODELS_README.md` | Which model is which, and how to swap them |

## Folder map
```
models/
  deployed/                  ← used by the system now
    detector.pth                               nodule detector (3D U-Net, 4.9M params)
    classifier_hybridlungnet_original_fold2.pth malignancy classifier (original)
    deploy_config.json, model_card.json
  candidates_not_adopted/    ← higher mean AUC, gain not statistically proven
    E2_sub_fold1-5.pth (+ _aux.json)           corrected-z patches
    E3_sub_fold1-5.pth (+ _aux.json)           corrected + jitter: robust to off-centre (best next candidate)
results/
  step4_original_model_test/   independent test of the original model
  step5_detector/              detector validation/test, split.json, all test candidates
  classifier_experiments/      E1/E2/E3 cross-validation results
  step6_end_to_end/            raw-DICOM pipeline on 122 held-out patients
  data_prep_check/             evidence of the original z-spacing bug
  demo_LIDC-1003/              one unseen patient, checked against XML afterwards
  test_dataset_check/          your Test_dataset (4 patients), checked against reader SEG/SR afterwards
source/                        full code snapshot (hln/, kaggle/, app/, step4 script)
docs/                          report and manuals
```

## Key numbers (all measured on patients never used for training or tuning)
| | Value |
|---|---|
| Nodule detection, CPM (LUNA16 metric) | 0.821 (95% CI 0.746–0.895); 0.809 from raw DICOM |
| Nodules found at "confident" level | 91.3% at 1.11 false alarms per scan |
| Nodules found incl. "possible" level | 96.0% at 3.16 false alarms per scan |
| Malignancy AUC, true nodule centres (honest 5-fold CV of the original recipe) | 0.906 |
| Patient-level malignancy AUC from the CT alone | 0.805 (91 patients) |
| Time per scan | 24.6 s (Kaggle T4 GPU), about 1.5–2.5 min (laptop CPU) |

**Run the app:** `source\app\start.bat` → http://127.0.0.1:8000

**Main limitations:**
* Labels are radiologists' suspicion scores, not biopsy results.
* Only LIDC-IDRI patients 0001–0600 were used.
* There is no external validation.
* The probabilities are not calibrated.
* About 1 in 5 false alarms is scored as suspicious.

## Data & acknowledgements
* Models were trained and evaluated on **LIDC-IDRI** (Armato et al., Med. Phys. 2011), obtained from **The Cancer Imaging Archive (TCIA)** and licensed CC BY 3.0. No CT images are included in this repository.
* Download the scans from TCIA to reproduce the results.
* The patient-level diagnoses in `results/test_dataset_check` come from TCIA's `tcia-diagnosis-data-2012-04-20.xls`.
* In this repo, code paths that the docs give as `C:\mini-project\v2\...` live in `source/`.
