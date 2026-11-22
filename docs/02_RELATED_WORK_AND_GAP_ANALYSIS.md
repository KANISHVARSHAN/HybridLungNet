# Related work (2021–2026) and how HybridLungNet v2 compares

*Researched November 2026. Numbers are quoted from the cited sources. Most papers use different data, splits and label definitions, so **a direct number-to-number comparison is only valid where noted**. Treat the gaps as indicative, not exact.*

## 1. Where we stand (our measured numbers)

| Task | Our result | How measured |
|---|---|---|
| Nodule detection (LUNA16-style, ≥3 readers, ≥3 mm) | **CPM 0.821** (95% CI 0.746–0.895); 0.809 from raw DICOM | 1 held-out split, 125 scans, LIDC patients 0001–0600 |
| Detection at about 1 FP/scan | 91.3% sensitivity at 1.11 FP/scan | same, raw DICOM, 122 patients |
| Malignancy (LIDC reader score, true centres) | AUC **0.906** (honest 5-fold), 0.956 for the fold-2 checkpoint (optimistic) | patient-level folds |
| Patient-level malignancy from the CT alone | AUC **0.805** | 91 held-out patients |
| Labels | radiologist suspicion score (1–5), **not pathology** | — |
| External validation | **none** | — |

## 2. The main existing work

### A. Nodule detection
| Work | Year | Result | Notes |
|---|---|---|---|
| **nnDetection** (DKFZ, MICCAI) | 2021 | **CPM 0.930** on LUNA16 | Self-configuring detector. Best among methods *without* a separate false-positive-reduction stage. 10-fold CV on 888 scans. [arXiv 2106.00817](https://www.alphaxiv.org/overview/2106.00817) |
| Liu et al. (with task-specific FP reduction) | 2019 | CPM 0.952 | Two-stage. Cited in the nnDetection comparison. |
| Multi-scale CNN + attention | 2025 | CPM 0.929; sensitivity 0.977 at 2 FP/scan | [PMC11986029](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11986029/) |
| **MONAI 3D RetinaNet** (NVIDIA) | 2022– | public pretrained LUNA16 model | Used as the base of later pipelines (Tri-Reader 2026). On 21 external LIDC cases without fine-tuning, sensitivity was 45.2% (different matching rule). [model card](https://huggingface.co/monai-test/lung_nodule_ct_detection), [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection), [Tri-Reader](https://www.alphaxiv.org/abs/2601.19380v1) |
| Conformal risk control for detection | 2024 | 91.35% sensitivity at 2.25 FP/scan (≥3-reader nodules) | Adds statistical guarantees on the sensitivity/FP trade-off. [arXiv 2412.20167](https://arxiv.org/html/2412.20167v2) |
| DLCS benchmark (Duke) | 2024–25 | DLCS-trained detector: sensitivity 0.72 vs 0.64 at 2 FP/scan on external NLST-3D | Shows that LUNA16-only training generalises worse. [arXiv 2405.04605](https://arxiv.org/abs/2405.04605) |

**Comparison:** our CPM of 0.82 is about **0.10 below the best reported LUNA16 results (0.93)**. Ours is one split of 125 scans with a 4.9 M-parameter single-stage model, and we trained on only about 350 scans. Our sensitivity at about 1 FP/scan (91%) is in the same range as the 2024 conformal paper (91% at 2.25 FP/scan). The CPM gap comes mostly from the low-FP end (0.125 FP/scan: 0.39–0.41).

### B. Malignancy classification on LIDC-IDRI
* Recent papers using **patient-level splits** report AUC **about 0.94** (e.g. a 2025 concept-bottleneck network, 94.35% under a "leakage-controlled patient-wise protocol").
* Reviews warn that **LIDC AUCs are not comparable across papers**. Results change with how reader scores are made binary (one external test: 0.70 with the median score vs 0.65 with the mean score), and nodule-level vs patient-level tasks differ.
* LIDC-trained models drop to **AUC ≈ 0.70** on other datasets.
* Sources: [PMC13608753](https://pmc.ncbi.nlm.nih.gov/articles/PMC13608753), [PMC13608555](https://pmc.ncbi.nlm.nih.gov/articles/PMC13608555), [medRxiv 2020.11.24](https://www.medrxiv.org/content/10.1101/2020.11.24.20237354v1.full)

**Comparison:** our honest 0.906 (5-fold) is a little below these reported 0.94s. Ours is measured with epoch selection on separate patients and bootstrap CIs; many papers do not report this. Our own selection-biased number (0.929–0.956) matches theirs, which suggests part of their advantage may be the same optimism. That is *suspected*, not proven.

### C. Pathology-grounded, clinically validated systems (the real bar)
| Work | Year | Result | What it has that we don't |
|---|---|---|---|
| **LUNA25 Challenge** (Radiology: AI) | 2025–26 | Best AI **AUC 0.78** vs 0.70 for the radiologists' mean on indeterminate screening nodules. 12% more cancers caught at matched specificity; 20% fewer false positives at matched sensitivity. | Trained on NLST with **confirmed cancer outcomes**, tested on **external European screening data**, compared against ~65–75 radiologists. [PubMed 42340186](https://pubmed.ncbi.nlm.nih.gov/42340186/), [DIAG](https://www.diagnijmegen.nl/publications/obre26/) |
| **Optellum Virtual Nodule Clinic / LCP-CNN** (FDA 510(k) K202300) | 2021 | Standalone **AUC 0.867** (0.811–0.916). Reader AUC 0.819 → 0.888 with AI. European validation AUC 0.945. | Regulatory clearance, reader study, multicentre data. Limited to solid/semi-solid nodules of 5–30 mm. [510(k)](https://510k.innolitics.com/device/K202300), [Applied Radiology](https://appliedradiology.com/articles/ai-powered-clinical-decision-support-software-for-early-lung-cancer-diagnosis-gets-fda-nod) |
| **Sybil** (MIT/MGH, J Clin Oncol) | 2023 | 1-year cancer risk AUC 0.92 (NLST), 0.86 (MGH), 0.94 (CGMH). Later cohorts: 0.94 at 1 year, 0.79 at 6 years. | Whole-scan **future cancer risk** without annotations or clinical data, validated on >27,000 scans from 3 sites. [PubMed 36634294](https://pubmed.ncbi.nlm.nih.gov/36634294/), [ecancer WCLC 2025](https://ecancer.org/en/news/27045-wclc-2025-study-validates-ai-lung-cancer-risk-model-sybil-in-predominantly-black-population-at-urban-safety-net-hospital) |
| Nodule risk stratification (Nature Medicine) | 2024 | cites an end-to-end CT model with AUC 0.944 on NLST | Large-scale data-driven management. [s41591-024-03211-3](https://www.nature.com/articles/s41591-024-03211-3) |

### D. Uncertainty, calibration, foundation models
* **MC dropout + deep ensembles** give *calibrated* detection and malignancy probabilities in a 3D CADe/CADx system (Ozdemir et al., 2019, IEEE TMI). [arXiv 1902.03233](https://arxiv.org/pdf/1902.03233)
* **Uncertainty-driven training** (MC dropout + evidential learning + temperature scaling) produces uncertainty that correlates with errors. [arXiv 2609.20905](https://arxiv.org/pdf/2609.20905)
* **Foundation / self-supervised features:**
  * FMCB (trained on >11,000 CT lesions), Models Genesis and Med3D were benchmarked across LUNA16, NLST and LUNA25. The best ResNet50-SWS scored AUC 0.71–0.90 depending on the dataset. [arXiv 2405.04605](https://arxiv.org/abs/2405.04605)
  * DINOv2-based end-to-end pipeline (Lung Nodule-SSM, 2025). [arXiv 2505.15120](https://arxiv.org/pdf/2505.15120)

## 3. What we are MISSING (ordered by importance)

1. **Pathology / outcome labels.** We predict what radiologists *think*, not whether it *is* cancer. LUNA25, NLST, Optellum and Sybil all use confirmed outcomes. **This is the biggest gap.**
2. **External validation.** Everything we measured is LIDC-IDRI patients 0001–0600. LIDC models typically drop to AUC ≈ 0.70 on other data.
3. **Detection accuracy at very low false-alarm rates:** CPM 0.82 vs 0.93. We have no second-stage false-positive reduction, no ensemble, no test-time augmentation, and little data.
4. **The classifier never saw non-nodules.** 19% of detector false alarms are scored ≥ 50% malignant, which lowers patient-level specificity to 63%. Two-stage systems train the second stage on detector candidates.
5. **Calibration and useful uncertainty.** Our ECE is 0.24 and MC-dropout spread barely flags errors (AUROC 0.63). Ensembles, temperature scaling or conformal methods are the standard fixes.
6. **Data scale.** About 350 training scans for the detector and about 1,700 nodules for the classifier. Competitors use thousands of scans (NLST > 6,000 test scans for Sybil alone).
7. **No clinical context.** Age, smoking and emphysema (Brock/PanCan model inputs) and prior scans (nodule growth) are not used.
8. **No nodule-type handling.** Solid, part-solid and ground-glass are not distinguished, and there is no Lung-RADS style output.
9. **No reader study and no regulatory path.** We cannot claim the tool helps a radiologist.
10. **Full LIDC not used.** Patients 0601–1010 were unavailable on Kaggle.

## 4. Where we have an ADVANTAGE

1. **Honest, reproducible evaluation.** Patient-level splits, decision rules fixed before results, bootstrap CIs, and selection optimism *measured* (0.929 → 0.906). Many LIDC papers do not do this, so our numbers are more trustworthy than many headline 0.94–0.99 claims.
2. **A complete system from raw DICOM to a usable viewer.** Most LIDC malignancy papers classify only at radiologist-given centres. Ours finds the nodules itself and shows them in an OHIF-style viewer, with no annotations needed.
3. **Small and fast.** 24.6 s/scan on a T4 GPU and about 1.5–2.5 min on a laptop CPU. Two models totalling 49 MB.
4. **Data bugs found and documented.** The z-spacing error affected 48% of the original patches.
5. **A shift-robust classifier candidate (E3)** keeps AUC 0.91 with 6 mm centre error, against 0.78 for the original. Detector-fed classifiers need exactly this.
6. **Transparent outputs.** Two confidence levels with validation-derived thresholds, a ± spread, a model card with measured metrics, and limitations shown in the UI.
7. **Fully open and documented.** Code, splits, thresholds and logs are all in this folder.

## 5. What we need to do to beat them

| # | Action | Beats / closes | How to test | Target |
|---|---|---|---|---|
| 1 | **Hard-negative training.** Add a "not a nodule" class to the classifier, using detector false alarms from *training* patients only. | gap 4 | Same 122 test patients, patient-level AUC and specificity | specificity 63% → ≥ 80% at same sensitivity; patient AUC > 0.85 |
| 2 | **Adopt E3 (jitter)** after a pre-registered end-to-end test. | shift robustness | Patient AUC on the same test set | > 0.805 with CI |
| 3 | **Stronger detector.** Train nnDetection or MONAI RetinaNet on *our exact split* as a baseline. Then add flip test-time augmentation, a 3–5 model ensemble, and a 2nd-stage FP reducer. | gap 3 | FROC/CPM on the same 125 test scans | CPM ≥ 0.90 |
| 4 | **More data.** Full LIDC (1,010 patients) via TCIA; LUNA16 10-fold protocol for a comparable number. | gaps 3, 6, 10 | LUNA16 10-fold CPM | ≥ 0.90 (comparable to nnDetection 0.93) |
| 5 | **Calibration.** Temperature scaling on a calibration split; deep ensemble (5 folds) instead of MC dropout; conformal prediction sets. | gap 5 | ECE, error-AUROC on held-out data | ECE < 0.05; error-AUROC > 0.75 |
| 6 | **Outcome labels.** Train/evaluate on LUNA25 (NLST-derived, confirmed outcomes) and DLCS. | gaps 1, 2 | LUNA25 test protocol | AUC ≥ 0.78 (challenge winner) |
| 7 | **External test.** Train on LIDC, test on LUNA25 / DLCS / NLST without retuning. | gap 2 | AUC/CPM on external set | report honestly, whatever it is |
| 8 | **Pretrained 3D backbone** (FMCB / Models Genesis / self-supervised on LIDC+NLST). | gap 6 | paired ΔAUC with CI | CI > 0 |
| 9 | **Clinical inputs + Lung-RADS output + nodule type.** | gaps 7, 8 | — | feature parity with Brock/Optellum |
| 10 | **Small reader study** (e.g. 3 radiologists, 100 cases, with vs without AI). | gap 9 | MRMC AUC change | positive with CI |

**Realistic headline goal:** detection CPM ≥ 0.90 and outcome-based malignancy AUC ≥ 0.78 on LUNA25, with calibrated probabilities. That would match the published state of the art on public benchmarks *with honest evaluation*.
