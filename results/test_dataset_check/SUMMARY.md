# Test_dataset check (C:\mini-project\Test_dataset\lidc_idri)

4 LIDC-IDRI patients. Each has a CT plus the readers' DICOM-SEG (outline) and DICOM-SR (ratings) files. There is no XML.
* The system received **only the CT series**.
* Reader ratings were read afterwards, only to check (`run_test.py`, `test_dataset_results.json`).
* LIDC-IDRI-0001 was in the detector's TRAINING set, so it is **not a fair test**. 1009, 1010 and 1011 were never used.

| Patient | Readers' nodules (mean malignancy rating, 1–5) | System result |
|---|---|---|
| 0001 (seen in training) | 33 mm, 4 readers, 4.75 → malignant-looking | found, confident; malignancy 0.90 ✔ |
| 1009 | 37 mm, **1 reader only**, 2.0 → benign-looking | **MISSED**: the mass touches the mediastinum and the lung mask cuts it out. 1 confident + 5 possible findings elsewhere (false alarms) |
| 1010 | 8 mm, 3 readers, 2.33 → benign-looking | found, confident; malignancy 0.37 (low) ✔. 9 "possible" false alarms |
| 1011 | 29 mm, 4 readers, 3.5 → malignant-looking | found, confident; 0.68 ✔ |
| | 33 mm, 4 readers, 4.5 → malignant-looking | found, confident; 0.87 ✔ (+1 duplicate finding inside the same nodule) |
| | 23 mm, 4 readers, 4.25 → malignant-looking | found, confident; 0.82 ✔ |
| | 6.5 mm, 1 reader, 2.0 → benign-looking | not found (1-reader nodule; not counted in our evaluation rule) |

**Unseen patients (1009–1011):**
* Of the 5 nodules marked by ≥ 3 readers (the evaluation rule), 4 exist in these patients and **all 4 were found** at the confident level.
* Malignancy ranking agrees with the readers: the three malignant-looking nodules scored 0.68–0.87; the benign-looking one scored 0.37.
* Failure: a 37 mm mass touching the mediastinum (1009) was missed because the lung mask excludes it. Large masses at the mediastinum are a known weakness. Possible fix (not yet tested): accept peaks near the mask edge, or dilate the mask more toward the mediastinum.
* Only 3 patients, so this is an example, not statistics.

## Official diagnosis (independent of the system)
Source: TCIA LIDC-IDRI diagnosis spreadsheet `tcia-diagnosis-data-2012-04-20.xls`, which covers 157 patients.

| Patient | Official diagnosis | Readers' ratings |
|---|---|---|
| LIDC-IDRI-0001 | not in the diagnosis list | malignant-looking (4.75) |
| LIDC-IDRI-1009 | not in the diagnosis list | benign-looking (2.0, 1 reader) |
| LIDC-IDRI-1010 | **Unknown** (the patient's primary-cancer note says "lymphoma"; the nodule diagnosis is unknown) | benign-looking (2.33) |
| LIDC-IDRI-1011 | **MALIGNANT, metastatic**, confirmed by **biopsy** (primary: small-cell carcinoma of the tongue). Nodule 1 metastatic (biopsy), nodule 2 metastatic (progression/response). | malignant-looking (3.5 / 4.5 / 4.25) |

The system scored 1011's nodules 0.68, 0.87 and 0.82, consistent with the confirmed malignancy. The spreadsheet's nodule numbers are not guaranteed to match our nodule order.
