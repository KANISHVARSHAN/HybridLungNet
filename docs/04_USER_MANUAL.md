# HybridLungNet v2: User Manual (the app)

> Research prototype. Not a medical device. Do not use it for patient care.

## Start and stop
1. Double-click `C:\mini-project\v2\app\start.bat`.
   * The first run builds the interface (needs Node.js). Later runs start in a few seconds.
   * Your browser opens at **http://127.0.0.1:8000**.
2. To stop, close the black server window (or press Ctrl+C in it).

## Analyse a scan
1. **Drop a CT study folder** onto the left panel, or click **Choose folder** or **Files**.
   * A whole patient folder is fine: the app uses the CT series with the most slices.
   * Accepted: `.dcm` files, DICOM files without an extension, or a `.zip`.
   * **Not accepted:** XML annotation files. The system finds nodules by itself.
   * At least 20 slices are needed.
2. Click **Analyze scan**. It takes about 1.5–2.5 minutes on a laptop CPU. The pipeline steps show their times when finished.
3. The **right panel** lists the findings, most confident first. The **viewer** jumps to them when you click a card.

## Reading the results
* **Confident** (solid circle): the detector is fairly sure this is a nodule.
  * On test patients it found 91% of nodules at this level, with about 1 false alarm per scan.
* **Possible** (dashed circle): weaker candidates.
  * Including these raises sensitivity to 96%, but adds about 2 more false alarms per scan.
* **Malignancy bar:** how much the nodule looks like those radiologists rated suspicious (3.5 or more out of 5).
  * It is **not** a cancer probability, and the numbers are not calibrated. Compare findings with each other rather than reading the value literally.
  * The thin range around it is the model's own spread (usually small). A large spread means "less sure"; a small spread does not guarantee it is right.
* **About 1 in 5 false alarms gets a malignancy score above 50%.** Always look at the image.

## Viewer controls
| Action | How |
|---|---|
| Change slice | mouse wheel, ↑/↓ keys, or drag the scrubber (ticks mark findings) |
| Change view | Axial / Coronal / Sagittal buttons |
| Window | Lung, Mediastinum, Wide presets |
| Go to a finding | click its card |
| Light/dark theme | button in the top bar |

## What it cannot do
* Diagnose cancer.
* Handle non-chest CT.
* Guarantee anything on scanners or hospitals outside the LIDC-IDRI dataset.
* Find every nodule: about 4–9% are missed, and small or faint ones more often.
* Use prior scans or clinical history.
