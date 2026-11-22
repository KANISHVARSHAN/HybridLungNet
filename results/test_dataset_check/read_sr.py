import glob, os, sys, json, pydicom
root = sys.argv[1]
def walk(seq, out, depth=0):
    for it in seq:
        name = it.ConceptNameCodeSequence[0].CodeMeaning if "ConceptNameCodeSequence" in it else "?"
        val = None
        if "ConceptCodeSequence" in it: val = it.ConceptCodeSequence[0].CodeMeaning
        elif "MeasuredValueSequence" in it:
            m = it.MeasuredValueSequence[0]; val = f"{m.NumericValue} {m.MeasurementUnitsCodeSequence[0].CodeValue}"
        elif "TextValue" in it: val = it.TextValue
        elif "UID" in it: val = it.UID
        out.append((depth, name, val))
        if "ContentSequence" in it: walk(it.ContentSequence, out, depth + 1)
for p in sorted(os.listdir(root)):
    for f in sorted(glob.glob(f"{root}/{p}/**/SR_*/*.dcm", recursive=True)):
        ds = pydicom.dcmread(f); out = []; walk(ds.ContentSequence, out)
        keep = [(n, v) for d, n, v in out if any(k in n for k in ["Malignancy", "Tracking Identifier", "Diameter", "Volume", "Person Observer", "Observer", "Texture", "Finding"])]
        print(p, os.path.basename(os.path.dirname(f))[:20], keep)
