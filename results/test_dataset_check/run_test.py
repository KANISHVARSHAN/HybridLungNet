"""Run HybridLungNet v2 on Test_dataset CTs (CT DICOM only), then compare with the readers'
DICOM-SEG/SR ratings, which are read only AFTER the prediction."""
import glob, os, sys, json, re, numpy as np, pydicom
sys.path.insert(0, r"C:\mini-project\v2")
from hln import lidc_io as L
import hln.pipeline as P, hln.models, hln.detect
for _m in (L, hln.models, hln.detect):
    for _k in dir(_m):
        if not _k.startswith("__"): setattr(P, _k, getattr(_m, _k))
c = json.load(open(r"C:\mini-project\v2\models\deploy_config.json"))
M = r"C:\mini-project\v2\models\\"
pipe = P.Pipeline(M + c["detector"], M + c["classifier"], np.array(c["aux_mean"], np.float32),
                  np.array(c["aux_std"], np.float32), c["thresholds"], device="cpu", mc_passes=c["mc_passes"])
ROOT = sys.argv[1]; OUT = os.path.dirname(os.path.abspath(__file__))
summary = {}
for pid in sorted(os.listdir(ROOT)):
    ct_files = glob.glob(f"{ROOT}/{pid}/**/CT_*/*.dcm", recursive=True)
    r = pipe.run(ct_files, max_findings=20)
    ct = L.load_ct(ct_files)
    # ---- readers' truth (read only now)
    marks = []
    for f in glob.glob(f"{ROOT}/{pid}/**/SR_*/*.dcm", recursive=True):
        sr = pydicom.dcmread(f); items = []
        def walk(seq):
            for it in seq:
                n = it.ConceptNameCodeSequence[0].CodeMeaning if "ConceptNameCodeSequence" in it else ""
                v = it.ConceptCodeSequence[0].CodeMeaning if "ConceptCodeSequence" in it else None
                if "MeasuredValueSequence" in it: v = float(it.MeasuredValueSequence[0].NumericValue)
                items.append((n, v, it))
                if "ContentSequence" in it: walk(it.ContentSequence)
        walk(sr.ContentSequence)
        mal = next((int(v[0]) for n, v, _ in items if n == "Malignancy" and v), None)
        diam = next((v for n, v, _ in items if n == "Diameter"), None)
        tex = next((v for n, v, _ in items if n == "Texture"), None)
        seg_uids = set()
        for e in list(sr.get("CurrentRequestedProcedureEvidenceSequence", [])) + list(sr.get("PertinentOtherEvidenceSequence", [])):
            for s in e.ReferencedSeriesSequence:
                for i in s.ReferencedSOPSequence: seg_uids.add(i.ReferencedSOPInstanceUID)
        marks.append(dict(mal=mal, diam=diam, tex=tex, seg_uids=seg_uids, observer=str(sr.get("SeriesInstanceUID"))))
    segs = {}
    for f in glob.glob(f"{ROOT}/{pid}/**/SEG_*/*.dcm", recursive=True):
        sg = pydicom.dcmread(f); arr = sg.pixel_array
        if arr.ndim == 2: arr = arr[None]
        pts = []
        for k, fg in enumerate(sg.PerFrameFunctionalGroupsSequence):
            ipp = [float(x) for x in fg.PlanePositionSequence[0].ImagePositionPatient]
            rr, cc = np.nonzero(arr[k])
            if len(rr): pts.append((ipp[2], rr.mean(), cc.mean(), len(rr)))
        if pts:
            w = np.array([p[3] for p in pts], float); z = np.average([p[0] for p in pts], weights=w)
            row = np.average([p[1] for p in pts], weights=w); col = np.average([p[2] for p in pts], weights=w)
            segs[sg.SOPInstanceUID] = L.world_to_1mm(ct, z, row, col)
    for m in marks:
        cs = [segs[u] for u in m["seg_uids"] if u in segs]
        m["pos"] = cs[0] if cs else None
    # group readers' marks of the same nodule (each mark comes from a different reader file set)
    groups = []
    for m in [m for m in marks if m["pos"] is not None]:
        for g in groups:
            d = np.linalg.norm(np.subtract(g["pos"], m["pos"]))
            if d <= max(0.5 * max(m["diam"] or 0, g["diam"]), 5.0):
                g["marks"].append(m); break
        else:
            groups.append(dict(pos=m["pos"], diam=m["diam"] or 0, marks=[m]))
    nods = []
    for g in groups:
        mals = [m["mal"] for m in g["marks"] if m["mal"]]
        pos = np.mean([m["pos"] for m in g["marks"]], 0); diam = float(np.mean([m["diam"] for m in g["marks"] if m["diam"]]))
        mm = float(np.mean(mals)) if mals else None
        verdict = None if mm is None else ("MALIGNANT-looking (>=3.5)" if mm >= 3.5 else "BENIGN-looking (<=2.5)" if mm <= 2.5 else "INDETERMINATE (2.5-3.5)")
        best = None
        for f in r["findings"]:
            v = f["voxel_1mm"]; d = float(np.linalg.norm(np.subtract([v["z"], v["y"], v["x"]], pos)))
            if d <= max(diam / 2, 1.5) and (best is None or d < best[0]): best = (d, f)
        nods.append(dict(readers=len(g["marks"]), ratings=mals, mean_rating=mm, readers_verdict=verdict,
                         diameter_mm=round(diam, 1), texture=[m["tex"] for m in g["marks"]],
                         location_1mm=[round(x, 1) for x in pos],
                         system=None if best is None else dict(dist_mm=round(best[0], 1), level=best[1]["detection_level"],
                                                               detection=best[1]["detection_score"], malignancy=best[1]["malignancy_prob"],
                                                               spread=best[1]["malignancy_mc_std"], slice=best[1]["dicom"]["slice_index"])))
    matched = {n["system"]["slice"] for n in nods if n["system"]}
    unmatched = [dict(level=f["detection_level"], detection=f["detection_score"], malignancy=f["malignancy_prob"], slice=f["dicom"]["slice_index"])
                 for f in r["findings"] if not any(n["system"] and n["system"]["detection"] == f["detection_score"] for n in nods)]
    summary[pid] = dict(n_ct_slices=r["n_slices"], spacing=r["spacing_mm"], seconds=round(r["timings_s"]["classify"], 1),
                        n_confident=r["n_confident"], n_possible=r["n_possible"],
                        highest_malignancy_confident=r["scan_max_malignancy_confident"],
                        reader_nodules=nods, system_findings_not_matching_any_reader_nodule=unmatched)
    print(json.dumps({pid: summary[pid]}, indent=1, default=float), flush=True)
json.dump(summary, open(os.path.join(OUT, "test_dataset_results.json"), "w"), indent=1, default=float)
