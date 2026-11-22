
# ============================================================================ END-TO-END EVALUATION
# Held-out TEST patients only.  Input to the pipeline = every .dcm file in the patient's raw
# folder (as a user would upload it).  XML-derived nodules are loaded only AFTER prediction,
# to score the output.
import os, json, glob, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

IN, OUT = "/kaggle/input", "/kaggle/working"
T0 = time.time()


def find(name, must=None):
    for r, d, f in os.walk(IN):
        if name in f and (must is None or must in r):
            return os.path.join(r, name)


series = pd.read_csv(find("series.csv")); nods = pd.read_csv(find("nodules.csv"))
final_df = pd.read_csv(find("final_dataset.csv"))
split = json.load(open(find("split.json")))
detv = json.load(open(find("det_val.json")))
op = detv["operating_thresholds"]
THR = {"confident": float(op["1"]), "possible": float(op["4"])}
print("thresholds from detector VALIDATION patients:", THR)
det_ckpt = find("detector_best.pth")
_cs = find("cls_summary.json"); cls_summary = json.load(open(_cs)) if _cs else None

# --- classifier variants -----------------------------------------------------------------
folds = classifier_folds(final_df)
tr2 = folds[1][0]
tdf = pd.concat([final_df[(final_df.label != -1) & final_df.patient_id.isin(tr2)],
                 final_df[(final_df.label == -1) & final_df.patient_id.isin(tr2)]])
AUX = ['vol_voxels', 'bbox_vol', 'solidity', 'extent', 'mean_hu']
variants = {"original": (find("best_hybrid_fold2.pth"), tdf[AUX].mean().values, tdf[AUX].std().values + 1e-6)}
for arm in ["E3_sub", "E2_sub"]:
    p = find(f"{arm}_fold2.pth")
    if p:
        a = json.load(open(find(f"{arm}_fold2_aux.json")))
        variants[arm] = (p, a["aux_mean"], a["aux_std"])
print("classifier variants", list(variants))

pipes = {k: Pipeline(det_ckpt, v[0], v[1], v[2], THR) for k, v in variants.items()}
# root folder of each patient
proots = {}
for r, d, f in os.walk(IN):
    b = os.path.basename(r)
    if b.startswith("LIDC-IDRI-") and len(b) == 14 and "lidcidri30" in r:
        proots[b] = r
        d[:] = []

test_pats = split["test"]
print("test patients", len(test_pats))
records, cand_rows = [], []
for i, pid in enumerate(test_pats):
    files = []
    for r, _, f in os.walk(proots[pid]):
        files += [os.path.join(r, x) for x in f if x.endswith(".dcm")]
    res = {}
    try:
        base = pipes["original"].run(files, max_findings=20, keep_patches=True)
    except Exception as e:
        records.append(dict(patient_id=pid, error=str(e))); print(pid, "ERROR", e); continue
    crop = base["_crop"]; vol_cls = normalize(crop)
    # other classifier variants on the SAME candidates
    centers = [(f["crop_center"]["z"], f["crop_center"]["y"], f["crop_center"]["x"]) for f in base["findings"]]
    other = {k: pipes[k].classify(vol_cls, centers) for k in pipes if k != "original"}
    # ---------------- scoring against XML ground truth (after prediction) ----------------
    key = f"{pid}_{base['series_uid'][-8:]}"
    srow = series[series.key == key]
    lo_prep = json.loads(srow.crop_lo.iloc[0]) if len(srow) else None
    g = nods[nods.key == key].copy()
    if lo_prep is not None:   # prep coords -> this run's crop coords (same functions => same crop)
        sh = np.array(lo_prep) - np.array(base["lung_crop_lo"])
        g["z"] += sh[0]; g["y"] += sh[1]; g["x"] += sh[2]
    gts, ign = gt_for_scan(g, 3)
    gts1, ign1 = gt_for_scan(g, 1)
    cands = [dict(z=f["crop_center"]["z"], y=f["crop_center"]["y"], x=f["crop_center"]["x"], score=f["detection_score"]) for f in base["findings"]]
    # label every candidate: TP (>=3 readers) / TP1 (any reader >=3mm) / FP
    for j, f in enumerate(base["findings"]):
        p = np.array([f["crop_center"][a] for a in "zyx"])
        lab, gmal, gi = "FP", None, -1
        for k, gg in enumerate(gts1):
            if np.linalg.norm(p - np.array([gg["z"], gg["y"], gg["x"]])) <= max(gg["diam_mm"] / 2, 1.5):
                lab, gmal, gi = ("TP3" if gg["n_readers"] >= 3 else "TP12"), gg["mal_mean"], k
                break
        if lab == "FP" and any(np.linalg.norm(p - np.array([gg["z"], gg["y"], gg["x"]])) <= max(gg["diam_mm"] / 2, 3) for gg in ign1):
            lab = "small_or_marked"
        row = dict(patient_id=pid, rank=j, label=lab, gt_mal_mean=gmal, gt_index=gi, det_score=f["detection_score"],
                   level=f["detection_level"], p_original=f["malignancy_prob"], sd_original=f["malignancy_mc_std"])
        for k in other:
            row[f"p_{k}"], row[f"sd_{k}"] = other[k][j][0], other[k][j][1]
        cand_rows.append(row)
    # oracle: classifiers at the GT locations (what the old system needed XML for)
    oracle = {}
    for k in pipes:
        oc = pipes[k].classify(vol_cls, [(gg["z"], gg["y"], gg["x"]) for gg in gts1])
        oracle[k] = [o[0] for o in oc]
    det_conf = [dict(c, score=c["score"]) for c in cands if c["score"] >= THR["confident"]]
    rec = dict(patient_id=pid, key=key, gt_matched_series=bool(len(srow)), n_gt3=len(gts), n_gt_any=len(gts1),
               gt_mal_max=max([gg["mal_mean"] for gg in gts1 if gg["mal_mean"] == gg["mal_mean"] and gg["mal_mean"] is not None], default=None),
               gt_list=[dict(z=gg["z"], y=gg["y"], x=gg["x"], d=gg["diam_mm"], mal=gg["mal_mean"], readers=gg["n_readers"]) for gg in gts1],
               n_confident=base["n_confident"], n_possible=base["n_possible"],
               match_conf=match_scan(det_conf, gts, ign), match_all=match_scan(cands, gts, ign),
               oracle=oracle, seconds=sum(v for k, v in base["timings_s"].items() if k == "classify"),
               timings=base["timings_s"], from_fold2=pid in split["test_from_fold2"])
    for k in pipes:
        ps = [r[f"p_{k}"] for r in cand_rows if r["patient_id"] == pid and r["level"] == "confident"]
        rec[f"scan_score_{k}"] = max(ps) if ps else 0.0
    records.append(rec)
    # save a few example overlays
    if i < 12:
        np.savez_compressed(f"{OUT}/example_{pid}.npz", crop=to_uint8(crop),
                            findings=json.dumps([{k2: v2 for k2, v2 in f.items() if not k2.startswith("_")} for f in base["findings"]]),
                            gt=json.dumps(rec["gt_list"]))
    print(i, pid, "gt3", len(gts), "conf", base["n_confident"], "poss", base["n_possible"], base["timings_s"], f"{time.time()-T0:.0f}s", flush=True)

json.dump(records, open(f"{OUT}/e2e_records.json", "w"), default=float)
C = pd.DataFrame(cand_rows); C.to_csv(f"{OUT}/e2e_candidates.csv", index=False)
R = [r for r in records if "error" not in r]

# ---------------------------------------------------------------- detection (from raw DICOM)
summ = {"n_test_patients": len(test_pats), "n_errors": len(records) - len(R), "thresholds": THR}
for lvl, key in [("confident(<=1FP/scan on val)", "match_conf"), ("possible(<=4FP/scan on val)", "match_all")]:
    summ[lvl] = at_threshold([r[key] for r in R], [r["n_gt3"] for r in R], 0)
summ["froc_from_raw_dicom"] = froc([r["match_all"] for r in R], [r["n_gt3"] for r in R])

# ---------------------------------------------------------------- malignancy on detected TPs
def auc_safe(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


tp = C[C.label.isin(["TP3", "TP12"]) & C.gt_mal_mean.notna()]
tp_clear = tp[(tp.gt_mal_mean >= 3.5) | (tp.gt_mal_mean <= 2.5)]
ytp = (tp_clear.gt_mal_mean >= 3.5).astype(int)
for k in pipes:
    summ[f"tp_malignancy_auc_{k}"] = auc_safe(ytp, tp_clear[f"p_{k}"])
summ["n_tp_clear"] = int(len(tp_clear)); summ["n_tp_clear_malignant"] = int(ytp.sum())
fp = C[C.label == "FP"]
for k in pipes:
    summ[f"FP_candidates_pred_malignant_frac_{k}"] = float((fp[f"p_{k}"] >= 0.5).mean()) if len(fp) else None
    summ[f"FP_candidates_mean_p_{k}"] = float(fp[f"p_{k}"].mean()) if len(fp) else None

# ---------------------------------------------------------------- patient level
pl = []
for r in R:
    m = r["gt_mal_max"]
    y = None if m is None else (1 if m >= 3.5 else (0 if m <= 2.5 else None))
    if r["n_gt_any"] == 0:
        y = 0
    pl.append(dict(patient_id=r["patient_id"], y=y, **{f"s_{k}": r[f"scan_score_{k}"] for k in pipes},
                   **{f"o_{k}": max(r["oracle"][k], default=0.0) for k in pipes}))
P = pd.DataFrame(pl); Pc = P[P.y.notna()]
summ["patient_level_n"] = int(len(Pc)); summ["patient_level_n_malignant"] = int(Pc.y.sum())
for k in pipes:
    summ[f"patient_auc_pipeline_{k}"] = auc_safe(Pc.y.astype(int), Pc[f"s_{k}"])
    summ[f"patient_auc_with_xml_locations_{k}"] = auc_safe(Pc.y.astype(int), Pc[f"o_{k}"])
    yb = (Pc[f"s_{k}"] >= 0.5).astype(int)
    summ[f"patient_sens_at_0.5_{k}"] = float(((yb == 1) & (Pc.y == 1)).sum() / max((Pc.y == 1).sum(), 1))
    summ[f"patient_spec_at_0.5_{k}"] = float(((yb == 0) & (Pc.y == 0)).sum() / max((Pc.y == 0).sum(), 1))
P.to_csv(f"{OUT}/e2e_patients.csv", index=False)

# missed nodules
missed = []
for r in R:
    hit = {gi for s, t, gi in r["match_all"] if t}
    gts3 = [g for g in r["gt_list"] if g["readers"] >= 3]
    for k, g in enumerate(gts3):
        if k not in hit:
            missed.append(dict(patient_id=r["patient_id"], **g))
pd.DataFrame(missed).to_csv(f"{OUT}/e2e_missed.csv", index=False)
summ["n_missed_gt3_at_possible"] = len(missed)
summ["mean_seconds_total_per_scan"] = float(np.mean([r["timings"].get("classify", 0) for r in R]))
json.dump(summ, open(f"{OUT}/e2e_summary.json", "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
