import json, sys, glob, time, numpy as np
sys.path.insert(0, r"C:\mini-project\v2")
from hln.pipeline import Pipeline
from hln import lidc_io as L
import hln.pipeline as P, hln.models, hln.detect
for _m in (L, hln.models, hln.detect):
    for _k in dir(_m):
        if not _k.startswith('__'): setattr(P, _k, getattr(_m, _k))
c = json.load(open(r"C:\mini-project\v2\models\deploy_config.json"))
p = Pipeline(r"C:\mini-project\v2\models\\" + c["detector"], r"C:\mini-project\v2\models\\" + c["classifier"],
             np.array(c["aux_mean"], np.float32), np.array(c["aux_std"], np.float32), c["thresholds"], device="cpu", mc_passes=c["mc_passes"])
root = r"C:\mini-project\v2\data\local_test\HybridLungNet-Testing\LIDC-IDRI\LIDC-IDRI-1003"
dcms = [f for f in glob.glob(root + "/**/*", recursive=True) if f.endswith(".dcm")]
t = time.time(); r = p.run(dcms, max_findings=30); print("runtime", round(time.time() - t, 1))
for k in list(r):
    if k.startswith("_"): r.pop(k)
# ---- check vs XML only after prediction
ct = L.load_ct(dcms)
xml = [f for f in glob.glob(root + "/**/*.xml", recursive=True)]
anns = []
for f in xml:
    try:
        su, a = L.parse_lidc_xml(f)
        if su == ct.series_uid: anns += a
    except Exception as e: print("xml", f, e)
gts = L.cluster_annotations(anns, ct)
for g in gts:
    best = None
    for f in r["findings"]:
        v = f["voxel_1mm"]; d = np.sqrt((v["z"]-g["z"])**2 + (v["y"]-g["y"])**2 + (v["x"]-g["x"])**2)
        if d <= max(g["diam_mm"]/2, 1.5) and (best is None or d < best[0]): best = (d, f)
    g["hit"] = None if best is None else dict(dist=round(best[0],1), level=best[1]["detection_level"], det=round(best[1]["detection_score"],3), mal=round(best[1]["malignancy_prob"],3))
    print(f"GT readers={g['n_readers']} large={g['n_readers_large']} d={g['diam_mm']:.1f} mal={g['mal_mean']} -> {g['hit']}")
for f in r["findings"]:
    print(f["detection_level"], round(f["detection_score"],3), round(f["malignancy_prob"],3), "+-", round(f["malignancy_mc_std"],3), f["voxel_1mm"])
print({k: r[k] for k in r if k != "findings"})
json.dump(dict(result=r, gt=gts), open(r"C:\mini-project\v2\results\local1003\out.json", "w"), indent=1, default=float)
