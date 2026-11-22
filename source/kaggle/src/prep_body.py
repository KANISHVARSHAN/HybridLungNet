
# ============================================================================ PREP KERNEL
# For every CT series of LIDC-IDRI-0001..0600:
#   DICOM -> 1 mm volume -> lung mask -> crop -> uint8 npz  (detector training cache)
#   XML   -> reader annotations clustered into nodules      (labels; never used at inference)
# For every row of the ORIGINAL classifier dataset (final_dataset.csv) in these patients:
#   re-extract a 64^3 patch at the CORRECTED centre and measure the original error.
import glob, json, sys, time, traceback
import pandas as pd
from multiprocessing import Pool

IN = "/kaggle/input"
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/vol", exist_ok=True)
os.makedirs(f"{OUT}/patch64", exist_ok=True)


def find_dir(pred):
    for r, d, f in os.walk(IN):
        if pred(r, d, f):
            return r


orig_csv = find_dir(lambda r, d, f: "final_dataset.csv" in f) + "/final_dataset.csv"
orig = pd.read_csv(orig_csv)
orig_pdir = find_dir(lambda r, d, f: sum(x.endswith(".npy") for x in f) > 500)
print("orig rows", len(orig), "orig patches", orig_pdir)

# ---- enumerate CT series folders (contain .dcm + .xml) ----------------------------------
series_dirs = []
for r, d, f in os.walk(IN):
    if "lidcidri30" not in r:
        continue
    dcm = [x for x in f if x.endswith(".dcm")]
    if len(dcm) >= 50:
        pid = [p for p in r.split("/") if p.startswith("LIDC-IDRI-") and len(p) == 14]
        if pid:
            series_dirs.append((pid[0], r, [x for x in f if x.endswith(".xml")]))
print("candidate series dirs", len(series_dirs))
ct_count = pd.Series([p for p, _, _ in series_dirs]).value_counts()


def process(item):
    pid, sdir, xmls = item
    t0 = time.time()
    try:
        ct = load_ct(sdir, min_slices=50)
        v = resample_1mm(ct)
        m = lung_mask(v)
        lo, hi = bbox(m, margin=4)
        crop = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        mcrop = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        key = f"{pid}_{ct.series_uid[-8:]}"
        np.savez_compressed(f"{OUT}/vol/{key}.npz", vol=to_uint8(crop), mask=np.packbits(mcrop), mshape=np.array(mcrop.shape))
        nods = []
        for xf in xmls:
            suid, anns = parse_lidc_xml(os.path.join(sdir, xf))
            if suid is not None and suid != ct.series_uid:
                continue
            for k, n in enumerate(cluster_annotations(anns, ct)):
                n.update(key=key, patient_id=pid, z=n["z"] - lo[0], y=n["y"] - lo[1], x=n["x"] - lo[2])
                n["malignancy_scores"] = json.dumps(n["malignancy_scores"])
                nods.append(n)
        # ---- re-extract the original classifier rows of this series --------------------
        checks = []
        vnorm = normalize(v)
        rows = orig[(orig.patient_id == pid) & (orig.series_uid == ct.series_uid)]
        for _, r in rows.iterrows():
            zc, yc, xc = world_to_1mm(ct, r.cz_mm, r.cy_pixel, r.cx_pixel)
            # original (buggy) conversion: nearest-slice index * SliceThickness tag
            idx = int(np.argmin(np.abs(ct.z_positions - r.cz_mm)))
            z_old = idx * ct.meta["slice_thickness_tag"]
            p64 = extract_patch(vnorm, (zc, yc, xc), 64).astype(np.float16)
            np.save(f"{OUT}/patch64/{r.patch_id}.npy", p64)
            try:
                op = np.load(f"{orig_pdir}/{r.patch_id}.npy").astype(np.float32)
                new48 = extract_patch(vnorm, (zc, yc, xc), 48)
                old48 = extract_patch(vnorm, (z_old, yc, xc), 48)
                c_new = float(np.corrcoef(op.ravel(), new48.ravel())[0, 1])
                c_old = float(np.corrcoef(op.ravel(), old48.ravel())[0, 1])
            except Exception:
                c_new = c_old = np.nan
            # does the original centre land in the nodule? fraction of soft-tissue voxels in 5mm core
            core = lambda P: float((P[19:29, 19:29, 19:29] > 0.6).mean())
            checks.append(dict(patch_id=r.patch_id, patient_id=pid, key=key, z_err_mm=float(z_old - zc),
                               corr_orig_vs_fixed=c_new, corr_orig_vs_oldmethod=c_old,
                               core_tissue_orig=core(op) if not np.isnan(c_new) else np.nan,
                               core_tissue_fixed=core(extract_patch(vnorm, (zc, yc, xc), 48)),
                               z_fixed=zc - lo[0], y_fixed=yc - lo[1], x_fixed=xc - lo[2]))
        info = dict(key=key, patient_id=pid, series_uid=ct.series_uid, shape=list(crop.shape), crop_lo=lo,
                    n_ct_series_patient=int(ct_count[pid]), seconds=time.time() - t0, **ct.meta)
        return info, nods, checks, None
    except Exception as e:
        return None, [], [], f"{pid} {sdir}: {e}\n{traceback.format_exc()[-500:]}"


if __name__ == "__main__":
    infos, nods, checks, errs = [], [], [], []
    with Pool(4) as pool:
        for i, (info, n, c, e) in enumerate(pool.imap_unordered(process, series_dirs, chunksize=1)):
            if info: infos.append(info)
            nods += n; checks += c
            if e: errs.append(e)
            if i % 25 == 0:
                print(i, len(series_dirs), "ok", len(infos), "err", len(errs), flush=True)
    pd.DataFrame(infos).to_csv(f"{OUT}/series.csv", index=False)
    pd.DataFrame(nods).to_csv(f"{OUT}/nodules.csv", index=False)
    pd.DataFrame(checks).to_csv(f"{OUT}/orig_patch_check.csv", index=False)
    open(f"{OUT}/errors.txt", "w").write("\n".join(errs))
    print("DONE series", len(infos), "nodules", len(nods), "orig rows re-extracted", len(checks), "errors", len(errs))
    if checks:
        c = pd.DataFrame(checks)
        print(c[["z_err_mm", "corr_orig_vs_fixed", "corr_orig_vs_oldmethod", "core_tissue_orig", "core_tissue_fixed"]].describe())
        print("rows with |z error| > 2mm:", int((c.z_err_mm.abs() > 2).sum()), "/", len(c))
    for e in errs[:10]: print(e)
