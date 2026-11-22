
# ============================================================================ CLASSIFIER CV
# Same architecture + recipe as the original notebook (HybridLungNet, soft labels, AdamW 1e-4,
# cosine, 100 epochs, batch 16, weighted sampler, same augmentation, same 5 patient folds).
# Protocol fixes (applied to EVERY arm, including the baseline):
#   * best epoch chosen on an INNER validation split of the training patients (15 %),
#     the outer fold is evaluated once.  (Notebook chose the epoch on the outer fold.)
#   * aux features normalised with TRAINING statistics (notebook: each set its own stats).
# The per-epoch outer AUC is still logged so the notebook-style number can be reproduced.
import os, sys, time, json, random, gc
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from skimage import measure

T0 = time.time()
IN, OUT = os.environ.get("KIN", "/kaggle/input"), os.environ.get("KOUT", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 100))
ARMS = os.environ.get("ARMS", "E1_full,E1_sub,E2_sub,E3_sub").split(",")
AUX = ['vol_voxels', 'bbox_vol', 'solidity', 'extent', 'mean_hu']


def find(pred):
    for r, d, f in os.walk(IN):
        x = pred(r, d, f)
        if x:
            return x


final_csv = find(lambda r, d, f: os.path.join(r, "final_dataset.csv") if "final_dataset.csv" in f else None)
df = pd.read_csv(final_csv)
if os.environ.get("SMOKE"):
    keep = df.patient_id.drop_duplicates().sample(60, random_state=0)
    df = df[df.patient_id.isin(keep)].reset_index(drop=True)
ORIG_DIR = find(lambda r, d, f: r if sum(x.endswith(".npy") for x in f) > 1000 and "patch64" not in r else None)
FIX_DIR = find(lambda r, d, f: r if r.endswith("patch64") else None)
chk_csv = find(lambda r, d, f: os.path.join(r, "orig_patch_check.csv") if "orig_patch_check.csv" in f else None)
print("orig", ORIG_DIR, "fixed", FIX_DIR)
chk = pd.read_csv(chk_csv)
fixed_ids = set(chk.patch_id)
df["has_fixed"] = df.patch_id.isin(fixed_ids)
folds = classifier_folds(df)   # identical to notebook (computed on the FULL table)


def compute_features(patch):   # identical to scripts/04_aux_features.py
    binary = (patch > 0.25).astype(np.uint8)
    try:
        regions = measure.regionprops(measure.label(binary))
        if regions:
            p = max(regions, key=lambda r: r.area)
            return [float(p.area), float(p.area_bbox), float(p.solidity), float(p.extent),
                    float(patch[binary == 1].mean()) if binary.any() else 0.0]
    except Exception:
        pass
    return [0.0] * 5


# aux features for fixed patches (recomputed on the corrected 48^3 centre crop)
fix_feat = {}
for pid in df.patch_id[df.has_fixed]:
    p = np.load(f"{FIX_DIR}/{pid}.npy").astype(np.float32)[8:56, 8:56, 8:56]
    fix_feat[pid] = compute_features(p)
print("fixed feats", len(fix_feat), time.time() - T0)


class DS(Dataset):
    def __init__(self, d, kind, aux_mean, aux_std, augment, soft, jitter=0, shift_eval=0):
        self.d, self.kind, self.aug, self.soft, self.jit, self.shift = d.reset_index(drop=True), kind, augment, soft, jitter, shift_eval
        A = np.stack([fix_feat[p] for p in self.d.patch_id]) if kind == "fixed" else self.d[AUX].values
        self.A = ((A - aux_mean) / aux_std).astype(np.float32)
        self.rng = np.random.default_rng(1)

    def __len__(self):
        return len(self.d)

    def patch(self, i):
        pid = self.d.patch_id.iloc[i]
        if self.kind == "orig":
            p = np.load(f"{ORIG_DIR}/{pid}.npy").astype(np.float32)
            if self.shift:
                v = self.rng.normal(size=3); v = np.round(v / np.linalg.norm(v) * self.shift).astype(int)
                p = np.roll(p, v, (0, 1, 2))
            return p
        p = np.load(f"{FIX_DIR}/{pid}.npy").astype(np.float32)
        o = np.array([8, 8, 8])
        if self.jit and self.aug:
            o = o + np.random.randint(-self.jit, self.jit + 1, 3)
        if self.shift:
            v = self.rng.normal(size=3); o = o + np.round(v / np.linalg.norm(v) * self.shift).astype(int)
        o = np.clip(o, 0, 16)
        return p[o[0]:o[0] + 48, o[1]:o[1] + 48, o[2]:o[2] + 48]

    def __getitem__(self, i):
        p = self.patch(i)
        if self.aug:   # identical to notebook NoduleDatasetV2.augment_patch (minus zoom: kept)
            for ax in range(3):
                if np.random.random() > 0.5:
                    p = np.flip(p, ax).copy()
            p = np.rot90(p, k=np.random.randint(0, 4), axes=(1, 2)).copy()
            p = np.clip(p + np.random.normal(0, 0.02, p.shape).astype(np.float32), 0, 1)
            if np.random.random() > 0.5:
                from scipy.ndimage import zoom as ndzoom
                z = np.random.uniform(0.85, 1.15); zz = ndzoom(p, z, order=1)
                if z > 1:
                    s = (zz.shape[0] - 48) // 2; p = zz[s:s + 48, s:s + 48, s:s + 48]
                else:
                    a = (48 - zz.shape[0]) // 2; b = 48 - zz.shape[0] - a; p = np.pad(zz, [(a, b)] * 3)
                p = p[:48, :48, :48]
            if np.random.random() > 0.5:
                p = np.clip(p + np.random.uniform(-0.1, 0.1), 0, 1)
        y = float(self.d.soft_label.iloc[i]) if self.soft else float(self.d.label.iloc[i])
        return torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))[None], torch.from_numpy(self.A[i]), torch.tensor([y])


def predict(model, ds):
    model.eval(); out = []
    with torch.no_grad():
        for x, a, _ in DataLoader(ds, batch_size=64, num_workers=int(os.environ.get("NW", 2))):
            with torch.autocast(device_type="cuda", enabled=DEV == "cuda"):
                out.append(torch.sigmoid(model(x.to(DEV), a.to(DEV)).float()).flatten().cpu().numpy())
    return np.concatenate(out)


def run_arm(arm):
    kind = "orig" if arm.startswith("E1") else "fixed"
    sub = not arm.endswith("full")
    jit = 5 if arm.startswith("E3") else 0
    D = df[df.has_fixed] if sub else df
    clean, amb = D[D.label != -1], D[D.label == -1]
    oof, hist, fold_rows = [], [], []
    for k, (trp, vap) in enumerate(folds):
        seed = 42 + k
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        trp = sorted(trp & set(D.patient_id)); vap = sorted(vap & set(D.patient_id))
        rng = np.random.default_rng(seed)
        inner = set(rng.choice(trp, size=max(1, int(round(0.15 * len(trp)))), replace=False))
        fit_p = set(trp) - inner
        tr = pd.concat([clean[clean.patient_id.isin(fit_p)], amb[amb.patient_id.isin(fit_p)]])
        iv = clean[clean.patient_id.isin(inner)]
        ov = clean[clean.patient_id.isin(vap)]
        A_tr = np.stack([fix_feat[p] for p in tr.patch_id]) if kind == "fixed" else tr[AUX].values
        am, asd = A_tr.mean(0), A_tr.std(0) + 1e-6
        cc = tr[tr.label != -1].label.value_counts().to_dict()
        w = tr.label.map(lambda l: 1.0 / cc.get(l, cc[0])).values
        dl = DataLoader(DS(tr, kind, am, asd, True, True, jit), batch_size=16, sampler=WeightedRandomSampler(w, len(w)),
                        num_workers=int(os.environ.get("NW", 4)), pin_memory=True, worker_init_fn=lambda i: np.random.seed(seed * 10 + i))
        ds_iv, ds_ov = DS(iv, kind, am, asd, False, False), DS(ov, kind, am, asd, False, False)
        torch.manual_seed(42)
        model = HybridLungNet().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
        best, best_ep = -1, 0
        for ep in range(EPOCHS):
            model.train()
            for x, a, y in dl:
                x, a, y = x.to(DEV), a.to(DEV), y.to(DEV)
                opt.zero_grad()
                with torch.autocast(device_type="cuda", enabled=DEV == "cuda"):
                    loss = F.binary_cross_entropy_with_logits(model(x, a).squeeze(1).float(), y.squeeze(1))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
            pi, po = predict(model, ds_iv), predict(model, ds_ov)
            ai = roc_auc_score(iv.label, pi) if iv.label.nunique() > 1 else np.nan
            ao = roc_auc_score(ov.label, po)
            hist.append(dict(arm=arm, fold=k + 1, epoch=ep + 1, inner_auc=ai, outer_auc=ao))
            if ai > best:
                best, best_ep = ai, ep + 1
                torch.save(model.state_dict(), f"{OUT}/{arm}_fold{k+1}.pth")
        model.load_state_dict(torch.load(f"{OUT}/{arm}_fold{k+1}.pth"))
        po = predict(model, ds_ov)
        shifted = {s: roc_auc_score(ov.label, predict(model, DS(ov, kind, am, asd, False, False, shift_eval=s))) for s in (3, 6)}
        h = pd.DataFrame([r for r in hist if r["fold"] == k + 1])
        fold_rows.append(dict(arm=arm, fold=k + 1, n_val=len(ov), n_mal=int(ov.label.sum()), best_epoch=best_ep,
                              auc=roc_auc_score(ov.label, po), ap=average_precision_score(ov.label, po),
                              auc_shift3=shifted[3], auc_shift6=shifted[6],
                              notebook_style_max_outer_auc=float(h.outer_auc.max())))
        json.dump(dict(aux_mean=am.tolist(), aux_std=asd.tolist(), kind=kind), open(f"{OUT}/{arm}_fold{k+1}_aux.json", "w"))
        oof.append(pd.DataFrame(dict(arm=arm, fold=k + 1, patch_id=ov.patch_id.values, patient_id=ov.patient_id.values,
                                     y=ov.label.values, p=po)))
        print(fold_rows[-1], f"{time.time()-T0:.0f}s", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(fold_rows), pd.concat(oof), pd.DataFrame(hist)


allf, allo, allh = [], [], []
for arm in ARMS:
    f, o, h = run_arm(arm)
    allf.append(f); allo.append(o); allh.append(h)
    pd.concat(allf).to_csv(f"{OUT}/cls_folds.csv", index=False)
    pd.concat(allo).to_csv(f"{OUT}/cls_oof.csv", index=False)
    pd.concat(allh).to_csv(f"{OUT}/cls_history.csv", index=False)

# ---------------------------------------------------------------- paired comparison
F_ = pd.concat(allf); O_ = pd.concat(allo)
summary = {}
for arm in ARMS:
    a = F_[F_.arm == arm]; o = O_[O_.arm == arm]
    summary[arm] = dict(mean_fold_auc=float(a.auc.mean()), sd_fold_auc=float(a.auc.std()),
                        pooled_oof_auc=float(roc_auc_score(o.y, o.p)), pooled_ap=float(average_precision_score(o.y, o.p)),
                        brier=float(brier_score_loss(o.y, o.p)),
                        sens_at_0_5=float(((o.p >= .5) & (o.y == 1)).sum() / max((o.y == 1).sum(), 1)),
                        spec_at_0_5=float(((o.p < .5) & (o.y == 0)).sum() / max((o.y == 0).sum(), 1)),
                        mean_auc_shift3=float(a.auc_shift3.mean()), mean_auc_shift6=float(a.auc_shift6.mean()),
                        notebook_style_mean_auc=float(a.notebook_style_max_outer_auc.mean()))
base = "E1_sub"
rng = np.random.default_rng(0)
for arm in [a for a in ARMS if a.endswith("sub") and a != base]:
    m = O_[O_.arm == base][["patch_id", "patient_id", "y", "p"]].merge(O_[O_.arm == arm][["patch_id", "p"]], on="patch_id", suffixes=("_b", "_u"))
    pats = m.patient_id.unique(); idx = {p: np.where(m.patient_id.values == p)[0] for p in pats}; diffs = []
    for _ in range(2000):
        s = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        if m.y.values[s].min() == m.y.values[s].max():
            continue
        diffs.append(roc_auc_score(m.y.values[s], m.p_u.values[s]) - roc_auc_score(m.y.values[s], m.p_b.values[s]))
    summary[f"{arm}_minus_{base}"] = dict(delta_auc=float(roc_auc_score(m.y, m.p_u) - roc_auc_score(m.y, m.p_b)),
                                          ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                          per_fold_delta=(F_[F_.arm == arm].auc.values - F_[F_.arm == base].auc.values).round(4).tolist())
json.dump(summary, open(f"{OUT}/cls_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print("total", time.time() - T0)
