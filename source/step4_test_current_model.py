"""STEP 4 - independent test of the existing trained model (models/best_hybrid_fold2.pth).

Only fold-2 validation patients are used: they are the only patients this checkpoint
was not trained on.  Caveat measured below: the checkpoint was *selected* on these same
patients (best epoch by val AUC), so every number here is optimistically biased.
"""
import json, math, os, sys
import numpy as np, pandas as pd, torch
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.dirname(__file__))
from hln.models import HybridLungNet, mc_dropout_predict

ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "data", "orig_patches")
CKPT = r"C:\mini-project\models\best_hybrid_fold2.pth"
HIST = r"C:\mini-project\01 Lung Cancer Mini Project\mini-project-model-and training\HybridLungNet\noduledatasetve-softlabel\hybrid_history.csv"
OUT = os.path.join(ROOT, "results", "step4")
os.makedirs(OUT, exist_ok=True)
AUX = ['vol_voxels', 'bbox_vol', 'solidity', 'extent', 'mean_hu']
rng = np.random.default_rng(0)
torch.set_num_threads(8)


def find(name):
    for r, d, f in os.walk(DATA):
        if name in f: return os.path.join(r, name)
        if name in d: return os.path.join(r, name)


df = pd.read_csv(find("final_dataset.csv"))
pdir = os.path.dirname(find(df.patch_id.iloc[0] + ".npy"))

# ---- reconstruct the exact split used in the notebook (cell 10) ---------------------
df_clean, df_ambig = df[df.label != -1].copy(), df[df.label == -1].copy()
pat = df_clean.groupby('patient_id')['label'].apply(lambda x: int(x.mean() >= 0.5)).reset_index()
pat.columns = ['patient_id', 'pat_label']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = list(skf.split(pat.patient_id, pat.pat_label))[1]           # fold 2
tr_p, va_p = set(pat.patient_id.iloc[tr_idx]), set(pat.patient_id.iloc[va_idx])
train_df = pd.concat([df_clean[df_clean.patient_id.isin(tr_p)], df_ambig[df_ambig.patient_id.isin(tr_p)]])
val = df_clean[df_clean.patient_id.isin(va_p)].reset_index(drop=True)
val_amb = df_ambig[df_ambig.patient_id.isin(va_p)].reset_index(drop=True)
assert not (tr_p & va_p)
R = {"split": dict(n_train_patients=len(tr_p), n_val_patients=len(va_p), n_val_nodules=len(val),
                   n_val_malignant=int(val.label.sum()), n_val_ambiguous=len(val_amb),
                   expected_from_notebook="239 nodules, 48 malignant")}
print(R["split"])

model = HybridLungNet()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()


def load(d):
    return np.stack([np.load(os.path.join(pdir, p + ".npy")).astype(np.float32) for p in d.patch_id])


def aux_norm(d, ref):
    m, s = ref[AUX].mean(), ref[AUX].std() + 1e-6
    return ((d[AUX] - m) / s).values.astype(np.float32)


def predict(X, A, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+bs, None]), torch.from_numpy(A[i:i+bs]))).flatten().numpy())
    return np.concatenate(out)


def metrics(y, p, thr):
    yb = (p >= thr).astype(int)
    tp, fp = int(((yb == 1) & (y == 1)).sum()), int(((yb == 1) & (y == 0)).sum())
    fn, tn = int(((yb == 0) & (y == 1)).sum()), int(((yb == 0) & (y == 0)).sum())
    return dict(thr=thr, acc=(tp+tn)/len(y), sens=tp/max(tp+fn, 1), spec=tn/max(tn+fp, 1),
                ppv=tp/max(tp+fp, 1), tp=tp, fp=fp, fn=fn, tn=tn)


def boot_ci(y, p, groups, fn, n=2000):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[s])) < 2: continue
        vals.append(fn(y[s], p[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ece(y, p, bins=10):
    e = 0
    for b in range(bins):
        m = (p >= b/bins) & (p < (b+1)/bins + (b == bins-1))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


X = load(val); y = val.label.values.astype(int); g = val.patient_id.values

# ---- A. reproduce notebook: aux normalised with the VALIDATION set's own stats -------
pA = predict(X, aux_norm(val, val))
# ---- B. correct protocol: aux normalised with TRAINING stats --------------------------
pB = predict(X, aux_norm(val, train_df))
for name, p in [("A_notebook_valstats", pA), ("B_trainstats", pB)]:
    R[name] = dict(auc=roc_auc_score(y, p), auc_ci95=boot_ci(y, p, g, roc_auc_score),
                   ap=average_precision_score(y, p), brier=brier_score_loss(y, p), ece=ece(y, p),
                   at_0_5=metrics(y, p, 0.5), at_0_3=metrics(y, p, 0.3))
    print(name, json.dumps({k: R[name][k] for k in ["auc", "auc_ci95", "ap", "ece"]}))

# ---- C. CNN only (aux features zeroed = their training mean) -> how much aux matters --
pC = predict(X, np.zeros((len(val), 5), np.float32))
R["C_aux_zeroed"] = dict(auc=roc_auc_score(y, pC))

# ---- D. the deployed UI path ----------------------------------------------------------
UI_MEAN = np.array([25000.0, 65000.0, 0.65, 0.35, 0.68], np.float32)
UI_STD = np.array([18000.0, 45000.0, 0.15, 0.12, 0.08], np.float32)


def ui_features(patch):
    lab, n = ndimage.label(patch > 0.3)
    if n == 0: return np.zeros(5, np.float32)
    c = np.bincount(lab.ravel()); c[0] = 0; b = int(np.argmax(c)); comp = lab == b
    z, yy, x = np.where(comp); bb = float((np.ptp(z)+1)*(np.ptp(yy)+1)*(np.ptp(x)+1)); ext = c[b]/bb
    return np.array([c[b], bb, ext, ext, patch[comp].mean()], np.float32)


A_ui = np.stack([(ui_features(x) - UI_MEAN) / UI_STD for x in X]).astype(np.float32)
R["aux_train_vs_ui_raw_feature_means"] = dict(
    train_mean=train_df[AUX].mean().round(3).to_dict(),
    ui_assumed_mean=dict(zip(AUX, UI_MEAN.tolist())),
    ui_recomputed_on_val_mean=dict(zip(AUX, np.stack([ui_features(x) for x in X]).mean(0).round(3).tolist())))
pD_eval = predict(X, A_ui)
# UI MC dropout: model.train() on a batch of ONE, sequentially (BN running stats drift)
ui_model = HybridLungNet(); ui_model.load_state_dict(torch.load(CKPT, map_location="cpu"))
pD_ui, sD_ui = [], []
ui_model.train()
with torch.no_grad():
    for i in range(len(X)):
        xx, aa = torch.from_numpy(X[i:i+1, None]), torch.from_numpy(A_ui[i:i+1])
        ps = [torch.sigmoid(ui_model(xx, aa)).item() for _ in range(20)]
        pD_ui.append(np.mean(ps)); sD_ui.append(np.std(ps))
pD_ui = np.array(pD_ui)
T = 1.35
pD_cal = 1 / (1 + np.exp(-np.log(np.clip(pD_ui, 1e-6, 1-1e-6) / (1 - np.clip(pD_ui, 1e-6, 1-1e-6))) / T))
R["D_ui_path"] = dict(auc_ui_aux_eval_mode=roc_auc_score(y, pD_eval),
                      auc_full_ui_path=roc_auc_score(y, pD_cal),
                      at_0_5_full_ui=metrics(y, pD_cal, 0.5),
                      frac_pred_malignant=float((pD_cal >= 0.5).mean()))
print("D", json.dumps(R["D_ui_path"]))

# ---- E. MC-dropout uncertainty done correctly (dropout only) --------------------------
model.eval()
_A = aux_norm(val, train_df); _mu, _sd = [], []
for i in range(0, len(X), 16):
    a, b = mc_dropout_predict(model, torch.from_numpy(X[i:i+16, None]), torch.from_numpy(_A[i:i+16]), n=20)
    _mu.append(a.numpy()); _sd.append(b.numpy())
mu, sd = np.concatenate(_mu), np.concatenate(_sd)
err = ((mu >= 0.5).astype(int) != y).astype(int)
R["E_uncertainty"] = dict(auc_of_mc_mean=roc_auc_score(y, mu), mean_std=float(sd.mean()), max_std=float(sd.max()),
                          auroc_std_predicts_error=roc_auc_score(err, sd) if 0 < err.sum() < len(err) else None,
                          auroc_entropy_predicts_error=roc_auc_score(err, -np.abs(mu - 0.5)) if 0 < err.sum() < len(err) else None,
                          n_errors=int(err.sum()))
print("E", json.dumps(R["E_uncertainty"]))

# ---- F. ambiguous nodules (avg malignancy 2.5-3.5) of the val patients ----------------
if len(val_amb):
    pa = predict(load(val_amb), aux_norm(val_amb, train_df))
    R["F_ambiguous"] = dict(n=len(val_amb), mean_pred=float(pa.mean()), frac_ge_0_5=float((pa >= .5).mean()),
                            spearman_with_mal_avg=float(pd.Series(pa).corr(val_amb.mal_avg.reset_index(drop=True), method="spearman")))

# ---- G. robustness to off-centre patches (relevant once a detector supplies centres) --
def shift(X, d):
    out = np.zeros_like(X)
    for i, x in enumerate(X):
        v = rng.normal(size=3); v = np.round(v / np.linalg.norm(v) * d).astype(int)
        out[i] = ndimage.shift(x, v, order=0, cval=0.0)
    return out
R["G_shift_auc"] = {}
for d in [0, 2, 4, 6, 8]:
    R["G_shift_auc"][d] = roc_auc_score(y, predict(shift(X, d) if d else X, aux_norm(val, train_df)))
print("G", R["G_shift_auc"])

# ---- H. optimism from best-epoch selection on the same fold ---------------------------
h = pd.read_csv(HIST); h2 = h[h.fold == 2]
R["H_selection_bias_fold2"] = dict(max_epoch_auc=float(h2.auc.max()), epoch_of_max=int(h2.loc[h2.auc.idxmax(), "epoch"]),
                                   mean_last10=float(h2.auc.tail(10).mean()), final_epoch=float(h2.auc.iloc[-1]))
allf = h.groupby("fold").auc.agg(["max", lambda s: s.tail(10).mean()])
R["H_all_folds"] = dict(mean_of_max=float(allf["max"].mean()), mean_of_last10=float(allf.iloc[:, 1].mean()))
print("H", R["H_selection_bias_fold2"], R["H_all_folds"])

R["is_LIDC1003_in_dataset"] = bool((df.patient_id == "LIDC-IDRI-1003").any())
R["is_LIDC1003_in_fold2_val"] = "LIDC-IDRI-1003" in va_p
json.dump(R, open(os.path.join(OUT, "step4_results.json"), "w"), indent=2, default=float)
pd.DataFrame(dict(patch_id=val.patch_id, patient=g, y=y, p_trainstats=pB, p_valstats=pA, p_ui=pD_cal,
                  mc_mean=mu, mc_std=sd)).to_csv(os.path.join(OUT, "fold2_predictions.csv"), index=False)
print("saved")
