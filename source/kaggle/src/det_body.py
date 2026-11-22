
# ============================================================================ DETECTOR TRAINING
import glob, json, math, os, random, sys, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

T0 = time.time()
IN, OUT, TMP = os.environ.get("KIN", "/kaggle/input"), os.environ.get("KOUT", "/kaggle/working"), os.environ.get("KTMP", "/tmp/vol")
os.makedirs(TMP, exist_ok=True)
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ITERS = int(os.environ.get("ITERS", 14000))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 2000))
BS, CROP = int(os.environ.get("BS", 8)), 96


def find_file(name):
    for r, d, f in os.walk(IN):
        if name in f:
            return os.path.join(r, name)


series = pd.read_csv(find_file("series.csv"))
nods = pd.read_csv(find_file("nodules.csv"))
final_df = pd.read_csv(find_file("final_dataset.csv"))
VOLDIR = os.path.dirname(find_file(series.key.iloc[0] + ".npz"))
split = make_split(final_df, series.patient_id.unique())
json.dump(split, open(f"{OUT}/split.json", "w"), indent=1)
print({k: len(v) for k, v in split.items()})


def scans_of(pats):
    return series[series.patient_id.isin(pats)].key.tolist()


tr_keys, va_keys, te_keys = scans_of(split["det_train"]), scans_of(split["det_val"]), scans_of(split["test"])
print("scans train/val/test", len(tr_keys), len(va_keys), len(te_keys))


def unpack(key):
    p = f"{TMP}/{key}.npy"
    if not os.path.exists(p):
        z = np.load(f"{VOLDIR}/{key}.npz")
        np.save(p, z["vol"])
        np.save(f"{TMP}/{key}_m.npy", np.unpackbits(z["mask"])[: int(np.prod(z["mshape"]))].reshape(z["mshape"]).astype(bool))
    return p


for k in tr_keys + va_keys:
    unpack(k)
print("unpacked", time.time() - T0)

# ---------------------------------------------------------------- training data
POS_MIN_READERS = 2
nod_by_key = {k: g for k, g in nods.groupby("key")}


class CropDS(Dataset):
    def __init__(self, keys, n):
        self.keys, self.n = keys, n
        self.pos = [(k, r) for k in keys if k in nod_by_key for _, r in nod_by_key[k].iterrows()
                    if (not r.is_small) and r.n_readers_large >= POS_MIN_READERS]
        print("positive nodules for training", len(self.pos))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random
        if rng.rand() < 0.6:
            key, r = self.pos[rng.randint(len(self.pos))]
            c = np.array([r.z, r.y, r.x]) + rng.uniform(-30, 30, 3)
        else:
            key = self.keys[rng.randint(len(self.keys))]
            m = np.load(f"{TMP}/{key}_m.npy", mmap_mode="r")
            for _ in range(20):
                c = np.array([rng.randint(s) for s in m.shape], float)
                if m[int(c[0]), int(c[1]), int(c[2])]:
                    break
        vol = np.load(f"{TMP}/{key}.npy", mmap_mode="r")
        lo = np.round(c).astype(int) - CROP // 2
        x = extract_patch(vol, lo + CROP // 2, CROP, fill=0).astype(np.float32) / 255.0
        # nodules in crop coordinates (1 mm)
        items = []
        g = nod_by_key.get(key)
        if g is not None:
            for _, n in g.iterrows():
                p = np.array([n.z, n.y, n.x]) - lo
                if np.any(p < -8) or np.any(p >= CROP + 8):
                    continue
                ign = bool(n.is_small or n.n_readers_large < POS_MIN_READERS)
                items.append((p, max(float(n.diam_mm), 3.0), ign))
        # augmentation applied to image AND coordinates, before building targets
        for ax in range(3):
            if rng.rand() < 0.5:
                x = np.flip(x, ax)
                items = [(np.where(np.arange(3) == ax, CROP - 1 - p, p), d, ig) for p, d, ig in items]
        if rng.rand() < 0.5:
            x = x.transpose(0, 2, 1)
            items = [(p[[0, 2, 1]], d, ig) for p, d, ig in items]
        x = np.clip(np.ascontiguousarray(x) * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
        N = CROP // 2
        zz, yy, xx = np.ogrid[:N, :N, :N]
        H = np.zeros((N, N, N), np.float32); W = np.ones_like(H)
        R = np.zeros((5, 4), np.float32); RV = np.zeros((5, 4), np.float32); j = 0
        for p, d, ig in items:
            q = p / 2.0                      # stride-2 coordinates
            d2 = (zz - q[0]) ** 2 + (yy - q[1]) ** 2 + (xx - q[2]) ** 2
            r_out = d / 4.0
            if ig:
                W[d2 <= (r_out + 2) ** 2] = 0
                continue
            H = np.maximum(H, np.exp(-d2 / (2 * max(0.8, r_out / 2) ** 2)))
            ip = np.floor(q).astype(int)
            if np.all(ip >= 0) and np.all(ip < N):
                H[tuple(ip)] = 1.0
                if j < 5:
                    R[j] = [ip[0], ip[1], ip[2], 1]; RV[j] = [*(q - ip), math.log(d)]; j += 1
        return torch.from_numpy(x[None]), torch.from_numpy(H[None]), torch.from_numpy(W[None]), torch.from_numpy(R), torch.from_numpy(RV)


def focal(pred_logit, gt, w):
    p = torch.sigmoid(pred_logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = (1 - pos) * w
    pl = torch.log(p) * (1 - p) ** 2 * pos
    nl = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    return -(pl.sum() + nl.sum()) / pos.sum().clamp(min=1)


def reg_loss(off, ls, R, RV):
    loss, n = 0.0, 0
    for b in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[b, j, 3] == 0:
                continue
            iz, iy, ix = R[b, j, :3].long()
            loss = loss + F.l1_loss(off[b, :, iz, iy, ix].float(), RV[b, j, :3]) + F.l1_loss(ls[b, 0, iz, iy, ix].float(), RV[b, j, 3])
            n += 1
    return loss / max(n, 1)


def evaluate(model, keys, cache_dir, min_readers=3):
    per, ngt, rows = [], [], []
    for k in keys:
        vol = np.load(f"{cache_dir}/{k}.npy").astype(np.float32) / 255.0
        m = np.load(f"{cache_dir}/{k}_m.npy")
        H, O, S = sliding_heatmap(model, vol, device=DEV)
        c = extract_peaks(H, O, S, mask=m, thr=0.02)
        g = nod_by_key.get(k, nods.iloc[:0])
        gts, ign = gt_for_scan(g, min_readers)
        per.append(match_scan(c, gts, ign)); ngt.append(len(gts))
        for cc in c:
            rows.append(dict(key=k, **cc))
    return per, ngt, rows


model = NoduleDetector3D().to(DEV)
print("detector params", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ITERS, pct_start=0.05)
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
dl = DataLoader(CropDS(tr_keys, ITERS * BS), batch_size=BS, num_workers=int(os.environ.get("NW", 4)), pin_memory=True, drop_last=True,
                worker_init_fn=lambda w: np.random.seed(SEED * 100 + w))
best, hist = -1, []
model.train()
t = time.time()
for it, (x, H, W, R, RV) in enumerate(dl):
    x, H, W, R, RV = [a.to(DEV, non_blocking=True) for a in (x, H, W, R, RV)]
    with torch.autocast(device_type="cuda", enabled=DEV == "cuda"):
        h, o, s = model(x)
    loss_h = focal(h, H, W)
    loss_r = reg_loss(o, s, R, RV)
    loss = loss_h + 0.5 * loss_r
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update(); sched.step()
    if it % 200 == 0:
        print(f"it {it} loss {loss.item():.3f} heat {loss_h.item():.3f} reg {float(loss_r):.3f} {time.time()-t:.0f}s", flush=True)
    if (it + 1) % VAL_EVERY == 0 or it + 1 == ITERS:
        per, ngt, _ = evaluate(model, va_keys, TMP)
        fr = froc(per, ngt)
        hist.append(dict(it=it + 1, cpm=fr["cpm"], **{f"s@{r}": v for r, v in fr["sens"].items()}))
        print("VAL", hist[-1], flush=True)
        if fr["cpm"] > best:
            best = fr["cpm"]
            torch.save(model.state_dict(), f"{OUT}/detector_best.pth")
        model.train()
    if time.time() - T0 > 10.5 * 3600:
        print("time budget reached"); break
pd.DataFrame(hist).to_csv(f"{OUT}/det_val_history.csv", index=False)

# ---------------------------------------------------------------- choose operating points on VAL
model.load_state_dict(torch.load(f"{OUT}/detector_best.pth"))
per_v, ngt_v, _ = evaluate(model, va_keys, TMP)
fr_v = froc(per_v, ngt_v)
op = {str(r): fr_v["thr"][r] for r in [0.5, 1, 2, 4]}
json.dump(dict(val_froc=fr_v, operating_thresholds=op), open(f"{OUT}/det_val.json", "w"), indent=1, default=float)
print("VAL FROC", fr_v)

# ---------------------------------------------------------------- TEST (never seen) -- evaluated once
for k in te_keys:
    unpack(k)
res = {}
for mr in (3, 1):
    per_t, ngt_t, rows = evaluate(model, te_keys, TMP, min_readers=mr)
    fr_t = froc(per_t, ngt_t)
    # patient-bootstrap CI of CPM
    rng = np.random.default_rng(0); cpms = []
    for _ in range(500):
        s = rng.integers(0, len(per_t), len(per_t))
        cpms.append(froc([per_t[i] for i in s], [ngt_t[i] for i in s])["cpm"])
    res[f"min_readers_{mr}"] = dict(froc=fr_t, cpm_ci95=[float(np.percentile(cpms, 2.5)), float(np.percentile(cpms, 97.5))],
                                    at_val_thresholds={r: at_threshold(per_t, ngt_t, th) for r, th in op.items()})
    if mr == 3:
        pd.DataFrame(rows).to_csv(f"{OUT}/test_candidates.csv", index=False)
    print("TEST min_readers", mr, json.dumps(res[f"min_readers_{mr}"], default=float))
json.dump(res, open(f"{OUT}/det_test.json", "w"), indent=1, default=float)
print("total time", time.time() - T0)
