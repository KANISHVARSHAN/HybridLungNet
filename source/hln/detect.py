
# ============================================================================ DETECTION UTILS
import numpy as np
import torch
import torch.nn.functional as F

FROC_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]


@torch.no_grad()
def sliding_heatmap(model, vol01, win=128, overlap=24, device="cuda", amp=True):
    """vol01: (Z,Y,X) float in [0,1] at 1 mm. Returns heat (sigmoid), offset, logsize at stride 2."""
    Z, Y, X = vol01.shape
    pad = [max(0, win - s) for s in (Z, Y, X)]
    v = np.pad(vol01, [(0, p) for p in pad], constant_values=0)
    Zp, Yp, Xp = v.shape
    step = win - overlap
    starts = lambda S: sorted(set(list(range(0, max(S - win, 0) + 1, step)) + [max(S - win, 0)]))
    H = np.zeros((Zp // 2, Yp // 2, Xp // 2), np.float32)
    O = np.zeros((3,) + H.shape, np.float32)
    S = np.zeros(H.shape, np.float32)
    model.eval()
    for z in starts(Zp):
        for y in starts(Yp):
            for x in starts(Xp):
                t = torch.from_numpy(np.ascontiguousarray(v[z:z + win, y:y + win, x:x + win]))[None, None].float().to(device)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=amp and device == "cuda"):
                    h, o, s = model(t)
                h = torch.sigmoid(h.float())[0, 0].cpu().numpy()
                o = o.float()[0].cpu().numpy(); s = s.float()[0, 0].cpu().numpy()
                sl = (slice(z // 2, z // 2 + win // 2), slice(y // 2, y // 2 + win // 2), slice(x // 2, x // 2 + win // 2))
                better = h > H[sl]
                H[sl] = np.where(better, h, H[sl])
                for c in range(3):
                    O[c][sl] = np.where(better, o[c], O[c][sl])
                S[sl] = np.where(better, s, S[sl])
    hz, hy, hx = (Z + 1) // 2, (Y + 1) // 2, (X + 1) // 2
    return H[:hz, :hy, :hx], O[:, :hz, :hy, :hx], S[:hz, :hy, :hx]


def extract_peaks(H, O, S, mask=None, thr=0.02, topk=200, nms_mm=6.0):
    """Local maxima of heat-map -> list of dict(z,y,x in 1 mm voxels, score, diam_mm)."""
    t = torch.from_numpy(H)[None, None]
    mx = F.max_pool3d(t, 3, 1, 1)[0, 0].numpy()
    cand = np.argwhere((H == mx) & (H >= thr))
    sc = H[tuple(cand.T)]
    order = np.argsort(-sc)[: topk * 3]
    out = []
    for i in order:
        iz, iy, ix = cand[i]
        c = np.array([(iz + O[0, iz, iy, ix]) * 2, (iy + O[1, iz, iy, ix]) * 2, (ix + O[2, iz, iy, ix]) * 2])
        if mask is not None:
            q = np.clip(np.round(c).astype(int), 0, np.array(mask.shape) - 1)
            if not mask[tuple(q)]:
                continue
        if any(np.linalg.norm(c - np.array([p["z"], p["y"], p["x"]])) < nms_mm for p in out):
            continue
        out.append(dict(z=float(c[0]), y=float(c[1]), x=float(c[2]), score=float(sc[i]),
                        diam_mm=float(np.exp(np.clip(S[iz, iy, ix], 0, 4)))))
        if len(out) >= topk:
            break
    return out


def match_scan(cands, gts, ignores):
    """LUNA16-style: a candidate is a hit if within the GT radius (diam/2) of an unmatched
    GT nodule; hits on already-detected GTs are dropped; candidates on 'ignore' nodules
    (1-2 reader or <3 mm) are dropped. Returns list of (score, is_tp, gt_index or -1)."""
    res = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        p = np.array([c["z"], c["y"], c["x"]])
        hit = -1
        for gi, g in enumerate(gts):
            if np.linalg.norm(p - np.array([g["z"], g["y"], g["x"]])) <= max(g["diam_mm"] / 2, 1.5):
                hit = gi
                break
        if hit >= 0:
            res.append((c["score"], 1, hit))
            continue
        if any(np.linalg.norm(p - np.array([g["z"], g["y"], g["x"]])) <= max(g["diam_mm"] / 2, 3.0) for g in ignores):
            continue
        res.append((c["score"], 0, -1))
    return res


def froc(per_scan, n_gt_per_scan, rates=FROC_RATES):
    """per_scan: list of match_scan outputs (one per scan). Returns dict with sensitivity at
    each FP/scan rate, CPM, and the score threshold achieving each rate."""
    n_scans = len(per_scan)
    total_gt = int(sum(n_gt_per_scan))
    allc = []
    for s, res in enumerate(per_scan):
        for score, tp, gi in res:
            allc.append((score, tp, s, gi))
    allc.sort(key=lambda t: -t[0])
    found, fp = set(), 0
    curve = []  # (fp_per_scan, sens, thr)
    for score, tp, s, gi in allc:
        if tp:
            if (s, gi) not in found:
                found.add((s, gi))
        else:
            fp += 1
        curve.append((fp / n_scans, len(found) / max(total_gt, 1), score))
    out = {}
    for r in rates:
        ok = [c for c in curve if c[0] <= r]
        out[r] = (ok[-1][1], ok[-1][2]) if ok else (0.0, 1.0)
    return dict(sens={r: out[r][0] for r in rates}, thr={r: out[r][1] for r in rates},
                cpm=float(np.mean([out[r][0] for r in rates])), total_gt=total_gt, n_scans=n_scans,
                max_sens=(len(found) / max(total_gt, 1)))


def at_threshold(per_scan, n_gt_per_scan, thr):
    found, fp = set(), 0
    for s, res in enumerate(per_scan):
        for score, tp, gi in res:
            if score < thr:
                continue
            if tp:
                found.add((s, gi))
            else:
                fp += 1
    return dict(found=len(found), total_gt=int(sum(n_gt_per_scan)), sens=len(found) / max(sum(n_gt_per_scan), 1),
                fp_total=fp, fp_per_scan=fp / max(len(per_scan), 1), n_scans=len(per_scan))


def gt_for_scan(nods_df_scan, min_readers=3):
    """Split clustered XML nodules into evaluation GT (>=3 mm, >=min_readers readers)
    and 'ignore' (everything else annotated)."""
    gts, ign = [], []
    for _, n in nods_df_scan.iterrows():
        d = dict(z=n.z, y=n.y, x=n.x, diam_mm=max(float(n.diam_mm), 3.0), mal_mean=n.mal_mean, n_readers=n.n_readers_large)
        (gts if (not n.is_small and n.n_readers_large >= min_readers) else ign).append(d)
    return gts, ign
