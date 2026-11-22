
# ============================================================================ END-TO-END PIPELINE
# Raw DICOM files  ->  1 mm volume + lung mask  ->  detector candidates  ->
# HybridLungNet malignancy (MC-dropout)  ->  findings with location + uncertainty.
# NO annotation input of any kind is accepted here.
import json, time
import numpy as np
import torch
from skimage import measure


def compute_features(patch):   # identical to scripts/04_aux_features.py (training definition)
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


class Pipeline:
    def __init__(self, det_ckpt, cls_ckpt, aux_mean, aux_std, thresholds, device=None, mc_passes=20):
        """thresholds: dict with detector score thresholds chosen on the detector
        VALIDATION patients, e.g. {"confident": thr@1FP/scan, "possible": thr@4FP/scan}."""
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det = NoduleDetector3D().to(self.dev)
        self.det.load_state_dict(torch.load(det_ckpt, map_location=self.dev)); self.det.eval()
        self.cls = HybridLungNet().to(self.dev)
        self.cls.load_state_dict(torch.load(cls_ckpt, map_location=self.dev)); self.cls.eval()
        self.am, self.asd = np.asarray(aux_mean, np.float32), np.asarray(aux_std, np.float32)
        self.thr = thresholds
        self.mc = mc_passes

    def classify(self, vol01, centers):
        out = []
        for c in centers:
            p = extract_patch(vol01, c, 48, fill=0.0)
            a = ((np.asarray(compute_features(p), np.float32) - self.am) / self.asd).astype(np.float32)
            mu, sd = mc_dropout_predict(self.cls, torch.from_numpy(p)[None, None].to(self.dev),
                                        torch.from_numpy(a)[None].to(self.dev), n=self.mc)
            out.append((float(mu[0]), float(sd[0]), p))
        return out

    def run(self, dicom_paths, max_findings=10, keep_patches=False):
        t = {"start": time.time()}
        ct = load_ct(dicom_paths, min_slices=20)
        v = resample_1mm(ct); t["load_resample"] = time.time()
        m = lung_mask(v)
        lo, hi = bbox(m, margin=4)
        crop = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        mc = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        vol_det = to_uint8(crop).astype(np.float32) / 255.0      # identical quantisation to training cache
        vol_cls = normalize(crop)
        t["mask"] = time.time()
        H, O, S = sliding_heatmap(self.det, vol_det, device=self.dev, amp=self.dev == "cuda")
        cands = extract_peaks(H, O, S, mask=mc, thr=min(0.02, self.thr["possible"]))
        t["detect"] = time.time()
        keep = [c for c in cands if c["score"] >= self.thr["possible"]][:max_findings]
        cls = self.classify(vol_cls, [(c["z"], c["y"], c["x"]) for c in keep])
        t["classify"] = time.time()
        findings = []
        for c, (mu, sd, patch) in zip(keep, cls):
            z1, y1, x1 = c["z"] + lo[0], c["y"] + lo[1], c["x"] + lo[2]
            z_mm = float(ct.z_positions[0] + z1)
            f = dict(
                detection_score=round(c["score"], 4),
                detection_level="confident" if c["score"] >= self.thr["confident"] else "possible",
                malignancy_prob=round(mu, 4), malignancy_mc_std=round(sd, 4),
                est_diameter_mm=round(c["diam_mm"], 1),
                voxel_1mm=dict(z=round(z1, 1), y=round(y1, 1), x=round(x1, 1)),
                dicom=dict(z_mm=round(z_mm, 2), slice_index=int(np.argmin(np.abs(ct.z_positions - z_mm))),
                           row=round(y1 / ct.spacing[1], 1), col=round(x1 / ct.spacing[2], 1)),
                crop_center=dict(z=c["z"], y=c["y"], x=c["x"]),
            )
            if keep_patches:
                f["_patch"] = patch
            findings.append(f)
        conf = [f for f in findings if f["detection_level"] == "confident"]
        scan_score = max([f["malignancy_prob"] for f in conf], default=None)
        return dict(
            series_uid=ct.series_uid, n_slices=int(ct.meta["n_slices"]), spacing_mm=[round(s, 3) for s in ct.spacing],
            volume_1mm_shape=list(v.shape), lung_crop_lo=lo, findings=findings,
            n_confident=len(conf), n_possible=len(findings) - len(conf),
            scan_max_malignancy_confident=scan_score,
            timings_s={k: round(t[k] - t["start"], 2) for k in t if k != "start"},
            _crop=crop if keep_patches else None, _mask=mc if keep_patches else None, _ct=ct if keep_patches else None,
        )
