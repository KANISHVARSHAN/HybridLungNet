"""
Shared CT I/O used by training (Kaggle) AND the end-to-end pipeline, so that
preprocessing is identical in both places.

Fixes relative to the original scripts (scripts/03, 06 and ui/backend/main.py):
  * Only ONE series is loaded (grouped by SeriesInstanceUID, CT only).  The original
    loader walked the whole patient folder and could interleave slices from
    different CT series of the same patient.
  * The z-axis voxel index is computed from real slice positions:
        z_1mm = (z_mm - z_min) / 1.0
    The original code used  argmin_slice_index * SliceThickness, which is wrong
    whenever slice spacing != SliceThickness (overlapping reconstructions).
  * Resampling uses the measured slice spacing (median diff of positions),
    not the SliceThickness tag.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

HU_MIN, HU_MAX = -1000.0, 400.0
NS = "{http://www.nih.gov}"


# ----------------------------------------------------------------------------- DICOM
@dataclass
class CTVolume:
    hu: np.ndarray                 # (Z,Y,X) float32 HU, original grid
    spacing: tuple                 # (dz, dy, dx) mm
    z_positions: np.ndarray        # sorted ascending, mm
    series_uid: str
    patient_id: str = ""
    meta: dict = field(default_factory=dict)


def _read_headers(paths):
    out = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "PixelSpacing"):
            continue
        out.append((p, ds))
    return out


def group_ct_series(paths):
    """Return {series_uid: [(path, header), ...]} for CT images only."""
    groups = defaultdict(list)
    for p, ds in _read_headers(paths):
        groups[str(ds.SeriesInstanceUID)].append((p, ds))
    return groups


def list_dicom_files(folder):
    files = []
    for r, _, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(".dcm"):
                files.append(os.path.join(r, f))
    return files


def load_ct(paths_or_folder, series_uid: str | None = None, min_slices: int = 20) -> CTVolume:
    """Load one CT series. If several series are present, pick `series_uid` or the
    one with the most slices."""
    paths = list_dicom_files(paths_or_folder) if isinstance(paths_or_folder, str) else list(paths_or_folder)
    groups = group_ct_series(paths)
    if not groups:
        raise ValueError("No CT DICOM images found")
    if series_uid is None:
        series_uid = max(groups, key=lambda k: len(groups[k]))
    items = groups[series_uid]
    if len(items) < min_slices:
        raise ValueError(f"Series has only {len(items)} slices (need >= {min_slices})")

    # de-duplicate identical z positions, sort ascending
    items.sort(key=lambda t: float(t[1].ImagePositionPatient[2]))
    seen, uniq = set(), []
    for p, ds in items:
        z = round(float(ds.ImagePositionPatient[2]), 3)
        if z in seen:
            continue
        seen.add(z)
        uniq.append(p)

    slices, zs = [], []
    ref_shape = None
    for p in uniq:
        ds = pydicom.dcmread(p, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if ref_shape is None:
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            continue
        arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        slices.append(arr)
        zs.append(float(ds.ImagePositionPatient[2]))
        last = ds
    zs = np.asarray(zs)
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
    ps = [float(v) for v in last.PixelSpacing]
    vol = np.stack(slices, 0)
    vol[vol < -1024] = -1024  # outside-FOV padding
    meta = {
        "slice_thickness_tag": float(getattr(last, "SliceThickness", np.nan) or np.nan),
        "slice_spacing_measured": dz,
        "n_slices": len(zs),
        "n_series_in_input": len(groups),
        "irregular_spacing": bool(len(zs) > 2 and np.ptp(np.diff(zs)) > 0.1),
    }
    return CTVolume(vol, (dz, ps[0], ps[1]), zs, series_uid, str(getattr(last, "PatientID", "")), meta)


def resample_1mm(ct: CTVolume) -> np.ndarray:
    """Resample to 1 mm isotropic. Voxel index (z,y,x) in output corresponds to
    world offset ((z_mm - z_min), row*dy, col*dx) in mm."""
    img = sitk.GetImageFromArray(ct.hu)
    dz, dy, dx = ct.spacing
    img.SetSpacing((dx, dy, dz))
    size = img.GetSize()
    new_size = [int(np.round(size[i] * img.GetSpacing()[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((1.0, 1.0, 1.0))
    rs.SetSize(new_size)
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-1024)
    return sitk.GetArrayFromImage(rs.Execute(img)).astype(np.float32)


def normalize(hu: np.ndarray) -> np.ndarray:
    return ((np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def to_uint8(hu: np.ndarray) -> np.ndarray:
    return np.round(normalize(hu) * 255).astype(np.uint8)


def world_to_1mm(ct: CTVolume, z_mm, row_px, col_px):
    """Annotation coords (z in mm, row/col in original pixels) -> 1 mm voxel index."""
    return (float(z_mm) - float(ct.z_positions[0]), float(row_px) * ct.spacing[1], float(col_px) * ct.spacing[2])


# ----------------------------------------------------------------------------- lung mask
def lung_mask(vol_hu_1mm: np.ndarray) -> np.ndarray:
    """Coarse lung mask on a 1 mm volume (threshold + connected components),
    dilated by ~10 mm so juxta-pleural nodules are kept. Done at 2 mm for speed."""
    from skimage.morphology import convex_hull_image
    small = vol_hu_1mm[::2, ::2, ::2]
    # body = largest soft-tissue component per slice; its convex hull contains the lungs
    # even when the field of view cuts the chest wall (the case that broke a
    # "drop components touching the border" rule).
    hull = np.zeros(small.shape, bool)
    for z in range(small.shape[0]):
        lab2, n2 = ndimage.label(small[z] > -500)
        if n2 == 0:
            continue
        s2 = ndimage.sum(np.ones_like(lab2), lab2, index=np.arange(1, n2 + 1))
        body = lab2 == (np.argmax(s2) + 1)
        if body.sum() > 200:
            hull[z] = ndimage.binary_erosion(convex_hull_image(body), iterations=2)
    air = (small < -320) & hull
    lab, n = ndimage.label(air)
    if n == 0:
        return np.ones_like(vol_hu_1mm, dtype=bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1] + 1
    keep = [l for l in order[:3] if sizes[l - 1] > 0.1 * sizes[order[0] - 1]]
    m = np.isin(lab, keep)
    if m.sum() < 2000:  # failure -> fall back to the body hull
        m = hull
    for z in range(m.shape[0]):
        m[z] = ndimage.binary_fill_holes(m[z])
    m = ndimage.binary_dilation(m, iterations=5)  # 5 voxels @2mm = 10 mm
    full = np.repeat(np.repeat(np.repeat(m, 2, 0), 2, 1), 2, 2)
    return full[: vol_hu_1mm.shape[0], : vol_hu_1mm.shape[1], : vol_hu_1mm.shape[2]]


def bbox(mask, margin=0):
    idx = np.where(mask)
    lo = [max(0, int(i.min()) - margin) for i in idx]
    hi = [min(s, int(i.max()) + 1 + margin) for i, s in zip(idx, mask.shape)]
    return lo, hi


# ----------------------------------------------------------------------------- XML
def parse_lidc_xml(path_or_bytes):
    """Return (series_uid, list of per-reader annotations).
    Each annotation: reader, z_mm, row, col (centroid), diam_px, z_extent_mm,
    malignancy (None for <3 mm nodules), is_small."""
    root = ET.fromstring(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else ET.parse(path_or_bytes).getroot()
    hdr = root.find(f"{NS}ResponseHeader")
    suid = hdr.find(f"{NS}SeriesInstanceUid").text.strip() if hdr is not None and hdr.find(f"{NS}SeriesInstanceUid") is not None else None
    anns = []
    for ri, sess in enumerate(root.findall(f"{NS}readingSession")):
        for nod in sess.findall(f"{NS}unblindedReadNodule"):
            chars = nod.find(f"{NS}characteristics")
            mal = chars.find(f"{NS}malignancy") if chars is not None else None
            mal = int(mal.text) if mal is not None and mal.text and mal.text.strip().isdigit() else None
            xs, ys, zs, per_slice = [], [], [], []
            for roi in nod.findall(f"{NS}roi"):
                inc = roi.find(f"{NS}inclusion")
                if inc is not None and inc.text and inc.text.strip().upper() != "TRUE":
                    continue
                zt = roi.find(f"{NS}imageZposition")
                if zt is None:
                    continue
                z = float(zt.text)
                ex = [(int(e.find(f"{NS}xCoord").text), int(e.find(f"{NS}yCoord").text)) for e in roi.findall(f"{NS}edgeMap")]
                if not ex:
                    continue
                a = np.array(ex)
                per_slice.append(max(np.ptp(a[:, 0]), np.ptp(a[:, 1])) + 1)
                xs += list(a[:, 0]); ys += list(a[:, 1]); zs += [z] * len(a)
            if not xs:
                continue
            anns.append(dict(reader=ri, z_mm=float(np.mean(zs)), row=float(np.mean(ys)), col=float(np.mean(xs)),
                             diam_px=float(max(per_slice)), z_extent_mm=float(np.ptp(zs)),
                             malignancy=mal, is_small=mal is None))
    return suid, anns


def cluster_annotations(anns, ct: CTVolume):
    """Group per-reader annotations of the same physical nodule.
    Two annotations are linked when their centroids are closer than
    max(0.5 * larger diameter, 5 mm) and they come from different readers.
    Returns list of nodule dicts in 1 mm voxel space."""
    pts = []
    for a in anns:
        z, y, x = world_to_1mm(ct, a["z_mm"], a["row"], a["col"])
        d = a["diam_px"] * ct.spacing[1]
        pts.append(dict(a, z=z, y=y, x=x, diam_mm=d))
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if pts[i]["reader"] == pts[j]["reader"]:
                continue
            dist = np.sqrt((pts[i]["z"] - pts[j]["z"]) ** 2 + (pts[i]["y"] - pts[j]["y"]) ** 2 + (pts[i]["x"] - pts[j]["x"]) ** 2)
            if dist <= max(0.5 * max(pts[i]["diam_mm"], pts[j]["diam_mm"]), 5.0):
                pairs.append((dist, i, j))
    for _, i, j in sorted(pairs):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        readers_i = {pts[k]["reader"] for k in range(n) if find(k) == ri}
        readers_j = {pts[k]["reader"] for k in range(n) if find(k) == rj}
        if readers_i & readers_j:
            continue  # never merge two marks by the same reader
        parent[rj] = ri
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(pts[i])
    out = []
    for g in groups.values():
        large = [p for p in g if not p["is_small"]]
        use = large if large else g
        mals = [p["malignancy"] for p in large]
        out.append(dict(
            z=float(np.mean([p["z"] for p in use])), y=float(np.mean([p["y"] for p in use])), x=float(np.mean([p["x"] for p in use])),
            diam_mm=float(np.mean([p["diam_mm"] for p in use])),
            n_readers=len({p["reader"] for p in g}), n_readers_large=len({p["reader"] for p in large}),
            malignancy_scores=mals, mal_mean=float(np.mean(mals)) if mals else None,
            is_small=len(large) == 0,
        ))
    return out


# ----------------------------------------------------------------------------- patches
def extract_patch(vol, center, size=48, fill=0.0):
    c = [int(round(v)) for v in center]
    h = size // 2
    out = np.full((size, size, size), fill, dtype=vol.dtype)
    src = [(max(0, c[i] - h), min(vol.shape[i], c[i] + h)) for i in range(3)]
    dst = [(s[0] - (c[i] - h), s[0] - (c[i] - h) + (s[1] - s[0])) for i, s in enumerate(src)]
    if any(s[1] <= s[0] for s in src):
        return out
    out[dst[0][0]:dst[0][1], dst[1][0]:dst[1][1], dst[2][0]:dst[2][1]] = vol[src[0][0]:src[0][1], src[1][0]:src[1][1], src[2][0]:src[2][1]]
    return out
