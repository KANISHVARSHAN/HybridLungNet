"""HybridLungNet v2 - inference API.

Input : CT DICOM files only (a whole patient folder is fine; the CT series with the most
        slices is used).  There is deliberately NO annotation / XML / coordinate input.
Output: nodule candidates (location, detection score, malignancy probability, MC-dropout
        spread) + slice images for the viewer.

Run:  C:\\mini-project\\.venv\\Scripts\\python.exe -m uvicorn main:app --port 8000
"""
import io
import json
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

V2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2))
from hln.lidc_io import *          # noqa: E402,F401
from hln.models import *           # noqa: E402,F401
from hln.detect import *           # noqa: E402,F401
import hln.pipeline as P           # noqa: E402

# make the pipeline module see the shared helpers (it is written to be concatenated in kernels)
for mod in ("hln.lidc_io", "hln.models", "hln.detect"):
    m = sys.modules[mod]
    for k in dir(m):
        if not k.startswith("__"):
            setattr(P, k, getattr(m, k))

MODEL_DIR = V2 / "models" if (V2 / "models" / "deploy_config.json").exists() else V2.parent / "models" / "deployed"  # v2 layout or repo layout
CFG = json.loads((MODEL_DIR / "deploy_config.json").read_text())
MODEL_CARD = json.loads((MODEL_DIR / "model_card.json").read_text()) if (MODEL_DIR / "model_card.json").exists() else {}

app = FastAPI(title="HybridLungNet v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PIPE = None
LOCK = threading.Lock()
SESSIONS = {}          # id -> dict(vol=uint8 crop, result=..., created=...)
MAX_SESSIONS = 4


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = P.Pipeline(str(MODEL_DIR / CFG["detector"]), str(MODEL_DIR / CFG["classifier"]),
                          CFG["aux_mean"], CFG["aux_std"], CFG["thresholds"], device="cpu",
                          mc_passes=CFG.get("mc_passes", 20))
    return PIPE


@app.get("/api/health")
def health():
    return {"status": "ok", "models_present": all((MODEL_DIR / CFG[k]).exists() for k in ("detector", "classifier")),
            "classifier": CFG.get("classifier_name"), "thresholds": CFG["thresholds"]}


@app.get("/api/model-card")
def model_card():
    return MODEL_CARD


def _collect(files: List[UploadFile], tmp: Path):
    paths = []
    for i, f in enumerate(files):
        data = f.file.read()
        name = Path(f.filename or f"f{i}").name
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for j, n in enumerate(z.namelist()):
                    if n.lower().endswith(".dcm"):
                        p = tmp / f"z{i}_{j}.dcm"; p.write_bytes(z.read(n)); paths.append(str(p))
        elif name.lower().endswith(".dcm") or "." not in name:
            p = tmp / f"{i}_{name}"; p.write_bytes(data); paths.append(str(p))
        elif name.lower().endswith(".xml"):
            raise HTTPException(400, "Annotation files are not accepted. Upload the CT DICOM files only.")
    return paths


@app.post("/api/analyze")
def analyze(files: List[UploadFile] = File(...)):
    with tempfile.TemporaryDirectory(prefix="hln_") as td:
        paths = _collect(files, Path(td))
        if len(paths) < 20:
            raise HTTPException(400, f"Found {len(paths)} DICOM files; a CT series needs at least 20 slices.")
        with LOCK:
            try:
                res = get_pipe().run(paths, max_findings=20, keep_patches=True)
            except ValueError as e:
                raise HTTPException(400, str(e))
    crop = res.pop("_crop"); res.pop("_mask"); ct = res.pop("_ct")
    for f in res["findings"]:
        f.pop("_patch", None)
    sid = uuid.uuid4().hex[:12]
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(min(SESSIONS, key=lambda k: SESSIONS[k]["created"]))
    SESSIONS[sid] = dict(vol=to_uint8(crop), result=res, created=time.time())
    res["session_id"] = sid
    res["crop_shape"] = list(crop.shape)
    return res


def _window(u8, wl, ww):
    hu = u8.astype(np.float32) / 255.0 * (HU_MAX - HU_MIN) + HU_MIN
    lo, hi = wl - ww / 2, wl + ww / 2
    return (np.clip((hu - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


@app.get("/api/session/{sid}/slice")
def slice_png(sid: str, axis: str = "axial", idx: int = 0, wl: float = -600, ww: float = 1500):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session expired - analyze the scan again.")
    v = s["vol"]
    ax = {"axial": 0, "coronal": 1, "sagittal": 2}[axis]
    idx = int(np.clip(idx, 0, v.shape[ax] - 1))
    img = np.take(v, idx, axis=ax)
    if ax != 0:
        img = img[::-1]          # superior at the top
    else:
        pass
    im = Image.fromarray(_window(img, wl, ww))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=False)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


FRONT = V2 / "app" / "frontend" / "dist"
if FRONT.exists():
    app.mount("/assets", StaticFiles(directory=FRONT / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONT / "index.html")
