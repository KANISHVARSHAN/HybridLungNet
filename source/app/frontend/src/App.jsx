import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const WINDOWS = {
  Lung: { wl: -600, ww: 1500 },
  Mediastinum: { wl: 40, ww: 400 },
  Wide: { wl: -300, ww: 1400 },
};
const PLANES = ["axial", "coronal", "sagittal"];

// ---------------------------------------------------------------- helpers
async function readEntry(entry, out) {
  if (entry.isFile) {
    await new Promise((res) => entry.file((f) => { out.push(f); res(); }, res));
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    let batch;
    do {
      batch = await new Promise((res) => reader.readEntries(res, () => res([])));
      for (const e of batch) await readEntry(e, out);
    } while (batch.length);
  }
}
const isDicom = (f) => /\.dcm$/i.test(f.name) || /\.zip$/i.test(f.name) || !/\./.test(f.name);
const pct = (x) => `${Math.round(x * 100)}%`;
function riskColor(p) {
  if (p >= 0.5) return "var(--risk)";
  if (p >= 0.3) return "var(--warn)";
  return "var(--ok)";
}
function uncLevel(sd) {
  if (sd < 0.05) return ["Low", "var(--ok)"];
  if (sd < 0.12) return ["Moderate", "var(--warn)"];
  return ["High", "var(--risk)"];
}

// ---------------------------------------------------------------- icons
const I = {
  lung: (s = 18) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v8m0 0c-1 2-3 2.5-4 2.5M12 11c1 2 3 2.5 4 2.5" />
      <path d="M8.5 6.5C5 7 3 12 3 16.5 3 19 4.5 20 6.5 20c2.5 0 3.5-2 3.5-5V8" />
      <path d="M15.5 6.5C19 7 21 12 21 16.5 21 19 19.5 20 17.5 20c-2.5 0-3.5-2-3.5-5V8" />
    </svg>
  ),
  upload: (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V4m0 0l-4 4m4-4l4 4" /><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" /></svg>
  ),
  sun: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>,
  moon: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" /></svg>,
};

// ---------------------------------------------------------------- top bar
function TopBar({ health, theme, setTheme }) {
  const ok = health?.models_present;
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">{I.lung(18)}</div>
        <div>HybridLungNet<small>CT NODULE DETECTION · MALIGNANCY ESTIMATE</small></div>
      </div>
      <div className="spacer" />
      <span className="chip warn">Research prototype — not for clinical use</span>
      <span className={`chip ${health ? (ok ? "ok" : "bad") : ""}`}>
        <span className="dot" />
        {health ? (ok ? `Models ready · ${health.classifier || ""}` : "Model files missing") : "Connecting…"}
      </span>
      <button className="icon-btn" aria-label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
        {theme === "dark" ? I.sun : I.moon}
      </button>
    </header>
  );
}

// ---------------------------------------------------------------- left panel
const STEP_NAMES = [
  ["load_resample", "Read DICOM & resample to 1 mm"],
  ["mask", "Lung segmentation"],
  ["detect", "Nodule detection (3-D CenterNet)"],
  ["classify", "Malignancy (HybridLungNet, MC-dropout)"],
];

function LeftPanel({ files, setFiles, onAnalyze, busy, result, error }) {
  const [over, setOver] = useState(false);
  const folderRef = useRef(null), fileRef = useRef(null);
  const dicoms = files.filter(isDicom);
  const onDrop = async (e) => {
    e.preventDefault(); setOver(false);
    const out = [];
    const items = [...(e.dataTransfer.items || [])];
    if (items.length && items[0].webkitGetAsEntry) {
      for (const it of items) { const en = it.webkitGetAsEntry(); if (en) await readEntry(en, out); }
    } else out.push(...e.dataTransfer.files);
    setFiles(out);
  };
  const t = result?.timings_s || {};
  let prev = 0;
  return (
    <aside className="panel">
      <div className="section">
        <h3>Study input</h3>
        <div className={`drop ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
          <div style={{ color: "var(--accent)" }}>{I.upload}</div>
          <div className="big">Drop a CT study folder</div>
          <p>DICOM slices (.dcm) or a .zip. A whole patient folder is fine — the CT series with the most slices is used.</p>
          <div className="row">
            <button className="btn" onClick={() => folderRef.current.click()}>Choose folder</button>
            <button className="btn ghost" onClick={() => fileRef.current.click()}>Files</button>
          </div>
          <input ref={folderRef} type="file" webkitdirectory="" directory="" multiple hidden onChange={(e) => setFiles([...e.target.files])} />
          <input ref={fileRef} type="file" multiple accept=".dcm,.zip" hidden onChange={(e) => setFiles([...e.target.files])} />
        </div>
        <div className="filecount"><span>{dicoms.length} DICOM files selected</span>{files.length > dicoms.length && <span>{files.length - dicoms.length} ignored</span>}</div>
        <button className="btn primary" style={{ width: "100%", marginTop: 12 }} disabled={!dicoms.length || busy} onClick={() => onAnalyze(dicoms)}>
          {busy ? "Analyzing…" : "Analyze scan"}
        </button>
        <div className="note">No annotations or coordinates are needed or accepted — the system finds nodule candidates on its own.</div>
        {error && <div className="err-box">{error}</div>}
      </div>
      <div className="section">
        <h3>Pipeline</h3>
        <ul className="steps">
          {STEP_NAMES.map(([k, name], i) => {
            const done = t[k] !== undefined;
            const dt = done ? (t[k] - prev).toFixed(1) : null;
            if (done) prev = t[k];
            // the analysis is one request, so per-step progress is unknown while busy: all pending steps pulse
            return (
              <li key={k} className={done ? "done" : busy ? "active" : ""}>
                <span className="bullet">{done ? "✓" : ""}</span><span>{name}</span><span className="t">{dt ? `${dt}s` : ""}</span>
              </li>
            );
          })}
        </ul>
      </div>
      {result && (
        <div className="section">
          <h3>Series</h3>
          <dl className="kv">
            <dt>Slices</dt><dd>{result.n_slices}</dd>
            <dt>Spacing (z,y,x)</dt><dd>{result.spacing_mm.map((s) => s.toFixed(2)).join(" × ")} mm</dd>
            <dt>Volume @1 mm</dt><dd>{result.volume_1mm_shape.join(" × ")}</dd>
            <dt>Series UID</dt><dd title={result.series_uid}>…{result.series_uid.slice(-18)}</dd>
          </dl>
        </div>
      )}
    </aside>
  );
}

// ---------------------------------------------------------------- viewport
function Viewport({ result, sel, setSel, busy, plane, setPlane }) {
  const [win, setWin] = useState("Lung");
  const [idx, setIdx] = useState({ axial: 0, coronal: 0, sagittal: 0 });
  const [showMarks, setShowMarks] = useState(true);
  const [showPossible, setShowPossible] = useState(true);
  const stageRef = useRef(null);
  const [box, setBox] = useState({ w: 600, h: 600 });
  const shape = result?.crop_shape; // [Z,Y,X]

  useEffect(() => {
    if (!shape) return;
    const f = result.findings[0];
    setIdx({ axial: Math.round(f ? f.crop_center.z : shape[0] / 2), coronal: Math.round(f ? f.crop_center.y : shape[1] / 2), sagittal: Math.round(f ? f.crop_center.x : shape[2] / 2) });
  }, [result?.session_id]);

  useEffect(() => {
    if (sel == null || !result) return;
    const f = result.findings[sel];
    setIdx({ axial: Math.round(f.crop_center.z), coronal: Math.round(f.crop_center.y), sagittal: Math.round(f.crop_center.x) });
  }, [sel]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setBox({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = shape ? { axial: shape[0], coronal: shape[1], sagittal: shape[2] }[plane] : 1;
  const cur = Math.min(idx[plane], n - 1);
  const dims = shape ? { axial: [shape[2], shape[1]], coronal: [shape[2], shape[0]], sagittal: [shape[1], shape[0]] }[plane] : [1, 1];
  const scale = Math.min((box.w - 24) / dims[0], (box.h - 24) / dims[1]);
  const step = useCallback((d) => setIdx((s) => ({ ...s, [plane]: Math.max(0, Math.min(n - 1, s[plane] + d)) })), [plane, n]);

  useEffect(() => {
    const k = (e) => {
      if (!result || e.target.tagName === "INPUT") return;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") { step(1); e.preventDefault(); }
      if (e.key === "ArrowDown" || e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [step, result]);

  // finding marks on this slice
  const marks = [];
  if (result && showMarks) {
    result.findings.forEach((f, i) => {
      if (!showPossible && f.detection_level !== "confident") return;
      const c = f.crop_center, r = f.est_diameter_mm / 2 + 3;
      const [along, u, v] = plane === "axial" ? [c.z, c.x, c.y] : plane === "coronal" ? [c.y, c.x, shape[0] - 1 - c.z] : [c.x, c.y, shape[0] - 1 - c.z];
      const d = Math.abs(along - cur);
      if (d > r + 4) return;
      const rr = d < r ? Math.sqrt(r * r - d * d) : 0;
      marks.push({ i, u, v, r: Math.max(rr, 3), faint: d >= r, f });
    });
  }
  const { wl, ww } = WINDOWS[win];
  const src = result ? `/api/session/${result.session_id}/slice?axis=${plane}&idx=${cur}&wl=${wl}&ww=${ww}` : null;

  // prefetch neighbours for smooth scrolling
  useEffect(() => {
    if (!result) return;
    [1, 2, -1, -2].forEach((d) => {
      const j = cur + d;
      if (j >= 0 && j < n) { const im = new Image(); im.src = `/api/session/${result.session_id}/slice?axis=${plane}&idx=${j}&wl=${wl}&ww=${ww}`; }
    });
  }, [cur, plane, win, result?.session_id]);

  const ticks = result ? result.findings.map((f, i) => ({ i, pos: (plane === "axial" ? f.crop_center.z : plane === "coronal" ? f.crop_center.y : f.crop_center.x) / Math.max(n - 1, 1), f })) : [];

  return (
    <section className="center">
      <div className="vtoolbar">
        <div className="seg">{PLANES.map((p) => <button key={p} className={plane === p ? "on" : ""} onClick={() => setPlane(p)}>{p[0].toUpperCase() + p.slice(1)}</button>)}</div>
        <span className="sep" />
        <div className="seg">{Object.keys(WINDOWS).map((w) => <button key={w} className={win === w ? "on" : ""} onClick={() => setWin(w)}>{w}</button>)}</div>
        <span className="sep" />
        <label className="tog"><input type="checkbox" checked={showMarks} onChange={(e) => setShowMarks(e.target.checked)} /> Markers</label>
        <label className="tog"><input type="checkbox" checked={showPossible} onChange={(e) => setShowPossible(e.target.checked)} /> Show “possible”</label>
      </div>
      <div className="stage" ref={stageRef} onWheel={(e) => result && step(e.deltaY > 0 ? 1 : -1)}>
        {!result && !busy && (
          <div className="empty">
            <div className="lungs" style={{ color: "#4b5875" }}>{I.lung(64)}</div>
            <h2>No study loaded</h2>
            <p>Drop a chest CT on the left. The detector scans the whole volume, marks nodule candidates, and the classifier estimates malignancy for each one with an uncertainty range.</p>
          </div>
        )}
        {result && (
          <div className="frame" style={{ width: dims[0] * scale, height: dims[1] * scale }}>
            <img src={src} alt={`${plane} slice ${cur}`} draggable={false} />
            <svg viewBox={`0 0 ${dims[0]} ${dims[1]}`} preserveAspectRatio="none">
              {marks.map((m) => {
                const color = m.f.detection_level === "confident" ? riskColor(m.f.malignancy_prob) : "#9aa6bf";
                return (
                  <g key={m.i} opacity={m.faint ? 0.35 : 1} style={{ pointerEvents: "auto", cursor: "pointer" }} onClick={() => setSel(m.i)}>
                    <circle cx={m.u} cy={m.v} r={m.r} fill="none" stroke={color} strokeWidth={sel === m.i ? 2 / scale * 1.6 : 1.4 / scale * 1.2} strokeDasharray={m.f.detection_level === "confident" ? "" : `${4 / scale} ${3 / scale}`} />
                    <text x={m.u + m.r + 2} y={m.v - m.r} fill={color} fontSize={12 / scale} fontFamily="Inter" fontWeight="700">{m.i + 1}</text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}
        {result && (
          <>
            <div className="corner tl">{plane.toUpperCase()}<br />Slice {cur + 1} / {n}</div>
            <div className="corner tr">W {ww} · L {wl}<br />1.0 mm isotropic</div>
            <div className="corner bl">{plane === "axial" ? "R ⟷ L (radiological)" : "S ↑"}</div>
            <div className="corner br">Scroll / ↑↓ to page</div>
          </>
        )}
        {busy && (
          <div className="overlay-msg"><div className="box"><div className="spinner" /><b>Analyzing scan</b><div className="note">Reading slices, detecting nodules and estimating malignancy. On CPU this takes about a minute.</div></div></div>
        )}
      </div>
      <div className="scrubber">
        <button className="icon-btn" onClick={() => step(-1)} disabled={!result} aria-label="Previous slice">‹</button>
        <div className="track">
          {ticks.map((t) => <span key={t.i} className="tick" title={`Finding ${t.i + 1}`} style={{ left: `calc(${t.pos * 100}% - 1px)`, background: t.f.detection_level === "confident" ? riskColor(t.f.malignancy_prob) : "var(--text-3)" }} />)}
          <input type="range" min={0} max={Math.max(n - 1, 0)} value={cur} disabled={!result} onChange={(e) => setIdx((s) => ({ ...s, [plane]: +e.target.value }))} />
        </div>
        <span className="n">{result ? `${cur + 1} / ${n}` : "—"}</span>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- right panel
function Meter({ label, value, sd, color, thr }) {
  return (
    <div className="meter">
      <div className="lab"><span>{label}</span><b>{pct(value)}{sd != null && <span style={{ color: "var(--text-3)", fontWeight: 500 }}> ± {pct(sd)}</span>}</b></div>
      <div className="bar">
        <div className="fill" style={{ width: pct(value), background: color }} />
        {sd != null && <div className="err" style={{ left: pct(Math.max(0, value - sd)), width: pct(Math.min(1, value + sd) - Math.max(0, value - sd)) }} />}
        {thr != null && <div className="thr" style={{ left: pct(thr) }} title="decision threshold" />}
      </div>
    </div>
  );
}

function FindingCard({ f, i, selected, onClick }) {
  const [ul, uc] = uncLevel(f.malignancy_mc_std);
  const conf = f.detection_level === "confident";
  const side = f.voxel_1mm.x < (f._width || 0) / 2 ? "" : "";
  return (
    <button className={`fcard ${selected ? "sel" : ""}`} onClick={onClick}>
      <div className="top">
        <span className="badge" style={{ background: conf ? riskColor(f.malignancy_prob) : "var(--panel)", color: conf ? "#0b0f17" : "var(--text-2)", border: conf ? 0 : "1px solid var(--line)" }}>{i + 1}</span>
        <div>
          <div className="title">≈ {f.est_diameter_mm} mm candidate</div>
          <div className="sub">DICOM slice {f.dicom.slice_index + 1} · row {Math.round(f.dicom.row)} · col {Math.round(f.dicom.col)}</div>
        </div>
        <span className={`pill ${conf ? "conf" : "poss"}`}>{conf ? "Confident" : "Possible"}</span>
      </div>
      <Meter label="Detection confidence" value={f.detection_score} color="var(--accent)" />
      <Meter label="Malignancy estimate" value={f.malignancy_prob} sd={f.malignancy_mc_std} color={riskColor(f.malignancy_prob)} thr={0.5} />
      <div className="unc">Model spread (MC-dropout): <span className="lvl" style={{ color: uc }}>{ul}</span>{!conf && <span>· may not be a nodule</span>}</div>
    </button>
  );
}

function RightPanel({ result, sel, setSel, card }) {
  const conf = result ? result.findings.filter((f) => f.detection_level === "confident") : [];
  const maxp = conf.length ? Math.max(...conf.map((f) => f.malignancy_prob)) : null;
  let cls = "ok", verdict = "No confident nodule found", text = "The detector found no candidate above its confident threshold. Small or subtle nodules can still be missed — see limitations.";
  if (conf.length && maxp >= 0.5) { cls = "risk"; verdict = "Suspicious nodule found"; text = `${conf.length} confident candidate${conf.length > 1 ? "s" : ""}; highest malignancy estimate ${pct(maxp)}. Needs review by a radiologist.`; }
  else if (conf.length) { cls = "warn"; verdict = `${conf.length} nodule candidate${conf.length > 1 ? "s" : ""}`; text = `Highest malignancy estimate ${pct(maxp)} (below the 50% threshold). Benign-appearing does not mean benign.`; }
  return (
    <aside className="panel right">
      <div className="section">
        <h3>Result</h3>
        {!result ? <div className="note" style={{ marginTop: 0 }}>Results appear here after analysis.</div> : (
          <div className={`summary ${cls}`}>
            <div className="head"><span className="verdict">{verdict}</span></div>
            <p>{text}</p>
            <div className="stats3">
              <div className="stat"><b>{result.n_confident}</b><span>confident</span></div>
              <div className="stat"><b>{result.n_possible}</b><span>possible</span></div>
              <div className="stat"><b>{maxp == null ? "—" : pct(maxp)}</b><span>max malignancy</span></div>
            </div>
          </div>
        )}
      </div>
      {result && result.findings.length > 0 && (
        <div className="section">
          <h3>Findings <span>{result.findings.length}</span></h3>
          <div className="findings">{result.findings.map((f, i) => <FindingCard key={i} f={f} i={i} selected={sel === i} onClick={() => setSel(i)} />)}</div>
        </div>
      )}
      <div className="section">
        <h3>How to read this</h3>
        <div className="callout">
          <b>Detection confidence</b> — how sure the detector is that a nodule is there.<br />
          <b>Malignancy estimate</b> — HybridLungNet's probability, <i>assuming</i> the spot is a real nodule. On false detections it is not meaningful.<br />
          <b>± spread</b> — variation across 20 MC-dropout passes; measured to be only a weak signal of errors.
        </div>
      </div>
      {card && card.metrics && (
        <div className="section">
          <details className="limits" open>
            <summary>Measured performance (held-out patients) <span>▾</span></summary>
            <table className="metric-table"><tbody>
              {card.metrics.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
            </tbody></table>
          </details>
        </div>
      )}
      {card && card.limitations && (
        <div className="section">
          <details className="limits">
            <summary>Known limitations <span>▾</span></summary>
            <ul>{card.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
          </details>
        </div>
      )}
    </aside>
  );
}

// ---------------------------------------------------------------- app
export default function App() {
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem("hln-theme") || "dark"; } catch { return "dark"; } });
  const [health, setHealth] = useState(null);
  const [card, setCard] = useState(null);
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [sel, setSel] = useState(null);
  const [plane, setPlane] = useState("axial");

  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("hln-theme", theme); } catch {} }, [theme]);
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => setHealth({ models_present: false }));
    fetch("/api/model-card").then((r) => r.json()).then(setCard).catch(() => {});
  }, []);

  const analyze = async (dicoms) => {
    setBusy(true); setError(null); setResult(null); setSel(null);
    const fd = new FormData();
    dicoms.forEach((f) => fd.append("files", f, f.name));
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || `Server error ${r.status}`);
      setResult(j); if (j.findings.length) setSel(0);
    } catch (e) { setError(String(e.message || e)); }
    setBusy(false);
  };

  return (
    <div className="app">
      <TopBar health={health} theme={theme} setTheme={setTheme} />
      <main className="main">
        <LeftPanel files={files} setFiles={setFiles} onAnalyze={analyze} busy={busy} result={result} error={error} />
        <Viewport result={result} sel={sel} setSel={setSel} busy={busy} plane={plane} setPlane={setPlane} />
        <RightPanel result={result} sel={sel} setSel={setSel} card={card} />
      </main>
      <footer className="footer">
        <span><strong>Not a medical device.</strong> Trained and tested on LIDC-IDRI only; outputs must be reviewed by a qualified radiologist.</span>
        <span style={{ marginLeft: "auto" }}>{result ? `Processed in ${result.timings_s.classify?.toFixed(1)} s` : "HybridLungNet v2"}</span>
      </footer>
    </div>
  );
}
