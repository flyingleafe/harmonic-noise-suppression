// MD2 Fig. 6(a) interactive waveform re-renderer: browser side.
// Served by docs/explainers/md2-fit-renderer/server.py at /app.js; the Quarto page's
// OJS boot cell imports this module and mounts the node returned by mountRenderer().
// Every synthesis happens in the backend; this module only edits parameters, sends
// them on an explicit Re-render, and displays what the backend returned.

const REGIMES = ["standby", "cruise"];

const SHAFT_FIELDS = [
  {
    key: "sigma_ref_rad_s",
    sym: "σ_ref",
    label: "σ_ref: speed-error sd at the reference rate f_ref",
    unit: "rad/s",
    rule: "nonneg",
  },
  {
    key: "a_sigma",
    sym: "a_σ",
    label: "a_σ: exponent of σ_ν in r/f_ref",
    unit: "dimensionless",
    rule: "finite",
  },
  {
    key: "lam_ref_per_s",
    sym: "λ_ref",
    label: "λ_ref: speed-error OU rate at f_ref",
    unit: "1/s",
    rule: "pos",
  },
  {
    key: "b_lam",
    sym: "b_λ",
    label: "b_λ: exponent of λ in r/f_ref",
    unit: "dimensionless",
    rule: "finite",
  },
  {
    key: "f_ref_rev_s",
    sym: "f_ref",
    label: "f_ref: reference rotor rate (fixed by the source fit, read-only)",
    unit: "rev/s",
    rule: "pos",
    readonly: true,
  },
];

const PED_FIELDS = [
  {
    key: "sigma2_amp",
    sym: "σ²_A",
    long: "amplitude log-variance σ²_A",
    unit: "Np²",
    cap: null,
  },
  {
    key: "gamma_amp_hz",
    sym: "γ_A",
    long: "amplitude decorrelation rate γ_A",
    unit: "Hz",
    cap: null,
  },
  {
    key: "sigma2_phase",
    sym: "σ²_P",
    long: "phase variance σ²_P",
    unit: "rad²",
    cap: 3.0,
  },
  {
    key: "width_phase_hz",
    sym: "W",
    long: "phase sideband mean half-width W",
    unit: "Hz",
    cap: 24.0,
  },
];
const PED_BY_KEY = Object.fromEntries(PED_FIELDS.map((f) => [f.key, f]));
const SHAFT_BY_KEY = Object.fromEntries(SHAFT_FIELDS.map((f) => [f.key, f]));

// Display-only probe rates: the regime-band edges of the smoothstep blend.
const PROBE_DEFAULT_REV_S = { standby: 45, cruise: 65 };
const FMAX_OPTIONS_HZ = [1000, 2000, 3000, 4000, 5000, 6000, 8000];
const FMAX_DEFAULT_HZ = 3000;
const DR_MIN_DB = 10;
const DR_MAX_DB = 120;
const MAX_LISTED_CHANGES = 40;
const MAGMA_COLORS = [
  [0, "#000004"], [1 / 9, "#180f3d"], [2 / 9, "#440f76"],
  [3 / 9, "#721f81"], [4 / 9, "#9e2f7f"], [5 / 9, "#cd4071"],
  [6 / 9, "#f1605d"], [7 / 9, "#fd9668"], [8 / 9, "#feca8d"],
  [1, "#fcfdbf"],
];

const LAUNCH_CMD = "python docs/explainers/md2-fit-renderer/server.py --port 8766";

// ---------------------------------------------------------------- utilities

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k in node && typeof v !== "string") node[k] = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function clone(x) {
  return JSON.parse(JSON.stringify(x));
}

function fmt(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(6)));
}

function fmtShort(v, digits = 4) {
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return String(Number(v.toPrecision(digits)));
  return v.toExponential(digits - 1);
}

const NUMBER_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function parseNumber(text, rule) {
  const s = String(text).trim();
  if (s === "") return { error: "empty" };
  if (!NUMBER_RE.test(s)) return { error: `“${s}” is not a number` };
  const v = Number(s);
  if (!Number.isFinite(v)) return { error: "must be finite" };
  if (rule === "nonneg" && v < 0) return { error: "must be ≥ 0" };
  if (rule === "pos" && !(v > 0)) return { error: "must be > 0" };
  return { value: v };
}

function gammaPhaseHz(width, sigma2) {
  if (sigma2 === 0) return width;
  return (width * -Math.expm1(-sigma2)) / sigma2;
}

function diffPaths(a, b, path, out) {
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
    for (let i = 0; i < a.length; i++) diffPaths(a[i], b[i], path.concat(i), out);
    return out;
  }
  const isObj = (x) => x !== null && typeof x === "object" && !Array.isArray(x);
  if (isObj(a) && isObj(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) diffPaths(a[k], b[k], path.concat(k), out);
    return out;
  }
  if (a !== b) out.push(path);
  return out;
}

function paramDiff(a, b) {
  const out = [];
  for (const r of REGIMES) diffPaths(a[r], b[r], [r], out);
  return out;
}

function pathLabel(path) {
  const [regime, ...rest] = path;
  if (rest[0] === "pedestal" && rest.length === 3 && PED_BY_KEY[rest[1]]) {
    const f = PED_BY_KEY[rest[1]];
    return `${regime}: ${f.long} at order k=${rest[2] + 1}`;
  }
  if (rest[0] === "shaft" && rest.length === 2 && SHAFT_BY_KEY[rest[1]]) {
    return `${regime}: shaft ${SHAFT_BY_KEY[rest[1]].sym}`;
  }
  let s = `${regime}: `;
  rest.forEach((p, i) => {
    s += typeof p === "number" ? `[${p}]` : (i ? "." : "") + p;
  });
  return s;
}

// Exact structure check against the canonical template the backend handed out.
function structureErrors(template, candidate, path, out) {
  if (out.length >= 12) return out;
  const where = path || "(root)";
  if (template === null) {
    if (candidate !== null) out.push(`${where}: must stay null`);
    return out;
  }
  if (Array.isArray(template)) {
    if (!Array.isArray(candidate)) out.push(`${where}: must be an array of length ${template.length}`);
    else if (candidate.length !== template.length)
      out.push(`${where}: length ${candidate.length}, expected ${template.length}`);
    else template.forEach((t, i) => structureErrors(t, candidate[i], `${path}[${i}]`, out));
    return out;
  }
  if (typeof template === "object") {
    if (candidate === null || typeof candidate !== "object" || Array.isArray(candidate)) {
      out.push(`${where}: must be an object`);
      return out;
    }
    for (const k of Object.keys(template))
      if (!(k in candidate)) out.push(`${where}: missing key “${k}”`);
    for (const k of Object.keys(candidate))
      if (!(k in template)) out.push(`${where}: unknown key “${k}” (the backend rejects new keys)`);
    for (const k of Object.keys(template))
      if (k in candidate) structureErrors(template[k], candidate[k], path ? `${path}.${k}` : k, out);
    return out;
  }
  if (typeof template === "number") {
    if (typeof candidate !== "number" || !Number.isFinite(candidate))
      out.push(`${where}: must be a finite number`);
    return out;
  }
  if (typeof candidate !== typeof template) out.push(`${where}: must be a ${typeof template}`);
  return out;
}

// JSON with numeric arrays kept on one line, so a regime fits in a textarea.
function prettyJSON(value, indent = "") {
  const inner = indent + "  ";
  if (Array.isArray(value)) {
    if (value.every((v) => v === null || typeof v !== "object")) return JSON.stringify(value);
    return `[\n${value.map((v) => inner + prettyJSON(v, inner)).join(",\n")}\n${indent}]`;
  }
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value);
    if (!keys.length) return "{}";
    return `{\n${keys.map((k) => `${inner}${JSON.stringify(k)}: ${prettyJSON(value[k], inner)}`).join(",\n")}\n${indent}}`;
  }
  return JSON.stringify(value);
}

function decodeSpec(spec, label) {
  if (!spec || !Array.isArray(spec.shape) || spec.shape.length !== 2)
    throw new Error(`${label} spectrogram: missing shape [n_bins, n_frames]`);
  const [nb, nf] = spec.shape.map(Number);
  if (!(nb > 0 && nf > 0)) throw new Error(`${label} spectrogram: invalid shape ${JSON.stringify(spec.shape)}`);
  if (typeof spec.db_f32_base64 !== "string") throw new Error(`${label} spectrogram: missing db_f32_base64`);
  const bin = atob(spec.db_f32_base64);
  if (bin.length !== nb * nf * 4)
    throw new Error(`${label} spectrogram: ${bin.length} bytes for shape ${nb}×${nf} (expected ${nb * nf * 4})`);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const freqs = Array.from(spec.freqs_hz || [], Number);
  const times = Array.from(spec.times_s || [], Number);
  if (freqs.length !== nb) throw new Error(`${label} spectrogram: ${freqs.length} freqs_hz for ${nb} bins`);
  if (times.length !== nf) throw new Error(`${label} spectrogram: ${times.length} times_s for ${nf} frames`);
  // Frequency-major: bin b occupies values [b*nf, (b+1)*nf).
  const rows = new Array(nb);
  const meanDb = new Array(nb);
  for (let b = 0; b < nb; b++) {
    const row = new Array(nf);
    let acc = 0;
    for (let t = 0; t < nf; t++) {
      const db = view.getFloat32((b * nf + t) * 4, true);
      row[t] = db;
      acc += Math.pow(10, db / 10);
    }
    rows[b] = row;
    meanDb[b] = 10 * Math.log10(Math.max(acc / nf, 1e-30));
  }
  return { nb, nf, rows, freqs, times, meanDb };
}

function signalFrom(obj, label) {
  if (!obj || typeof obj !== "object") throw new Error(`${label}: missing signal block`);
  if (typeof obj.waveform_sha256 !== "string") throw new Error(`${label}: missing waveform_sha256`);
  if (typeof obj.rms !== "number") throw new Error(`${label}: missing rms`);
  return { spec: decodeSpec(obj.spectrogram, label), sha: obj.waveform_sha256, rms: obj.rms };
}

function checkAudio(audio, where) {
  for (const k of ["real", "baseline", "edited"])
    if (!audio || typeof audio[k] !== "string" || !audio[k].startsWith("data:"))
      throw new Error(`${where}: audio.${k} is not a data URL`);
}

function checkParams(p, where) {
  for (const r of REGIMES) {
    const q = p && p[r];
    if (!q || !q.shaft || !q.pedestal) throw new Error(`${where}: parameters.${r} lacks shaft/pedestal`);
    for (const f of SHAFT_FIELDS)
      if (typeof q.shaft[f.key] !== "number") throw new Error(`${where}: parameters.${r}.shaft.${f.key} missing`);
    const K = q.pedestal.sigma2_amp && q.pedestal.sigma2_amp.length;
    for (const f of PED_FIELDS)
      if (!Array.isArray(q.pedestal[f.key]) || q.pedestal[f.key].length !== K)
        throw new Error(`${where}: parameters.${r}.pedestal.${f.key} missing or not length ${K}`);
  }
}

class ApiError extends Error {}

async function api(path, { method = "GET", body } = {}) {
  let resp;
  try {
    resp = await fetch(path, {
      method,
      cache: "no-store",
      headers: body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    throw new ApiError(
      `Cannot reach the renderer backend at ${window.location.origin}${path} (${err.message}). Is server.py still running?`
    );
  }
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!resp.ok) {
    let msg = text.slice(0, 600) || resp.statusText;
    if (data && typeof data === "object") {
      if (typeof data.error === "string") msg = data.error;
      else if (data.error && typeof data.error.message === "string") msg = data.error.message;
      else if (typeof data.message === "string") msg = data.message;
    }
    const busy = resp.status === 409 ? " (another render is still running)" : "";
    throw new ApiError(`${method} ${path} failed with HTTP ${resp.status}${busy}: ${msg}`);
  }
  if (data === null || typeof data !== "object") throw new ApiError(`${method} ${path} returned non-JSON content`);
  return data;
}

function loadStyle(href) {
  if (document.querySelector(`link[data-md2r-style]`)) return Promise.resolve(true);
  return new Promise((resolve) => {
    const link = el("link", { rel: "stylesheet", href, "data-md2r-style": "1" });
    link.addEventListener("load", () => resolve(true));
    link.addEventListener("error", () => resolve(false));
    document.head.append(link);
  });
}

function loadPlotly(src) {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  return new Promise((resolve, reject) => {
    // Hide an AMD loader, if a page script installed one, so Plotly attaches to window.
    const amd = typeof window.define === "function" && window.define.amd ? window.define : null;
    if (amd) window.define = undefined;
    const script = el("script", { src });
    const restore = () => {
      if (amd) window.define = amd;
    };
    script.addEventListener("load", () => {
      restore();
      if (window.Plotly) resolve(window.Plotly);
      else reject(new Error(`${src} loaded but did not define Plotly`));
    });
    script.addEventListener("error", () => {
      restore();
      reject(new Error(`Could not load ${src} from the renderer backend`));
    });
    document.head.append(script);
  });
}

function downloadJSON(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- application

export async function mountRenderer() {
  const root = el("div", { id: "md2r", class: "md2r" });
  const bootStatus = el("p", { id: "md2r-boot", class: "md2r-boot", role: "status", "aria-live": "polite" },
    "Connecting to the local renderer backend…");
  root.append(bootStatus);

  const styleOk = await loadStyle("/style.css");
  const boot = async () => {
    root.replaceChildren(bootStatus);
    bootStatus.textContent = "Loading Plotly and the fitted-baseline state from the backend…";
    try {
      const Plotly = await loadPlotly("/plotly.min.js");
      const state = await api("/api/state");
      const app = buildApp(Plotly, state, styleOk);
      root.replaceChildren(app);
    } catch (err) {
      const retry = el("button", { type: "button", id: "md2r-retry", class: "md2r-btn", onclick: boot }, "Retry connection");
      root.replaceChildren(
        el("div", { id: "md2r-boot-error", class: "md2r-offline", role: "alert" },
          el("p", {}, el("strong", { text: "The renderer could not start." }), " ", err.message),
          el("p", { text: "No controls are available without the live backend. Start it from the repository root:" }),
          el("pre", {}, el("code", { text: LAUNCH_CMD })),
          retry)
      );
    }
  };
  boot();
  return root;
}

function buildApp(Plotly, state, styleOk) {
  // ---------- validate and decode the backend state
  const ex = state.example;
  if (!ex || typeof ex.vmax_db !== "number") throw new Error("GET /api/state: example.vmax_db missing");
  if (ex.shaft_law_evaluation !== "local_work_grid")
    throw new Error("The backend still uses the old clip-mean shaft renderer. Restart server.py to load the corrected local-speed baseline.");
  checkParams(state.parameters, "GET /api/state");
  checkAudio(state.audio, "GET /api/state");
  if (typeof state.playback_gain !== "number") throw new Error("GET /api/state: playback_gain missing");
  const real = signalFrom(state.real, "real");
  const baseline = signalFrom(state.baseline, "baseline");

  const S = {
    original: clone(state.parameters),
    applied: clone(state.parameters),
    draft: clone(state.parameters),
    edited: { ...baseline, source: "baseline" },
    audio: { ...state.audio },
    gain: state.playback_gain,
    exports: null,
    renderCount: 0,
    lastElapsed: null,
    lastRoundtrip: null,
    busy: false,
    invalid: new Map(), // input id -> message
    tableRegime: "standby",
    advDirty: false,
    view: { fmax: FMAX_DEFAULT_HZ, dr: Number(ex.dynamic_range_db) || 50, spectrum: true },
  };
  const vmax = ex.vmax_db;

  // ---------- status and error area
  const statusEl = el("p", { id: "md2r-status", class: "md2r-status", role: "status", "aria-live": "polite" });
  const errorText = el("div", { id: "md2r-error-text" });
  const errorBox = el("div", { id: "md2r-error", class: "md2r-error", role: "alert", hidden: true },
    el("strong", { text: "Error. " }), errorText,
    el("button", { type: "button", class: "md2r-btn md2r-btn-small", onclick: () => clearError() }, "Dismiss"));
  const infoEl = el("p", { id: "md2r-info", class: "md2r-info", role: "status", "aria-live": "polite" });

  function showError(msg) {
    errorText.textContent = msg;
    errorBox.hidden = false;
  }
  function clearError() {
    errorBox.hidden = true;
    errorText.textContent = "";
  }
  function info(msg) {
    infoEl.textContent = msg;
  }

  // ---------- clip identity
  const fits = ex.source_fits || {};
  const clip = el("section", { id: "md2r-clip", class: "md2r-card", "aria-labelledby": "md2r-clip-h" },
    el("h3", { id: "md2r-clip-h", text: "Clip and synthesis settings" }),
    el("dl", { class: "md2r-dl" },
      ...[
        ["Recording", `${ex.recording}, ${ex.start_s} s + ${ex.duration_s} s`],
        ["Real microphone", `index ${ex.mic}, ${ex.sr} Hz`],
        ["Synthesis", `render_noise_regimes, n_mics = ${ex.n_mics_render}, seed ${ex.seed} (fixed)`],
        ["Spectrogram", `periodic Hann, N_FFT ${ex.n_fft}, hop ${ex.hop}`],
        ["Shaft coefficients", "Local rotor speed at each work-grid sample; corrected baseline regenerated"],
        ["Colour top v_max", `${fmtShort(vmax, 5)} dB (fixed: 99.5th percentile of real + baseline ≤ 3 kHz)`],
      ].flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]),
      ...REGIMES.flatMap((r) => [
        el("dt", { text: `${r} fit (read-only)` }),
        el("dd", {},
          el("code", { class: "md2r-path", text: (fits[r] && fits[r].path) || "?" }),
          el("br"),
          el("span", { class: "md2r-hash", text: `sha256 ${(fits[r] && fits[r].sha256) || "?"}` })),
      ])
    )
  );

  // ---------- action bar
  const renderBtn = el("button", { type: "button", id: "md2r-render", class: "md2r-btn md2r-btn-primary", onclick: doRender }, "Re-render waveform");
  const discardBtn = el("button", { type: "button", id: "md2r-discard", class: "md2r-btn", onclick: discardPending }, "Discard pending edits");
  const resetBtn = el("button", { type: "button", id: "md2r-reset", class: "md2r-btn", onclick: resetAll }, "Reset to fitted baseline");
  const dlBtns = Object.fromEntries(
    REGIMES.map((r) => [
      r,
      el("button", { type: "button", id: `md2r-download-${r}`, class: "md2r-btn", onclick: () => doDownload(r) }, `Download ${r} payload`),
    ])
  );
  const pendingSummary = el("p", { id: "md2r-pending", class: "md2r-pending" });
  const pendingList = el("ul", { id: "md2r-pending-list", class: "md2r-pending-list" });
  const pendingDetails = el("details", { id: "md2r-pending-details" }, el("summary", { text: "List pending edits" }), pendingList);
  const appliedSummary = el("p", { id: "md2r-applied", class: "md2r-applied" });
  const exportNote = el("p", { id: "md2r-export-note", class: "md2r-note" });
  const actions = el("section", { id: "md2r-actions", class: "md2r-card md2r-actions", "aria-label": "Render and export" },
    el("div", { class: "md2r-btnrow" }, renderBtn, discardBtn, resetBtn),
    statusEl, errorBox, infoEl, pendingSummary, pendingDetails, appliedSummary,
    el("div", { class: "md2r-btnrow" }, dlBtns.standby, dlBtns.cruise),
    exportNote);

  // ---------- view controls
  const fmaxSel = el("select", { id: "md2r-fmax" },
    ...FMAX_OPTIONS_HZ.map((f) => el("option", { value: String(f), selected: f === S.view.fmax }, `${f / 1000} kHz`)));
  const drInput = el("input", { id: "md2r-dr", type: "range", min: String(DR_MIN_DB), max: String(DR_MAX_DB), step: "5", value: String(S.view.dr) });
  const drOut = el("output", { id: "md2r-dr-out", for: "md2r-dr" });
  const specToggle = el("input", { id: "md2r-show-spectrum", type: "checkbox", checked: S.view.spectrum });
  const gainEl = el("p", { id: "md2r-gain", class: "md2r-note" });
  const view = el("section", { id: "md2r-view", class: "md2r-card", "aria-labelledby": "md2r-view-h" },
    el("h3", { id: "md2r-view-h", text: "Display (no re-render)" }),
    el("div", { class: "md2r-viewrow" },
      el("label", { for: "md2r-fmax" }, "Maximum frequency ", fmaxSel),
      el("label", { for: "md2r-dr" }, "Dynamic range below v_max ", drInput, " ", drOut),
      el("label", { for: "md2r-show-spectrum", class: "md2r-check" }, specToggle, " Mean linear-power spectrum")),
    gainEl);
  fmaxSel.addEventListener("change", () => {
    S.view.fmax = Number(fmaxSel.value);
    drawAll();
  });
  drInput.addEventListener("input", () => {
    S.view.dr = Number(drInput.value);
    drawAll();
  });
  specToggle.addEventListener("change", () => {
    S.view.spectrum = specToggle.checked;
    spectrumPanel.hidden = !S.view.spectrum;
    drawSpectrum();
  });

  // ---------- plots
  function panel(key, title) {
    const plot = el("div", { id: `md2r-plot-${key}`, class: "md2r-plot", role: "img", "aria-label": `${title} spectrogram` });
    const audio = el("audio", { id: `md2r-audio-${key}`, controls: true, preload: "metadata" });
    audio.setAttribute("aria-label", `${title} audio, shared playback gain`);
    const heading = el("h4", { id: `md2r-title-${key}`, text: title });
    const caption = el("p", { id: `md2r-meta-${key}`, class: "md2r-hash" });
    const node = el("figure", { class: "md2r-panel", id: `md2r-panel-${key}` }, heading, plot, audio, caption);
    return { node, plot, audio, heading, caption };
  }
  const P = {
    real: panel("real", "Real recording"),
    baseline: panel("baseline", "Frozen fitted baseline"),
    edited: panel("edited", "Edited render"),
  };
  const staleBadge = el("p", { id: "md2r-edited-stale", class: "md2r-stale", hidden: true });
  P.edited.node.insertBefore(staleBadge, P.edited.plot);
  const spectrumPlot = el("div", { id: "md2r-plot-spectrum", class: "md2r-plot md2r-plot-spectrum", role: "img", "aria-label": "Mean linear-power spectra" });
  const spectrumPanel = el("figure", { class: "md2r-panel md2r-panel-wide", id: "md2r-panel-spectrum" },
    el("h4", { text: "Mean spectrum: frame average of the linear periodogram, in dB" }), spectrumPlot);
  const plots = el("section", { id: "md2r-plots", class: "md2r-plots", "aria-label": "Spectrograms and audio" },
    P.real.node, P.baseline.node, P.edited.node, spectrumPanel);

  // ---------- shaft editor
  const probeInputs = {};
  const shaftReadouts = {};
  const shaftSets = REGIMES.map((r) => {
    const rows = SHAFT_FIELDS.map((f) => {
      const id = `md2r-${r}-shaft-${f.key}`;
      const input = el("input", {
        id, type: "text", inputmode: "decimal", spellcheck: "false", autocomplete: "off",
        "data-regime": r, "data-group": f.readonly ? "fixed" : "shaft", "data-key": f.key,
        "aria-describedby": `${id}-msg`, readOnly: Boolean(f.readonly), "aria-readonly": f.readonly ? "true" : null,
        class: f.readonly ? "md2r-readonly" : null,
      });
      return el("div", { class: "md2r-field" },
        el("label", { for: id }, f.label, el("span", { class: "md2r-unit", text: ` (${f.unit})` })),
        input,
        el("span", { id: `${id}-msg`, class: "md2r-fieldmsg", "aria-live": "polite" }));
    });
    const probeId = `md2r-${r}-probe-r`;
    probeInputs[r] = el("input", { id: probeId, type: "text", inputmode: "decimal", value: String(PROBE_DEFAULT_REV_S[r]), "aria-describedby": `md2r-${r}-shaft-readout` });
    shaftReadouts[r] = el("p", { id: `md2r-${r}-shaft-readout`, class: "md2r-readout", "aria-live": "polite" });
    probeInputs[r].addEventListener("input", () => updateShaftReadout(r));
    return el("fieldset", { class: "md2r-shaft", id: `md2r-${r}-shaft` },
      el("legend", { text: `${r}: shaft speed-error law` }),
      el("p", { class: "md2r-formula" },
        "σ_ν(r) = σ_ref · (r / f_ref)^a_σ  rad/s,   λ(r) = λ_ref · (r / f_ref)^b_λ  1/s,   r = rotor rate (rev/s). ",
        "σ_ref is the law's magnitude at r = f_ref, not this regime's operating jitter."),
      ...rows,
      el("div", { class: "md2r-field md2r-probe" },
        el("label", { for: probeId }, "Evaluate the law at rotor rate r (display only, rev/s)"),
        probeInputs[r]),
      shaftReadouts[r]);
  });

  // ---------- mean line levels and their speed laws
  const amplitudeRegime = el("select", { id: "md2r-amplitude-regime" },
    ...REGIMES.map((r) => el("option", { value: r, selected: r === "cruise", text: r })));
  const amplitudeFamily = el("select", { id: "md2r-amplitude-family" },
    el("option", { value: "aero", text: "Aerodynamic harmonics" }),
    el("option", { value: "motor", text: "Motor lines" }));
  const amplitudeRotor = el("select", { id: "md2r-amplitude-rotor" },
    el("option", { value: "all", text: "All rotors (preserve relative levels)" }),
    ...state.parameters.cruise.profile_aero_db.map((_, i) =>
      el("option", { value: String(i), text: `Rotor ${i + 1}` })));
  const amplitudeParity = el("select", { id: "md2r-amplitude-parity" },
    el("option", { value: "all", text: "All orders" }),
    el("option", { value: "even", text: "Even orders / BPF harmonics" }),
    el("option", { value: "odd", text: "Odd orders" }));
  const amplitudeShift = el("input", {
    id: "md2r-amplitude-shift", type: "text", inputmode: "decimal", value: "6",
    "aria-label": "Gain to add to all shown harmonic levels, dB",
  });
  const amplitudeShiftButton = el("button", {
    id: "md2r-amplitude-shift-apply", type: "button", class: "md2r-btn",
    onclick: () => shiftAmplitudeLevels(false),
  }, "Add gain to shown orders");
  const amplitudeResetButton = el("button", {
    id: "md2r-amplitude-reset", type: "button", class: "md2r-btn",
    onclick: () => shiftAmplitudeLevels(true),
  }, "Reset shown levels");
  const amplitudeMessage = el("p", {
    id: "md2r-amplitude-message", class: "md2r-fieldmsg", role: "status", "aria-live": "polite",
  });
  const amplitudeRow = el("div", {
    id: "md2r-amplitude-sliders", class: "md2r-amplitude-sliders", tabindex: "0",
    "aria-label": "Harmonic amplitude gain sliders; scroll horizontally",
  });
  const amplitudeLawTitle = el("h4", { id: "md2r-amplitude-law-title" });
  const amplitudeLaws = [
    ["alpha_aero", "α: aerodynamic line-power exponent"],
    ["alpha_motor", "β: motor line-power exponent"],
  ].map(([key, label]) => {
    const input = el("input", {
      id: `md2r-amplitude-${key}`, type: "text", inputmode: "decimal",
      "aria-describedby": `md2r-amplitude-${key}-msg`,
    });
    const message = el("span", {
      id: `${input.id}-msg`, class: "md2r-fieldmsg", "aria-live": "polite",
    });
    input.addEventListener("input", () => {
      const parsed = parseNumber(input.value, "finite");
      message.textContent = parsed.error || "";
      if (parsed.error) {
        input.setAttribute("aria-invalid", "true");
        S.invalid.set(input.id, parsed.error);
      } else {
        input.removeAttribute("aria-invalid");
        S.invalid.delete(input.id);
        const r = amplitudeRegime.value;
        S.draft[r][key] = parsed.value;
        markPending(input, parsed.value, S.applied[r][key]);
        if (!S.advDirty) loadAdvanced(false);
      }
      refreshStatus();
    });
    return { key, input, message, node: el("div", { class: "md2r-field" },
      el("label", { for: input.id }, label), input, message) };
  });
  const amplitudes = el("fieldset", { id: "md2r-amplitudes", class: "md2r-orders" },
    el("legend", { text: "Harmonic mean amplitudes and speed laws" }),
    el("p", { class: "md2r-note" },
      "Each vertical slider changes the mean line level relative to its fitted profile, in dB at 80 rev/s. ",
      "+6 dB is approximately twice the RMS amplitude. These are mean levels, not amplitude-fluctuation variances. ",
      "All rotors applies a common gain while preserving their current relative levels."),
    el("div", { class: "md2r-viewrow" },
      el("label", { for: amplitudeRegime.id }, "Regime ", amplitudeRegime),
      el("label", { for: amplitudeFamily.id }, "Line family ", amplitudeFamily),
      el("label", { for: amplitudeRotor.id }, "Rotor ", amplitudeRotor),
      el("label", { for: amplitudeParity.id }, "Show ", amplitudeParity)),
    amplitudeLawTitle,
    el("p", { class: "md2r-formula" },
      "P_a(r) = P_a,80 · (r/80)^α; P_m(r) = P_m,80 · (r/80)^β. ",
      "RMS amplitudes scale with exponents α/2 and β/2. Negative exponents are allowed."),
    el("div", { class: "md2r-amplitude-laws" }, ...amplitudeLaws.map((f) => f.node)),
    el("div", { class: "md2r-viewrow" },
      el("label", { for: amplitudeShift.id }, "Gain to add (dB) ", amplitudeShift),
      amplitudeShiftButton, amplitudeResetButton),
    amplitudeMessage, amplitudeRow);

  function amplitudeSelection() {
    const regime = amplitudeRegime.value;
    const family = amplitudeFamily.value;
    const key = family === "aero" ? "profile_aero_db" : "profile_motor_db";
    const matrix = S.draft[regime][key];
    const rotors = amplitudeRotor.value === "all"
      ? matrix.map((_, i) => i) : [Number(amplitudeRotor.value)];
    const orders = family === "aero"
      ? matrix[0].map((_, i) => i + 1) : S.draft[regime].motor_orders;
    const columns = orders.map((order, index) => ({ order, index })).filter(({ order }) =>
      amplitudeParity.value === "all" ||
      (amplitudeParity.value === "even" ? order % 2 === 0 : order % 2 !== 0));
    return { regime, family, key, matrix, rotors, columns };
  }

  function selectedPowerDb(matrix, index, rotors) {
    const top = Math.max(...rotors.map((r) => matrix[r][index]));
    return top + 10 * Math.log10(
      rotors.reduce((sum, r) => sum + 10 ** ((matrix[r][index] - top) / 10), 0) / rotors.length);
  }

  function amplitudeGain(selection, index) {
    return selectedPowerDb(selection.matrix, index, selection.rotors) -
      selectedPowerDb(S.original[selection.regime][selection.key], index, selection.rotors);
  }

  function updateAmplitudeColumn(node, selection, index) {
    const gain = amplitudeGain(selection, index);
    const slider = node.querySelector("input");
    slider.value = String(gain);
    slider.classList.toggle("md2r-pending-cell", selection.rotors.some((r) =>
      selection.matrix[r][index] !== S.applied[selection.regime][selection.key][r][index]));
    node.querySelector("output").value = `${gain >= 0 ? "+" : ""}${gain.toFixed(2)} dB`;
    const power = selectedPowerDb(selection.matrix, index, selection.rotors);
    node.querySelector(".md2r-level-absolute").textContent = `${power.toFixed(1)} dB ref.`;
    slider.setAttribute("aria-valuetext", `${gain.toFixed(2)} dB gain; reference line level ${power.toFixed(2)} dB`);
  }

  function syncAmplitudeEditor() {
    const selection = amplitudeSelection();
    amplitudeLawTitle.textContent = `${selection.regime}: amplitude speed laws`;
    for (const field of amplitudeLaws) {
      field.input.value = fmt(S.draft[selection.regime][field.key]);
      field.input.title = `fitted value ${fmt(S.original[selection.regime][field.key])}`;
      field.input.removeAttribute("aria-invalid");
      field.message.textContent = "";
      S.invalid.delete(field.input.id);
      markPending(field.input, S.draft[selection.regime][field.key], S.applied[selection.regime][field.key]);
    }
    const gains = selection.columns.map(({ index }) => amplitudeGain(selection, index));
    const lower = Math.min(-30, ...gains.map((g) => Math.floor(g / 5) * 5));
    const upper = Math.max(30, ...gains.map((g) => Math.ceil(g / 5) * 5));
    const fragment = document.createDocumentFragment();
    for (const { order, index } of selection.columns) {
      const id = `md2r-level-${selection.regime}-${selection.family}-${order}`;
      const slider = el("input", {
        id, type: "range", min: String(lower), max: String(upper), step: "any",
        class: "md2r-level-slider",
        "aria-label": `${selection.regime} ${selection.family} order ${order} mean-amplitude gain, dB`,
      });
      const column = el("div", { class: "md2r-level-column" },
        el("label", { for: id, text: `k=${order}` }),
        el("output", { for: id }), slider,
        el("span", { class: "md2r-level-absolute" }));
      slider.addEventListener("input", () => {
        const current = amplitudeSelection();
        const delta = Number(slider.value) - amplitudeGain(current, index);
        for (const r of current.rotors) current.matrix[r][index] += delta;
        updateAmplitudeColumn(column, current, index);
        if (!S.advDirty) loadAdvanced(false);
        refreshStatus();
      });
      updateAmplitudeColumn(column, selection, index);
      fragment.append(column);
    }
    amplitudeRow.replaceChildren(fragment);
    amplitudeShiftButton.disabled = !selection.columns.length;
    amplitudeResetButton.disabled = !selection.columns.length;
    amplitudeMessage.classList.remove("md2r-bad");
    amplitudeMessage.textContent =
      `${selection.regime}: ${selection.columns.length} ${selection.family} levels; gains relative to the fitted profile. Scroll horizontally for more orders.`;
    refreshStatus();
  }

  function shiftAmplitudeLevels(reset) {
    if (S.busy) return;
    const selection = amplitudeSelection();
    const parsed = reset ? { value: 0 } : parseNumber(amplitudeShift.value, "finite");
    if (parsed.error) {
      amplitudeMessage.textContent = `Gain ${parsed.error}`;
      amplitudeMessage.classList.add("md2r-bad");
      return;
    }
    for (const { index } of selection.columns)
      for (const r of selection.rotors)
        selection.matrix[r][index] = reset
          ? S.original[selection.regime][selection.key][r][index]
          : selection.matrix[r][index] + parsed.value;
    syncAmplitudeEditor();
    if (!S.advDirty) loadAdvanced(false);
  }

  for (const select of [amplitudeRegime, amplitudeFamily, amplitudeRotor, amplitudeParity])
    select.addEventListener("change", syncAmplitudeEditor);

  // ---------- per-order editor
  const orderRegimeSel = el("select", { id: "md2r-order-regime" },
    ...REGIMES.map((r) => el("option", { value: r, text: r })));
  const bulk = {
    from: el("input", { id: "md2r-bulk-from", type: "number", min: "1", step: "1" }),
    to: el("input", { id: "md2r-bulk-to", type: "number", min: "1", step: "1" }),
    parity: el("select", { id: "md2r-bulk-parity" },
      el("option", { value: "all", text: "all orders" }),
      el("option", { value: "even", text: "even orders" }),
      el("option", { value: "odd", text: "odd orders" })),
    field: el("select", { id: "md2r-bulk-field" },
      ...PED_FIELDS.map((f) => el("option", { value: f.key, text: `${f.long} (${f.unit})` }))),
    op: el("select", { id: "md2r-bulk-op" },
      el("option", { value: "set", text: "set to" }),
      el("option", { value: "mul", text: "multiply by" })),
    value: el("input", { id: "md2r-bulk-value", type: "text", inputmode: "decimal", autocomplete: "off" }),
    apply: el("button", { type: "button", id: "md2r-bulk-apply", class: "md2r-btn" }, "Apply to draft"),
    msg: el("p", { id: "md2r-bulk-msg", class: "md2r-fieldmsg", role: "status", "aria-live": "polite" }),
  };
  bulk.apply.addEventListener("click", applyBulk);
  bulk.value.addEventListener("keydown", (e) => {
    if (e.key === "Enter") applyBulk();
  });
  const tbody = el("tbody", { id: "md2r-order-tbody" });
  const table = el("table", { id: "md2r-order-table", class: "md2r-table" },
    el("caption", { id: "md2r-order-caption" }),
    el("thead", {},
      el("tr", {},
        el("th", { scope: "col", text: "order k" }),
        ...PED_FIELDS.map((f) =>
          el("th", { scope: "col" }, `${f.sym} (${f.unit})`,
            f.cap === null ? null : el("span", { class: "md2r-capref", text: `fit cap ${f.cap} ${f.unit}, reference only` }))),
        el("th", { scope: "col" }, "γ_P (Hz)", el("span", { class: "md2r-capref", text: "derived: W(1−e^−σ²_P)/σ²_P" })))),
    tbody);
  const orders = el("fieldset", { id: "md2r-orders", class: "md2r-orders" },
    el("legend", { text: "Per-order pedestal (aerodynamic order k = shaft harmonic)" }),
    el("p", { class: "md2r-note" },
      "Amplitude: log-amplitude OU, variance σ²_A, correlation e^(−2πγ_A τ). ",
      "Phase: variance σ²_P and mean sideband half-width W (not the STFT line's full width); ",
      "the phase rate γ_P is derived, never edited. Values above the fitting caps are allowed and are highlighted."),
    el("div", { class: "md2r-viewrow" }, el("label", { for: "md2r-order-regime" }, "Regime ", orderRegimeSel)),
    el("div", { class: "md2r-bulk", role: "group", "aria-labelledby": "md2r-bulk-h" },
      el("p", { id: "md2r-bulk-h", class: "md2r-bulk-title", text: "Bulk edit (current regime)" }),
      el("label", { for: "md2r-bulk-from" }, "from k ", bulk.from),
      el("label", { for: "md2r-bulk-to" }, "to k ", bulk.to),
      el("label", { for: "md2r-bulk-parity" }, "parity ", bulk.parity),
      el("label", { for: "md2r-bulk-field" }, "field ", bulk.field),
      el("label", { for: "md2r-bulk-op" }, "operation ", bulk.op),
      el("label", { for: "md2r-bulk-value" }, "value ", bulk.value),
      bulk.apply,
      bulk.msg),
    el("div", { class: "md2r-tablewrap", tabindex: "0", "aria-label": "Per-order parameter table, scrollable" }, table));
  orderRegimeSel.addEventListener("change", () => {
    S.tableRegime = orderRegimeSel.value;
    resetBulkRange();
    buildTable();
  });

  // ---------- advanced JSON
  const advRegimeSel = el("select", { id: "md2r-adv-regime" }, ...REGIMES.map((r) => el("option", { value: r, text: r })));
  const advText = el("textarea", { id: "md2r-adv-json", rows: "18", spellcheck: "false", "aria-describedby": "md2r-adv-help" });
  const advLoad = el("button", { type: "button", id: "md2r-adv-load", class: "md2r-btn" }, "Reload from draft");
  const advApply = el("button", { type: "button", id: "md2r-adv-apply", class: "md2r-btn" }, "Apply JSON to draft");
  const advMsg = el("p", { id: "md2r-adv-msg", class: "md2r-fieldmsg", role: "status", "aria-live": "polite" });
  const glossary = el("details", { class: "md2r-glossary" },
    el("summary", { text: "Key meanings and units" }),
    el("dl", { class: "md2r-dl" },
      ...[
        ["profile_aero_db", "aerodynamic line power per rotor × order k = 1…K at r = f_ref, dB (periodogram units)"],
        ["alpha_aero", "speed exponent of the aerodynamic lines: power ∝ (r/f_ref)^alpha_aero"],
        ["motor_orders", "shaft orders of the motor lines (positive integers)"],
        ["profile_motor_db, alpha_motor", "motor-line power per rotor × motor order at f_ref (dB) and its speed exponent"],
        ["pedestal.*", "the per-order arrays of the table above (index i → order k = i + 1)"],
        ["shaft.*", "the shaft law of the fieldsets above; f_ref_rev_s is read-only (fixed at the source fit's value)"],
        ["floor.mean_db, shape_z, shape_sd_db, ctrl_hz", "broadband floor at f_ref: mean level (dB), whitened shape coefficients, their scale (dB) and increasing control frequencies (Hz)"],
        ["floor.b0, floor.b1", "floor speed exponent b(f) = b0 + b1·log2(max(f, band_lo)/2 kHz)"],
        ["floor.static_rel, floor.floor_rule", "speed-independent floor term (linear, relative to the floor at f_ref) and the floor composition rule"],
        ["mic_response.*", "array response: per-mic band gains (dB) and rotor × mic geometry gains (dB)"],
        ["wind", "flight wind block; null here (no wind rendered)"],
      ].flatMap(([k, v]) => [el("dt", {}, el("code", { text: k })), el("dd", { text: v })])));
  const advanced = el("details", { id: "md2r-advanced", class: "md2r-advanced" },
    el("summary", { text: "Advanced: canonical parameter JSON for one regime" }),
    el("p", { id: "md2r-adv-help", class: "md2r-note" },
      "Edit any parameter the renderer reads. Keys and array shapes must stay exactly as given; the backend rejects added or missing keys and any change of the read-only shaft.f_ref_rev_s. ",
      "Apply changes the pending draft only; press Re-render to synthesise it."),
    el("div", { class: "md2r-viewrow" }, el("label", { for: "md2r-adv-regime" }, "Regime ", advRegimeSel), advLoad, advApply),
    advText, advMsg, glossary);
  advRegimeSel.addEventListener("change", () => loadAdvanced(true));
  advLoad.addEventListener("click", () => loadAdvanced(true));
  advApply.addEventListener("click", applyAdvanced);
  advText.addEventListener("input", () => {
    S.advDirty = true;
    advMsg.textContent = "Unapplied JSON edits: press Apply JSON to draft.";
  });

  const editor = el("fieldset", { id: "md2r-editor", class: "md2r-editor" },
    el("legend", { text: "Parameter editor (changes are pending until Re-render)" }),
    el("p", { class: "md2r-note" },
      "Corrected renderer: shaft-noise magnitude and rate follow each rotor's local speed at every work-grid sample, with continuous state evolution. ",
      "The fit uses a locally stationary frame approximation. The baseline was regenerated; the rate probes below remain display-only."),
    amplitudes,
    el("div", { class: "md2r-shaftgrid" }, ...shaftSets),
    orders,
    advanced);
  editor.addEventListener("input", onEditorInput);

  // ---------- assemble
  const app = el("div", { class: "md2r-app" },
    styleOk ? null : el("p", { class: "md2r-error", role: "alert", text: "Warning: /style.css failed to load; layout is unstyled." }),
    clip, actions, view, plots, editor);

  // ---------- behaviour: editing
  function fieldRule(input) {
    if (input.dataset.group === "shaft") return SHAFT_BY_KEY[input.dataset.key].rule;
    return "nonneg";
  }

  function onEditorInput(e) {
    const input = e.target;
    const group = input.dataset && input.dataset.group;
    if (group !== "shaft" && group !== "ped") return;
    const r = input.dataset.regime;
    const res = parseNumber(input.value, fieldRule(input));
    const msgEl = group === "shaft" ? editor.querySelector(`#${input.id}-msg`) : null;
    if (res.error) {
      S.invalid.set(input.id, res.error);
      input.setAttribute("aria-invalid", "true");
      input.title = res.error;
      if (msgEl) msgEl.textContent = res.error;
      refreshStatus();
      return;
    }
    S.invalid.delete(input.id);
    input.removeAttribute("aria-invalid");
    if (msgEl) msgEl.textContent = "";
    if (group === "shaft") {
      const key = input.dataset.key;
      S.draft[r].shaft[key] = res.value;
      input.title = `fitted value ${fmt(S.original[r].shaft[key])} ${SHAFT_BY_KEY[key].unit}`;
      updateShaftReadout(r);
      markPending(input, S.draft[r].shaft[key], S.applied[r].shaft[key]);
    } else {
      const k = Number(input.dataset.idx);
      S.draft[r].pedestal[input.dataset.key][k] = res.value;
      input.title = titleFor(r, input.dataset.key, k);
      updateRowDerived(r, k);
    }
    if (!S.advDirty) loadAdvanced(false);
    refreshStatus();
  }

  function markPending(input, draftV, appliedV) {
    input.classList.toggle("md2r-pending-cell", draftV !== appliedV);
  }

  function updateShaftReadout(r) {
    const sh = S.draft[r].shaft;
    const out = shaftReadouts[r];
    const eps = sh.sigma_ref_rad_s / (2 * Math.PI * sh.f_ref_rev_s);
    const pr = parseNumber(probeInputs[r].value, "pos");
    let probe;
    if (pr.error) {
      probeInputs[r].setAttribute("aria-invalid", "true");
      probe = `probe rate ${pr.error}`;
    } else {
      probeInputs[r].removeAttribute("aria-invalid");
      const x = pr.value / sh.f_ref_rev_s;
      const sig = sh.sigma_ref_rad_s * Math.pow(x, sh.a_sigma);
      const lam = sh.lam_ref_per_s * Math.pow(x, sh.b_lam);
      probe = `at r = ${fmtShort(pr.value)} rev/s: σ_ν = ${fmtShort(sig)} rad/s (order-1 frequency sd σ_ν/2π = ${fmtShort(sig / (2 * Math.PI))} Hz), λ = ${fmtShort(lam)} 1/s`;
    }
    out.textContent = `Stored payload value ε = σ_ref/(2π f_ref) = ${fmtShort(eps)}. ${probe}.`;
  }

  function titleFor(r, key, i) {
    const o = S.original[r].pedestal[key][i];
    const f = PED_BY_KEY[key];
    let t = `fitted value ${fmt(o)} ${f.unit}`;
    const v = S.draft[r].pedestal[key][i];
    if (f.cap !== null && v > f.cap) t += `; above the fitting cap ${f.cap} ${f.unit} (allowed)`;
    return t;
  }

  function updateRowDerived(r, i) {
    if (r !== S.tableRegime) return;
    const row = tbody.rows[i];
    if (!row) return;
    const ped = S.draft[r].pedestal;
    PED_FIELDS.forEach((f, j) => {
      const input = row.cells[j + 1].firstChild;
      const v = ped[f.key][i];
      input.classList.toggle("md2r-above-cap", f.cap !== null && v > f.cap);
      markPending(input, v, S.applied[r].pedestal[f.key][i]);
    });
    row.cells[PED_FIELDS.length + 1].textContent = fmtShort(gammaPhaseHz(ped.width_phase_hz[i], ped.sigma2_phase[i]));
  }

  function buildTable() {
    const r = S.tableRegime;
    const ped = S.draft[r].pedestal;
    const K = ped.sigma2_amp.length;
    for (const id of [...S.invalid.keys()]) if (id.startsWith("md2r-ped-")) S.invalid.delete(id);
    table.caption.textContent = `${r}: ${K} aerodynamic orders. Edits apply to the draft; cells that differ from the applied render are outlined.`;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < K; i++) {
      const k = i + 1;
      const tr = el("tr", {}, el("th", { scope: "row", text: String(k) }));
      for (const f of PED_FIELDS) {
        const input = el("input", {
          type: "text", inputmode: "decimal", spellcheck: "false", autocomplete: "off",
          id: `md2r-ped-${r}-${f.key}-${k}`,
          value: fmt(ped[f.key][i]),
          "data-regime": r, "data-group": "ped", "data-key": f.key, "data-idx": String(i),
          "aria-label": `${r} order ${k} ${f.long} (${f.unit})`,
        });
        input.title = titleFor(r, f.key, i);
        tr.append(el("td", {}, input));
      }
      tr.append(el("td", { class: "md2r-derived" }));
      frag.append(tr);
    }
    tbody.replaceChildren(frag);
    for (let i = 0; i < K; i++) updateRowDerived(r, i);
    refreshStatus();
  }

  function syncShaft(r) {
    for (const f of SHAFT_FIELDS) {
      const input = editor.querySelector(`#md2r-${r}-shaft-${f.key}`);
      input.value = fmt(S.draft[r].shaft[f.key]);
      input.removeAttribute("aria-invalid");
      input.title = `fitted value ${fmt(S.original[r].shaft[f.key])} ${f.unit}`;
      S.invalid.delete(input.id);
      const msg = editor.querySelector(`#md2r-${r}-shaft-${f.key}-msg`);
      if (msg) msg.textContent = "";
      markPending(input, S.draft[r].shaft[f.key], S.applied[r].shaft[f.key]);
    }
    updateShaftReadout(r);
  }

  function syncEditor() {
    REGIMES.forEach(syncShaft);
    buildTable();
    syncAmplitudeEditor();
    if (!S.advDirty) loadAdvanced(false);
  }

  function resetBulkRange() {
    const K = S.draft[S.tableRegime].pedestal.sigma2_amp.length;
    bulk.from.max = String(K);
    bulk.to.max = String(K);
    bulk.from.value = "1";
    bulk.to.value = String(K);
    bulk.msg.textContent = "";
  }

  function applyBulk() {
    if (S.busy) return;
    const r = S.tableRegime;
    const K = S.draft[r].pedestal.sigma2_amp.length;
    const from = Number(bulk.from.value);
    const to = Number(bulk.to.value);
    const fail = (m) => {
      bulk.msg.textContent = m;
      bulk.msg.classList.add("md2r-bad");
    };
    bulk.msg.classList.remove("md2r-bad");
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 1 || to > K || from > to)
      return fail(`Order range must be integers with 1 ≤ from ≤ to ≤ ${K}.`);
    const op = bulk.op.value;
    const res = parseNumber(bulk.value.value, "nonneg");
    if (res.error)
      return fail(op === "mul" ? `Factor ${res.error} (it must keep the values non-negative).` : `Value ${res.error}.`);
    const f = PED_BY_KEY[bulk.field.value];
    const arr = S.draft[r].pedestal[f.key];
    const parity = bulk.parity.value;
    let n = 0;
    for (let k = from; k <= to; k++) {
      if (parity === "even" && k % 2) continue;
      if (parity === "odd" && !(k % 2)) continue;
      arr[k - 1] = op === "set" ? res.value : arr[k - 1] * res.value;
      n++;
    }
    if (!n) return fail("No order in that range matches the parity.");
    buildTable();
    if (!S.advDirty) loadAdvanced(false);
    const parityTxt = parity === "all" ? "" : ` ${parity}`;
    bulk.msg.textContent = `${op === "set" ? "Set" : "Multiplied"} ${f.long} ${op === "set" ? "to" : "by"} ${fmt(res.value)}${op === "set" ? " " + f.unit : ""} on ${n}${parityTxt} orders k = ${from}…${to} of ${r}. Pending until Re-render.`;
    refreshStatus();
  }

  function loadAdvanced(force) {
    if (!force && S.advDirty) return;
    advText.value = prettyJSON(S.draft[advRegimeSel.value]);
    S.advDirty = false;
    if (force) advMsg.textContent = `Loaded the current ${advRegimeSel.value} draft.`;
  }

  function applyAdvanced() {
    if (S.busy) return;
    const r = advRegimeSel.value;
    advMsg.classList.remove("md2r-bad");
    let obj;
    try {
      obj = JSON.parse(advText.value);
    } catch (err) {
      advMsg.textContent = `Invalid JSON: ${err.message}`;
      advMsg.classList.add("md2r-bad");
      return;
    }
    const errs = structureErrors(S.draft[r], obj, "", []);
    const ped = obj && obj.pedestal;
    if (!errs.length && ped)
      for (const f of PED_FIELDS)
        ped[f.key].forEach((v, i) => {
          if (v < 0 && errs.length < 12) errs.push(`pedestal.${f.key}[${i}]: must be ≥ 0`);
        });
    if (!errs.length && obj.shaft)
      for (const f of SHAFT_FIELDS) {
        if (f.readonly) {
          if (obj.shaft[f.key] !== S.original[r].shaft[f.key])
            errs.push(`shaft.${f.key} is read-only and must stay ${fmt(S.original[r].shaft[f.key])} ${f.unit}`);
          continue;
        }
        const chk = parseNumber(String(obj.shaft[f.key]), f.rule);
        if (chk.error) errs.push(`shaft.${f.key}: ${chk.error}`);
      }
    if (errs.length) {
      advMsg.textContent = `Not applied. ${errs.join("; ")}`;
      advMsg.classList.add("md2r-bad");
      return;
    }
    const changed = diffPaths(S.draft[r], obj, [r], []).length;
    S.draft[r] = obj;
    S.advDirty = false;
    REGIMES.forEach(syncShaft);
    buildTable();
    syncAmplitudeEditor();
    loadAdvanced(false);
    advMsg.textContent = `Applied to the ${r} draft: ${changed} value${changed === 1 ? "" : "s"} changed. Pending until Re-render.`;
    refreshStatus();
  }

  // ---------- behaviour: render / reset / export
  let ticker = null;

  function setBusy(on) {
    S.busy = on;
    editor.disabled = on;
    app.setAttribute("aria-busy", on ? "true" : "false");
    P.edited.node.classList.toggle("md2r-busy", on);
    if (on) {
      const t0 = performance.now();
      statusEl.textContent = `Rendering in the backend… The edited panel still shows ${appliedName()}.`;
      ticker = setInterval(() => {
        statusEl.textContent = `Rendering in the backend… ${((performance.now() - t0) / 1000).toFixed(1)} s. The edited panel still shows ${appliedName()}.`;
      }, 200);
    } else if (ticker) {
      clearInterval(ticker);
      ticker = null;
    }
    refreshStatus();
  }

  function appliedName() {
    return S.renderCount && S.edited.source === "render" ? `render #${S.renderCount}` : "the fitted baseline";
  }

  async function doRender() {
    if (S.busy) return;
    if (S.invalid.size) return;
    if (S.advDirty) {
      showError("The advanced JSON box has unapplied edits. Apply them to the draft or press Reload from draft before rendering.");
      return;
    }
    clearError();
    info("");
    const sent = clone(S.draft);
    const t0 = performance.now();
    setBusy(true);
    try {
      const res = await api("/api/render", {
        method: "POST",
        body: { parameters: sent, shaft_law_evaluation: ex.shaft_law_evaluation },
      });
      checkParams(res.parameters, "POST /api/render");
      checkAudio(res.audio, "POST /api/render");
      if (typeof res.playback_gain !== "number") throw new Error("POST /api/render: playback_gain missing");
      if (!res.export_fits || !REGIMES.every((r) => res.export_fits[r]))
        throw new Error("POST /api/render: export_fits missing a regime");
      const edited = signalFrom(res.edited, "edited");
      if (edited.spec.nb !== baseline.spec.nb || edited.spec.nf !== baseline.spec.nf)
        throw new Error(`POST /api/render: edited spectrogram ${edited.spec.nb}×${edited.spec.nf} does not match baseline ${baseline.spec.nb}×${baseline.spec.nf}`);
      const normalised = paramDiff(sent, res.parameters);
      S.renderCount += 1;
      S.edited = { ...edited, source: "render" };
      S.applied = clone(res.parameters);
      S.draft = clone(res.parameters);
      S.audio = { ...res.audio };
      S.gain = res.playback_gain;
      S.exports = { fits: res.export_fits, render: S.renderCount };
      S.lastElapsed = typeof res.elapsed_s === "number" ? res.elapsed_s : null;
      S.lastRoundtrip = (performance.now() - t0) / 1000;
      S.advDirty = false;
      setAudio();
      syncEditor();
      drawEdited();
      drawSpectrum();
      info(
        normalised.length
          ? `Render #${S.renderCount} done. The backend's effective parameters differ from those sent in ${normalised.length} value(s), e.g. ${pathLabel(normalised[0])}; the editor now shows the effective values.`
          : `Render #${S.renderCount} done.`
      );
    } catch (err) {
      showError(`${err.message} The edited panel still shows ${appliedName()}; your draft is kept as pending.`);
    } finally {
      setBusy(false);
    }
  }

  function discardPending() {
    if (S.busy) return;
    S.draft = clone(S.applied);
    S.invalid.clear();
    S.advDirty = false;
    syncEditor();
    info(`Pending edits discarded; the draft equals the parameters of ${appliedName()}.`);
  }

  function resetAll() {
    if (S.busy) return;
    const touched = paramDiff(S.draft, S.original).length || S.renderCount;
    if (touched && !window.confirm("Reset all parameters to the fitted baseline and show the baseline render again? Unsaved edits and the latest render are dropped."))
      return;
    S.draft = clone(S.original);
    S.applied = clone(S.original);
    S.edited = { ...baseline, source: "baseline" };
    S.audio = { ...state.audio };
    S.gain = state.playback_gain;
    S.exports = null;
    S.renderCount = 0;
    S.lastElapsed = null;
    S.lastRoundtrip = null;
    S.invalid.clear();
    S.advDirty = false;
    clearError();
    setAudio();
    syncEditor();
    drawEdited();
    drawSpectrum();
    info("Reset to the fitted baseline parameters and the frozen baseline render.");
  }

  function doDownload(r) {
    if (!S.exports) return;
    downloadJSON(`md2_fly125_${r}__interactive_edit__render${S.exports.render}.json`, S.exports.fits[r]);
  }

  // ---------- behaviour: status
  let statusFrame = 0;
  function refreshStatus() {
    if (statusFrame) return;
    statusFrame = requestAnimationFrame(() => {
      statusFrame = 0;
      updateStatusNow();
    });
  }

  function updateStatusNow() {
    const pending = paramDiff(S.draft, S.applied);
    const fromFit = paramDiff(S.applied, S.original).length;
    const nInvalid = S.invalid.size;
    renderBtn.disabled = S.busy || nInvalid > 0;
    discardBtn.disabled = S.busy || (!pending.length && !nInvalid);
    resetBtn.disabled = S.busy;
    for (const r of REGIMES) dlBtns[r].disabled = S.busy || !S.exports;

    if (!S.busy) {
      statusEl.textContent = nInvalid
        ? `${nInvalid} invalid field${nInvalid === 1 ? "" : "s"}: fix ${nInvalid === 1 ? "it" : "them"} to enable Re-render.`
        : pending.length
          ? `Ready. ${pending.length} pending edit${pending.length === 1 ? "" : "s"}: press Re-render to synthesise them.`
          : "Ready. No pending edits.";
    }
    pendingSummary.textContent = pending.length
      ? `Pending (not rendered): ${pending.length} value${pending.length === 1 ? " differs" : "s differ"} from the applied parameters.`
      : "Pending: none. The draft equals the applied parameters.";
    pendingList.replaceChildren(
      ...pending.slice(0, MAX_LISTED_CHANGES).map((p) => el("li", { text: pathLabel(p) })),
      ...(pending.length > MAX_LISTED_CHANGES ? [el("li", { text: `… and ${pending.length - MAX_LISTED_CHANGES} more` })] : [])
    );
    pendingDetails.hidden = !pending.length;

    const timing = S.lastElapsed === null ? "" : ` Backend render time ${S.lastElapsed.toFixed(2)} s (round trip ${S.lastRoundtrip.toFixed(2)} s).`;
    appliedSummary.textContent =
      S.edited.source === "render"
        ? `Applied: the edited panel shows render #${S.renderCount}, whose parameters differ from the fitted baseline in ${fromFit} value${fromFit === 1 ? "" : "s"}.${timing}`
        : "Applied: the edited panel shows the fitted baseline (nothing re-rendered yet).";
    exportNote.textContent = S.exports
      ? `Downloads are copies of the source payloads carrying the parameters of render #${S.exports.render}, labelled as interactive edits, not fitted results.${pending.length ? " Pending edits are not included." : ""}`
      : "Downloads become available after a successful Re-render. Source fit files are never modified.";

    staleBadge.hidden = !pending.length;
    staleBadge.textContent = `Showing ${appliedName()}. ${pending.length} pending edit${pending.length === 1 ? " is" : "s are"} not in this panel.`;
    P.edited.heading.textContent = S.edited.source === "render" ? `Edited render #${S.renderCount}` : "Edited render (= fitted baseline)";
    P.edited.caption.textContent = signalCaption(S.edited) +
      (S.edited.sha === baseline.sha ? " Waveform identical to the baseline." : " Waveform differs from the baseline.");
  }

  function signalCaption(sig) {
    return `waveform sha256 ${sig.sha}, RMS ${fmtShort(sig.rms, 5)} (before playback gain).`;
  }

  // ---------- behaviour: plots and audio
  function setAudio() {
    for (const k of ["real", "baseline", "edited"]) {
      P[k].audio.pause();
      P[k].audio.src = S.audio[k];
    }
    gainEl.textContent = `Shared playback gain ${fmtShort(S.gain, 5)} (linear) applied to all three players, so their loudness compares directly. Spectrograms are in absolute units, before this gain.`;
  }

  function heatLayout() {
    return {
      height: 300,
      margin: { l: 58, r: 12, t: 30, b: 42 },
      xaxis: { title: { text: "time (s)" } },
      yaxis: { title: { text: "frequency (Hz)" }, range: [0, S.view.fmax] },
      paper_bgcolor: "rgba(0,0,0,0)",
    };
  }

  function drawHeat(div, spec) {
    let n = 0;
    while (n < spec.nb && spec.freqs[n] <= S.view.fmax) n++;
    n = Math.min(spec.nb, n + 1);
    const trace = {
      type: "heatmap",
      x: spec.times,
      y: spec.freqs.slice(0, n),
      z: spec.rows.slice(0, n),
      zmin: vmax - S.view.dr,
      zmax: vmax,
      zauto: false,
      colorscale: MAGMA_COLORS,
      colorbar: { title: { text: "dB" }, thickness: 10 },
      hovertemplate: "t %{x:.2f} s<br>f %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
    };
    Plotly.react(div, [trace], heatLayout(), { responsive: true, displaylogo: false });
  }

  function drawEdited() {
    drawHeat(P.edited.plot, S.edited.spec);
  }

  function drawSpectrum() {
    if (!S.view.spectrum) return;
    const mk = (sig, name, line) => {
      let n = 0;
      while (n < sig.spec.nb && sig.spec.freqs[n] <= S.view.fmax) n++;
      return { type: "scatter", mode: "lines", name, x: sig.spec.freqs.slice(0, n), y: sig.spec.meanDb.slice(0, n), line };
    };
    const traces = [
      mk(real, "real", { color: "#222", width: 1.2 }),
      mk(baseline, "baseline", { color: "#888", width: 1.2, dash: "dot" }),
      mk(S.edited, S.edited.source === "render" ? `edited #${S.renderCount}` : "edited (= baseline)", { color: "#c2410c", width: 1.4 }),
    ];
    Plotly.react(spectrumPlot, traces, {
      height: 320,
      margin: { l: 58, r: 12, t: 10, b: 42 },
      xaxis: { title: { text: "frequency (Hz)" }, range: [0, S.view.fmax] },
      yaxis: { title: { text: "mean power (dB)" } },
      legend: { orientation: "h", y: 1.12 },
      paper_bgcolor: "rgba(0,0,0,0)",
    }, { responsive: true, displaylogo: false });
  }

  function drawAll() {
    drOut.value = `${S.view.dr} dB (colour ${fmtShort(vmax - S.view.dr, 4)} to ${fmtShort(vmax, 4)} dB)`;
    drInput.setAttribute("aria-valuetext", `${S.view.dr} decibels`);
    drawHeat(P.real.plot, real.spec);
    drawHeat(P.baseline.plot, baseline.spec);
    drawEdited();
    drawSpectrum();
  }

  // ---------- initial fill (after the node is in the DOM, Plotly needs layout)
  P.real.caption.textContent = signalCaption(real);
  P.baseline.caption.textContent = signalCaption(baseline);
  setAudio();
  resetBulkRange();
  syncEditor();
  const firstDraw = () => {
    if (!app.isConnected) {
      requestAnimationFrame(firstDraw);
      return;
    }
    drawAll();
    updateStatusNow();
  };
  requestAnimationFrame(firstDraw);
  return app;
}
