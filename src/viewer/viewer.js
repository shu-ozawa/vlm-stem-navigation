// The viewer page's behaviour (viewer.html). API: app.py.

const $ = id => document.getElementById(id);

// ------------------------------------------------------------ state
let META = null;       // /api/meta, plus the episode's specimen size and targets (set by reset)
let STATE = null;      // the last /api/reset or /api/step answer
let running = false;   // Start pressed and not paused
let inflight = false;  // a step request is on its way
let ovImg = null;      // the overview image of the episode
const TRACES = {};     // step number -> the model's decision of that step

// ------------------------------------------------------------ API
async function api(path, body) {
  const options = body === undefined ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, options);
  const d = await r.json().catch(() => ({ error: "HTTP " + r.status }));
  if (!r.ok) throw new Error(d.error || d.detail || ("HTTP " + r.status));
  return d;
}

function settings() {
  const dose = $("c-dose").value;
  return {
    task: $("c-task").value, seed: +$("c-seed").value, start_mode: $("c-mode").value,
    dose: dose ? +dose : null,
    reader: $("c-reader").value,
  };
}

// ------------------------------------------------------------ status bar
function setStatus(text, kind) {
  $("status-text").textContent = text;
  $("status-text").title = text;
  $("dot").className = "dot" + (kind ? " " + kind : "");
}

function setStartLabel(on) {
  $("btn-start-label").textContent = on ? "Pause" : "Start";
  $("btn-start").classList.toggle("is-running", on);
}

// how an episode ended
function outcome(info) {
  if (info.success) return ["Success", "ok"];
  if (info.truncated) return ["Failed: step limit", "err"];
  // at the highest magnification and in the square, but not in the required orientation
  if (info.zoom_index === 0 && info.target_in_center && info.orientation_ok === false)
    return [`Failed: found ${info.orientation_dev_deg.toFixed(0)}° from the required orientation `
            + `(allowed ±${info.orientation_tol_deg}°)`, "err"];
  return ["Failed: found outside the success square", "err"];
}

// ------------------------------------------------------------ map of the whole specimen
function drawOverview() {
  const c = $("overview"), s = ovImg.width / META.W;
  c.width = ovImg.width;
  c.height = ovImg.height;
  const x = c.getContext("2d");
  x.drawImage(ovImg, 0, 0);

  // the path through the FOV centres (a jump across half the specimen is not drawn as a line)
  const hist = STATE.history;
  x.strokeStyle = "rgba(250,204,21,.9)";
  x.lineWidth = 2;
  x.beginPath();
  let prev = null;
  for (const h of hist) {
    const jump = !prev || Math.abs(h.center_row - prev.center_row) >= META.H / 2 || Math.abs(h.center_col - prev.center_col) >= META.W / 2;
    if (jump) x.moveTo(h.center_col * s, h.center_row * s);
    else x.lineTo(h.center_col * s, h.center_row * s);
    prev = h;
  }
  x.stroke();

  // the start
  x.fillStyle = "#facc15";
  x.beginPath();
  x.arc(hist[0].center_col * s, hist[0].center_row * s, 5, 0, 7);
  x.fill();

  // the field of view and the green square
  const cur = STATE.info, F = cur.fov_size * s, q = F * META.box_frac;
  x.save();
  x.translate(cur.center_col * s, cur.center_row * s);
  x.strokeStyle = "#38bdf8";
  x.lineWidth = 2.5;
  x.strokeRect(-F / 2, -F / 2, F, F);
  x.setLineDash([4, 3]);
  x.strokeStyle = "rgba(34,197,94,.9)";
  x.lineWidth = 1.5;
  x.strokeRect(-q / 2, -q / 2, q, q);
  x.restore();

  // the target (ground truth, viewer only): a circle for an atom, a dashed line for a boundary
  if (!$("c-truth").checked) return;
  x.strokeStyle = "#ef4444";
  x.lineWidth = 2;
  for (const m of META.markers) {
    x.beginPath();
    if (m.line) {
      x.setLineDash([6, 4]);
      x.moveTo(m.line[0][1] * s, m.line[0][0] * s);
      x.lineTo(m.line[1][1] * s, m.line[1][0] * s);
      x.stroke();
      x.setLineDash([]);
    } else {
      x.arc(m.rc[1] * s, m.rc[0] * s, 9, 0, 7);
      x.stroke();
    }
  }
}

// ------------------------------------------------------------ what the model sees
// the green square exactly as drawn on the model's images: a fixed fraction of the image at every zoom
function placeCenterBox() {
  const frac = META.box_frac, a = (1 - frac) / 2 * 100;
  Object.assign($("center-box").style, { left: a + "%", top: a + "%", width: frac * 100 + "%", height: frac * 100 + "%" });
}

function render(d) {
  STATE = d;
  $("obs").src = d.obs;
  $("m-step").textContent = `${d.info.step} / ${d.max_steps}`;

  const o = d.operator;
  const level = o.zoom_index === 0 ? "highest" : o.zoom_index === o.n_zoom - 1 ? "lowest" : `${o.zoom_index + 1} of ${o.n_zoom}`;
  $("m-zoom").textContent = `FOV ${o.fov_A.toFixed(0)} Å (${level})`;  // as the model is told
  drawOverview();
}

// ------------------------------------------------------------ how the model decides
// the order of the answers in the bar charts: directions as on a map, around "inside"
const LABEL_ORDER = {
  locate: ["up", "left", "inside", "right", "down", "none"],
};
const NUM = ["①", "②", "③", "④", "⑤", "⑥"];

const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const truthHas = (t, l) => Array.isArray(t) ? t.includes(l) : t === l;
const truthStr = t => Array.isArray(t) ? t.join(" / ") : t;

// the messages sent for a question, as text
function messagesText(messages) {
  return messages.map(m => `SYSTEM\n${m.system}\n\nUSER\n${m.user}\n\nMODEL\n${m.answer}`).join("\n\n----------\n\n");
}

// one card: the question, its image, the probability of each answer, its prompt and reasoning
function stageCard(st, i, images, showTruth) {
  const part = st.parts[0];

  const bars = LABEL_ORDER[st.id].map(l => {
    const p = st.probs[l], chosen = l === st.choice, correct = showTruth && truthHas(st.truth, l);
    return `<div class="lrow${chosen ? " chosen" : ""}${correct ? " truth" : ""}">
      <div class="lab mono">${l}</div>
      <div class="bar"><div class="bar-fill${chosen ? " chosen" : ""}" style="width:${(p * 100).toFixed(1)}%"></div></div>
      <div class="pct mono">${(p * 100).toFixed(1)}%</div></div>`;
  }).join("");

  let badge = "";
  if (showTruth)
    badge = st.truth == null ? '<span class="badge na">no truth</span>'
      : st.correct ? `<span class="badge ok">✔ truth: ${truthStr(st.truth)}</span>`
      : `<span class="badge ng">✘ truth: ${truthStr(st.truth)}</span>`;

  const stats = st.image_stats && `<div class="muted mono">histogram: mean ${st.image_stats.mean.toFixed(0)},
    sd ${st.image_stats.sd.toFixed(0)}, p1 ${st.image_stats.p1.toFixed(0)}, p99 ${st.image_stats.p99.toFixed(0)},
    saturated ${st.image_stats.saturated_pct.toFixed(1)}%, black ${st.image_stats.black_pct.toFixed(1)}%</div>`;
  const reasoning = part.reasoning &&
    `<details><summary>Reasoning (${part.usage.completion_tokens} tokens)</summary><pre>${esc(part.reasoning)}</pre></details>`;

  return `<div class="stage${showTruth && st.correct === false ? " wrong" : ""}">
    <div class="stage-top"><span class="qnum">${NUM[i]}</span><b class="mono">${st.id}</b>
      ${badge}<span class="lat mono" title="time of this question's model call">${Math.round(part.latency_s * 1000)} ms</span></div>
    <div class="stage-body"><div class="thumb"><img src="${images[part.view]}" alt=""></div><div class="bars">${bars}</div></div>
    ${stats || ""}
    <details><summary>Prompt</summary><pre>${esc(messagesText(part.messages))}</pre></details>
    ${reasoning || ""}
  </div>`;
}

function renderTrace(stepNo) {
  document.querySelectorAll("#log div").forEach(el => el.classList.toggle("sel", el.dataset.step === String(stepNo)));
  const dec = TRACES[stepNo], tr = dec.trace, showTruth = $("c-gt").checked;

  let html = tr.stages.map((st, i) => stageCard(st, i, dec.trace_images, showTruth)).join('<div class="arrow">↓</div>');
  if (tr.vector)
    html += `<div class="arrow">↓</div><div class="axisbox"><b>weighted direction</b> — ${esc(tr.vector.rule)}
      <div class="mono">dx = ${tr.vector.dx.toFixed(3)} (right +), dy = ${tr.vector.dy.toFixed(3)} (down +)</div></div>`;
  $("trace").innerHTML = html;
}

// ------------------------------------------------------------ all steps
function logLine(text, stepNo) {
  const el = $("log");
  el.insertAdjacentHTML("beforeend", `<div data-step="${stepNo ?? ""}">${text}</div>`);
  el.scrollTop = el.scrollHeight;
}

// ------------------------------------------------------------ run
async function reset() {
  running = false;
  setStartLabel(false);
  try {
    const d = await api("/api/reset", settings());
    Object.assign(META, { H: d.H, W: d.W, markers: d.markers });
    $("c-model").textContent = d.model || "-";

    ovImg = new Image();
    ovImg.src = d.overview;
    await ovImg.decode();
    render(d);

    $("m-ms").textContent = "-";
    $("log").innerHTML = "";
    for (const key in TRACES) delete TRACES[key];
    $("trace").innerHTML = '<div class="muted">No step yet.</div>';
    logLine(`reset task=${$("c-task").value} model=${d.model} start=${$("c-mode").value} seed=${$("c-seed").value}`);

    const m = await api("/api/model");  // the model server and its decision server
    setStatus(m.message, m.model && (m.decision_server || $("c-reader").value !== "typed") ? "" : "err");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }
}

async function step() {
  if (inflight || !STATE || STATE.done) return;
  inflight = true;
  try {
    const d = await api("/api/step", {});
    TRACES[d.info.step] = d.decision;
    $("m-ms").textContent = `${Math.round(d.decision.latency_s * 1000)} ms / step`;  // the model calls of this step
    render(d);

    const nWrong = d.decision.trace.stages.filter(s => s.correct === false).length;
    logLine(`${String(d.info.step).padStart(2)} ${d.action_name}${d.info.blocked ? " (blocked)" : ""}`
            + `  · target ${d.info.distance_A.toFixed(1)} Å from centre`
            + (nWrong ? `  <span class="err-text">✘${nWrong}</span>` : ""), d.info.step);
    renderTrace(d.info.step);

    if (d.done) {
      running = false;
      setStartLabel(false);
      setStatus(...outcome(d.info));
    } else {
      setStatus(running ? "Running" : "Paused", running ? "live" : "");
    }
  } catch (e) {
    const end = ["stuck", "no_answer"].find(r => e.message.startsWith(r));  // run.py: result stuck / no_answer
    running = false;
    setStartLabel(false);
    setStatus(end ? `Failed: ${end.replace("_", " ")}` : e.message, "err");
    logLine(`<span class="err-text">${end ? "Failed: " : ""}${esc(e.message)}</span>`);
  }
  inflight = false;
}

async function loop() {
  while (running && STATE && !STATE.done) {
    await step();
    await new Promise(r => setTimeout(r, 0));  // no pause between steps: only let the page redraw
  }
}

// ------------------------------------------------------------ wiring
$("btn-start").onclick = async () => {
  if (!STATE || STATE.done) await reset();
  running = !running;
  setStartLabel(running);
  if (running) {
    setStatus("Running", "live");
    loop();
  } else {
    setStatus("Paused", "");
  }
};
$("btn-step").onclick = () => step();
$("btn-reset").onclick = () => reset();

$("c-truth").onchange = () => drawOverview();
$("c-gt").onchange = () => {
  const sel = document.querySelector("#log div.sel");
  if (sel) renderTrace(+sel.dataset.step);
};
$("log").onclick = e => {
  const el = e.target.closest("div[data-step]");
  if (el && el.dataset.step) renderTrace(+el.dataset.step);
};

// every setting starts a new episode
for (const id of ["c-task", "c-reader", "c-mode", "c-seed", "c-dose"])
  $(id).onchange = () => reset();

(async () => {
  META = await api("/api/meta");
  $("c-task").innerHTML = META.tasks.map(n => `<option value="${n}"${n === META.default_task ? " selected" : ""}>${n}</option>`).join("");
  placeCenterBox();
  await reset();
})();
