// BeltWatch review UI. Vanilla JS, no dependencies. Server data is inserted
// with textContent only (never innerHTML) so uploaded filenames cannot inject markup.

const $ = (id) => document.getElementById(id);
const MATERIALS = [
  ["cardboard", "Cardboard"],
  ["soft_plastic", "Soft plastic"],
  ["rigid_plastic", "Rigid plastic"],
  ["metal", "Metal"],
];
const ROUTE_TEXT = {
  manual_review: "Manual review: a large part of the region is uncertain.",
  review: "Review: suspected target material, or a random audit of a low-risk image.",
  retake_image: "Retake image: the input is unusable (blur or exposure).",
  none: "No review required. This does not certify that the belt is clean.",
};
const TERMINAL = new Set(["complete", "partial", "failed"]);

const state = {
  files: [],
  firstImage: null,
  points: [], // normalized [x, y]
  auditId: null,
  selected: null,
  view: "overlay",
  poll: null,
};

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.error) message = `${body.error.message} (${body.error.code})`;
    } catch { /* not JSON */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

// --- model panel -------------------------------------------------------------

async function loadModel() {
  const button = $("model-button");
  try {
    const model = await api("/v1/model");
    button.textContent = `Model: ${model.version}`;
    const details = $("model-details");
    details.replaceChildren();
    const rows = [
      ["Version", model.version],
      ["Kind", model.model_kind],
      ["Preprocessing", model.preprocessing_version],
      ["Calibration", model.calibration],
      ["Review policy", model.review_policy.version],
      ["Measurement", model.measurement],
      ["Checksum", model.model_sha256 || "–"],
    ];
    if (model.provenance?.note) rows.unshift(["Note", model.provenance.note]);
    if (model.evaluation) {
      rows.push(["Evaluated on", `${model.evaluation.split} (${model.evaluation.n_images} images)`]);
      rows.push(["Foreground macro IoU", model.evaluation.foreground_macro_iou?.toFixed(3) ?? "–"]);
      rows.push(["Coverage MAE", `${model.evaluation.coverage_mae_pp?.toFixed(2) ?? "–"} pp`]);
    } else {
      rows.push(["Evaluation", "no evaluation report bundled with this release"]);
    }
    for (const [k, v] of rows) details.append(el("dt", { text: k }), el("dd", { text: String(v) }));
    $("model-limitations").replaceChildren(...model.limitations.map((t) => el("li", { text: t })));
  } catch (error) {
    button.textContent = "Model unavailable";
    button.title = error.message;
  }
}

$("model-button").addEventListener("click", () => $("model-dialog").showModal());

// --- upload and region ---------------------------------------------------------

function setFiles(fileList) {
  state.files = [...fileList].filter((f) => /^image\/(png|jpeg|webp)$/.test(f.type));
  $("file-list").replaceChildren(
    ...state.files.map((f) => el("li", { text: `${f.name} · ${(f.size / 1e6).toFixed(1)} MB` })),
  );
  $("submit").disabled = state.files.length === 0;
  state.points = [];
  if (state.files.length) {
    const img = new Image();
    img.onload = () => {
      state.firstImage = img;
      drawRegion();
    };
    img.src = URL.createObjectURL(state.files[0]);
  } else {
    state.firstImage = null;
    drawRegion();
  }
}

$("files").addEventListener("change", (e) => setFiles(e.target.files));
const dropzone = $("dropzone");
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  setFiles(e.dataTransfer.files);
});

function drawRegion() {
  const canvas = $("region-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("region-empty").classList.toggle("hidden", Boolean(state.firstImage));
  if (state.firstImage) ctx.drawImage(state.firstImage, 0, 0, canvas.width, canvas.height);
  const pts = state.points.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  if (pts.length) {
    ctx.fillStyle = "rgba(67, 181, 151, 0.25)";
    ctx.strokeStyle = "#43b597";
    ctx.lineWidth = 3;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (pts.length >= 3) ctx.closePath();
    if (pts.length >= 3) ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#43b597";
    for (const [x, y] of pts) ctx.fillRect(x - 4, y - 4, 8, 8);
  }
  $("region-status").textContent =
    state.points.length >= 3 ? `Region: ${state.points.length} corners`
      : state.points.length ? `Add ${3 - state.points.length} more corner(s)` : "Region: full frame";
}

$("region-canvas").addEventListener("click", (e) => {
  if (!state.firstImage) return;
  const rect = e.target.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const y = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
  state.points.push([Number(x.toFixed(4)), Number(y.toFixed(4))]);
  drawRegion();
});
$("region-full").addEventListener("click", () => { state.points = []; drawRegion(); });
$("region-clear").addEventListener("click", () => { state.points = []; drawRegion(); });

$("submit").addEventListener("click", async () => {
  const error = $("upload-error");
  error.textContent = "";
  if (state.points.length > 0 && state.points.length < 3) {
    error.textContent = "An inspection region needs at least 3 corners (or use Full frame).";
    return;
  }
  const options = {
    camera_id: $("camera-id").value.trim() || "demo-camera",
    training_consent: $("consent").checked,
  };
  if (state.points.length >= 3) options.inspection_region = state.points;
  const form = new FormData();
  state.files.forEach((f) => form.append("files", f, f.name));
  form.append("options", JSON.stringify(options));
  $("submit").disabled = true;
  try {
    const created = await api("/v1/audits", {
      method: "POST",
      body: form,
      headers: { "Idempotency-Key": crypto.randomUUID() },
    });
    openAudit(created.audit_id);
  } catch (e) {
    error.textContent = e.message;
    $("submit").disabled = false;
  }
});

// --- audit view ----------------------------------------------------------------

function openAudit(auditId) {
  state.auditId = auditId;
  state.selected = null;
  history.replaceState(null, "", `#audit=${auditId}`);
  $("upload-view").classList.add("hidden");
  $("audit-view").classList.remove("hidden");
  $("detail").classList.add("hidden");
  $("report-json").href = `/v1/audits/${auditId}/report`;
  $("report-csv").href = `/v1/audits/${auditId}/report?format=csv`;
  clearInterval(state.poll);
  refreshAudit();
  state.poll = setInterval(refreshAudit, 1000);
}

function imageButton(img) {
  const badgeText = img.status === "complete" ? img.route : img.status;
  const button = el(
    "button",
    {
      type: "button",
      class: `image-item${state.selected === img.image_id ? " selected" : ""}`,
      onclick: () => selectImage(img.image_id),
    },
    el("span", { class: "name", text: img.filename }),
    el("span", { class: `badge ${badgeText}`, text: badgeText.replace("_", " ") }),
    el("span", {
      class: "hint",
      text: img.status === "complete"
        ? `target coverage ${pct(img.total_target_coverage)} · priority ${img.priority?.toFixed(3)}`
        : img.error || "waiting for the worker",
    }),
  );
  return el("li", {}, button);
}

async function refreshAudit() {
  if (!state.auditId) return;
  let audit;
  try {
    audit = await api(`/v1/audits/${state.auditId}`);
  } catch (e) {
    $("audit-status").textContent = e.message;
    clearInterval(state.poll);
    return;
  }
  const { total, complete, failed } = audit.progress;
  const done = complete + failed;
  $("audit-title").textContent = `Audit ${audit.audit_id}`;
  $("audit-meta").textContent =
    `Camera ${audit.camera_id} · model ${audit.model_version} · policy ${audit.review_policy_version} · ` +
    `created ${new Date(audit.created_at).toLocaleString()} · uploads expire ${new Date(audit.expires_at).toLocaleString()}`;
  $("progress-bar").style.width = `${total ? (100 * done) / total : 0}%`;
  $("audit-status").textContent =
    `${audit.status} · ${complete} complete, ${failed} failed, ${audit.progress.pending} pending` +
    (audit.error ? ` · ${audit.error}` : "");

  const byId = Object.fromEntries(audit.images.map((i) => [i.image_id, i]));
  const queued = audit.review_queue.map((id) => byId[id]);
  const others = audit.images.filter((i) => !audit.review_queue.includes(i.image_id));
  $("queue").replaceChildren(...(queued.length ? queued.map(imageButton) : [el("li", { class: "hint", text: "Nothing flagged yet." })]));
  $("others").replaceChildren(...(others.length ? others.map(imageButton) : [el("li", { class: "hint", text: "None." })]));

  if (!state.selected && queued.length) selectImage(queued[0].image_id);
  if (TERMINAL.has(audit.status)) clearInterval(state.poll);
}

async function selectImage(imageId) {
  state.selected = imageId;
  const detail = await api(`/v1/audits/${state.auditId}/images/${imageId}`);
  $("detail").classList.remove("hidden");
  $("detail-title").textContent = detail.filename;
  state.detail = detail;
  showView(detail.status === "complete" ? state.view : "original");

  const rows = $("coverage-rows");
  rows.replaceChildren();
  if (detail.status === "complete") {
    for (const [key, label] of MATERIALS) {
      const value = detail.visible_coverage[key];
      const bar = el("div", { class: "bar" }, el("i", { style: `width:${Math.min(100, value * 100 * 4)}%;background:var(--${key})` }));
      rows.append(el("tr", {}, el("td", { text: label }), el("td", {}, bar), el("td", { text: pct(value) })));
    }
    rows.append(el("tr", {}, el("td", { text: "Total target" }), el("td"), el("td", { text: pct(detail.total_target_coverage) })));
    $("detail-route").textContent = `${ROUTE_TEXT[detail.route]} Reasons: ${detail.review_reasons.join(", ") || "none"}.`;
    const issues = detail.quality.issues.length ? `quality issues: ${detail.quality.issues.join(", ")}` : "input quality OK";
    $("detail-notes").textContent =
      `Uncertain pixels in region: ${pct(detail.uncertain_pixel_fraction)} · ${issues} · ` +
      `processed in ${detail.processing_seconds}s${detail.cache_hit ? " (cached result)" : ""}`;
  } else {
    $("detail-route").textContent = detail.error ? `Failed: ${detail.error}` : "Waiting for the worker…";
    $("detail-notes").textContent = "";
  }
  $("feedback-form").classList.toggle("hidden", detail.status !== "complete");
  renderHistory(detail.feedback);
  refreshAudit();
}

function showView(view) {
  state.view = view;
  const d = state.detail;
  const urls = { overlay: d.overlay_url, original: d.original_url, uncertainty: d.uncertainty_url };
  $("detail-image").src = urls[view] || d.original_url;
  for (const b of document.querySelectorAll(".toggle button")) {
    b.classList.toggle("active", b.dataset.view === view);
    b.disabled = d.status !== "complete" && b.dataset.view !== "original";
  }
}

for (const b of document.querySelectorAll(".toggle button")) {
  b.addEventListener("click", () => showView(b.dataset.view));
}

function renderHistory(feedback) {
  $("history").replaceChildren(
    ...(feedback.length
      ? feedback.map((f) => el("li", {
        text: `${new Date(f.created_at).toLocaleString()} · ${f.decision}` +
          (f.materials_present.length ? ` · present: ${f.materials_present.join(", ")}` : "") +
          (f.reviewer ? ` · ${f.reviewer}` : "") + (f.note ? ` · “${f.note}”` : ""),
      }))
      : [el("li", { text: "No decisions yet." })]),
  );
}

$("feedback-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = {
    image_id: state.selected,
    decision: form.decision.value,
    materials_present: [...form.querySelectorAll("input[name=materials]:checked")].map((i) => i.value),
  };
  const reviewer = $("reviewer").value.trim();
  const note = $("note").value.trim();
  if (reviewer) body.reviewer = reviewer;
  if (note) body.note = note;
  try {
    await api(`/v1/audits/${state.auditId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("feedback-status").textContent = "Saved.";
    form.reset();
    $("reviewer").value = reviewer;
    selectImage(state.selected);
  } catch (error) {
    $("feedback-status").textContent = error.message;
  }
});

$("delete-audit").addEventListener("click", async () => {
  if (!confirm("Delete this audit, its uploads, and its results?")) return;
  await api(`/v1/audits/${state.auditId}`, { method: "DELETE" });
  resetToUpload();
});
$("new-audit").addEventListener("click", resetToUpload);

function resetToUpload() {
  clearInterval(state.poll);
  state.auditId = null;
  history.replaceState(null, "", location.pathname);
  $("audit-view").classList.add("hidden");
  $("upload-view").classList.remove("hidden");
  $("files").value = "";
  setFiles([]);
}

loadModel();
drawRegion();
const match = location.hash.match(/audit=([\w-]+)/);
if (match) openAudit(match[1]);
