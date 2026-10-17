// BeltWatch usability-study session (docs/usability_study.md). Vanilla JS, no
// dependencies. Server data is inserted with textContent only (never innerHTML).
// The page never receives ground truth; it records answers and decision times.

const $ = (id) => document.getElementById(id);
const RESULTS_VERSION = 1;
const MATERIAL_NAMES = {
  cardboard: "Cardboard",
  soft_plastic: "Soft plastic",
  rigid_plastic: "Rigid plastic",
  metal: "Metal",
};
const INTRO = {
  unaided: [
    "In this part you see each conveyor photo on its own.",
    "For every image, decide whether it needs follow-up and tick the materials you can see.",
    "Work at your normal pace. The first two images are practice.",
  ],
  assisted: [
    "In this part BeltWatch shows its estimate: coloured overlays of suspected material, the estimated visible coverage, and suggested answers.",
    "The suggestions are already filled in. Check them against the image and change anything you disagree with.",
    "Work at your normal pace. The first two images are practice.",
  ],
};

const state = { plan: null, trials: [], index: 0, results: null, detail: null, t0: null, views: {} };

function show(id) {
  for (const view of ["setup-view", "intro-view", "trial-view", "done-view"]) {
    $(view).classList.toggle("hidden", view !== id);
  }
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${path}`);
  return response.json();
}

function storageKey(plan) {
  return `beltwatch-study:${plan.study}:${plan.participant}`;
}

function save() {
  try {
    localStorage.setItem(storageKey(state.plan), JSON.stringify(state.results));
  } catch { /* storage unavailable: the download at the end still works */ }
}

function download(results) {
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `results-${results.participant}${results.completed ? "" : "-incomplete"}.json`;
  document.body.append(link);
  link.click();
  link.remove();
}

function validatePlan(plan) {
  if (plan.plan_version !== 1) throw new Error("unsupported plan version");
  if (!Array.isArray(plan.blocks) || plan.blocks.length !== 2) throw new Error("a plan has two blocks");
  for (const block of plan.blocks) {
    if (!["unaided", "assisted"].includes(block.condition)) throw new Error("unknown condition");
  }
}

function buildTrials(plan) {
  const trials = [];
  plan.blocks.forEach((block, b) => {
    block.practice_image_ids.forEach((id, i) =>
      trials.push({ block: b, condition: block.condition, set: "practice", practice: true,
        audit_id: block.practice_audit_id, image_id: id, position: i }));
    block.image_ids.forEach((id, i) =>
      trials.push({ block: b, condition: block.condition, set: block.set, practice: false,
        audit_id: block.audit_id, image_id: id, position: i }));
  });
  return trials;
}

$("plan-file").addEventListener("change", async (event) => {
  $("setup-error").textContent = "";
  const file = event.target.files[0];
  if (!file) return;
  try {
    const plan = JSON.parse(await file.text());
    validatePlan(plan);
    state.plan = plan;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(storageKey(plan)) || "null"); } catch { saved = null; }
    if (saved) {
      const note = $("saved-session");
      note.replaceChildren(
        document.createTextNode(`${saved.completed ? "A complete" : "An incomplete"} session for ${plan.participant} is saved in this browser. `),
      );
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Download it";
      button.addEventListener("click", () => download(saved));
      const discard = document.createElement("button");
      discard.type = "button";
      discard.textContent = "Discard it";
      discard.addEventListener("click", () => {
        if (!window.confirm(`Delete the saved ${plan.participant} session from this browser? This cannot be undone.`)) return;
        try { localStorage.removeItem(storageKey(plan)); } catch { /* storage unavailable: nothing saved */ }
        note.replaceChildren(document.createTextNode(`Saved ${plan.participant} session discarded.`));
      });
      note.append(button, " ", discard);
      note.classList.remove("hidden");
    }
    state.trials = buildTrials(plan);
    state.index = 0;
    state.results = {
      results_version: RESULTS_VERSION,
      study: plan.study,
      participant: plan.participant,
      arm: plan.arm,
      started_at: new Date().toISOString(),
      finished_at: null,
      completed: false,
      user_agent: navigator.userAgent,
      viewport: [window.innerWidth, window.innerHeight],
      trials: [],
    };
    const start = document.createElement("button");
    start.type = "button";
    start.className = "primary";
    start.textContent = `Start session for ${plan.participant}`;
    start.addEventListener("click", () => { save(); showIntro(0); });
    $("setup-view").querySelector(".row").replaceChildren(start);
  } catch (error) {
    $("setup-error").textContent = `Could not load the plan: ${error.message}`;
  }
});

function showIntro(block) {
  const condition = state.plan.blocks[block].condition;
  $("intro-title").textContent = `Part ${block + 1} of 2: ${condition === "assisted" ? "with BeltWatch" : "photo only"}`;
  $("intro-text").replaceChildren(...INTRO[condition].map((line) => {
    const p = document.createElement("p");
    p.textContent = line;
    return p;
  }));
  show("intro-view");
}

$("start-block").addEventListener("click", () => loadTrial());

function setView(view) {
  const d = state.detail;
  const url = { overlay: d.overlay_url, original: d.original_url, uncertainty: d.uncertainty_url }[view];
  for (const button of $("view-tabs").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
  }
  state.views[view] = (state.views[view] || 0) + 1;
  $("trial-image").src = url;
}

for (const button of $("view-tabs").querySelectorAll("button")) {
  button.addEventListener("click", () => setView(button.dataset.view));
}

function suggestionFor(detail) {
  const coverage = detail.visible_coverage || {};
  return {
    follow_up: (detail.total_target_coverage || 0) > state.plan.audit_positive_threshold,
    materials: Object.keys(coverage).filter((m) => coverage[m] >= state.plan.material_present_threshold).sort(),
  };
}

function resetForm(suggestion) {
  const form = $("answer-form");
  for (const input of form.querySelectorAll("input[name=follow_up]")) {
    input.checked = suggestion ? input.value === (suggestion.follow_up ? "yes" : "no") : false;
  }
  for (const input of form.querySelectorAll("input[name=materials]")) {
    input.checked = suggestion ? suggestion.materials.includes(input.value) : false;
  }
}

async function loadTrial() {
  const trial = state.trials[state.index];
  const practiceCount = state.trials.filter((t) => t.block === trial.block && t.practice).length;
  const blockTrials = state.trials.filter((t) => t.block === trial.block);
  const n = blockTrials.indexOf(trial) + 1;
  $("trial-title").textContent = trial.practice
    ? `Practice ${n} of ${practiceCount}`
    : `Image ${n - practiceCount} of ${blockTrials.length - practiceCount}`;
  $("progress").textContent = `Part ${trial.block + 1} of 2`;
  $("trial-error").textContent = "";
  $("submit-answer").disabled = true;
  $("trial-image").classList.add("hidden");
  $("trial-image").removeAttribute("src"); // a fresh src always fires load, which starts the clock
  $("loading").classList.remove("hidden");
  show("trial-view");
  state.t0 = null;
  state.views = {};
  try {
    const detail = await api(`/v1/audits/${encodeURIComponent(trial.audit_id)}/images/${encodeURIComponent(trial.image_id)}`);
    if (detail.status !== "complete") throw new Error(`image is ${detail.status}`);
    state.detail = detail;
    const assisted = trial.condition === "assisted";
    trial.suggestion = assisted ? suggestionFor(detail) : null;
    $("view-tabs").classList.toggle("hidden", !assisted);
    $("assist-panel").classList.toggle("hidden", !assisted);
    if (assisted) {
      $("suggestion-text").textContent = trial.suggestion.follow_up
        ? "Suggested: follow up (estimated visible target coverage above 5%)."
        : "Suggested: no follow-up (estimated visible target coverage at or below 5%).";
      $("coverage-list").replaceChildren(...Object.entries(MATERIAL_NAMES).map(([key, name]) => {
        const li = document.createElement("li");
        const value = (detail.visible_coverage || {})[key];
        li.textContent = `${name}: ${value === undefined ? "–" : `${(value * 100).toFixed(1)}%`}`;
        return li;
      }));
      $("reasons-text").textContent = detail.review_reasons?.length
        ? `Review reasons: ${detail.review_reasons.join(", ")}`
        : "No review reasons.";
      setView("overlay");
    } else {
      state.views.original = 1;
      $("trial-image").src = detail.original_url;
    }
    resetForm(trial.suggestion);
  } catch (error) {
    $("loading").classList.add("hidden");
    $("trial-error").textContent = `Could not load this image (${error.message}). Tell the facilitator.`;
  }
}

$("trial-image").addEventListener("load", () => {
  $("loading").classList.add("hidden");
  $("trial-image").classList.remove("hidden");
  if (state.t0 === null) state.t0 = performance.now(); // the clock starts once the image is visible
  updateSubmit();
});

$("trial-image").addEventListener("error", () => {
  if (!$("trial-image").getAttribute("src")) return;
  $("loading").classList.add("hidden");
  $("trial-error").textContent = "Could not load this image. Tell the facilitator.";
});

function updateSubmit() {
  const answered = $("answer-form").querySelector("input[name=follow_up]:checked");
  $("submit-answer").disabled = state.t0 === null || !answered;
}

$("answer-form").addEventListener("change", updateSubmit);

$("answer-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.t0 === null) return;
  const seconds = (performance.now() - state.t0) / 1000;
  const form = $("answer-form");
  const trial = state.trials[state.index];
  state.results.trials.push({
    block: trial.block,
    condition: trial.condition,
    set: trial.set,
    practice: trial.practice,
    audit_id: trial.audit_id,
    image_id: trial.image_id,
    position: trial.position,
    seconds: Math.round(seconds * 1000) / 1000,
    follow_up: form.querySelector("input[name=follow_up]:checked").value === "yes",
    materials: [...form.querySelectorAll("input[name=materials]:checked")].map((i) => i.value).sort(),
    suggestion: trial.suggestion,
    views: state.views,
    answered_at: new Date().toISOString(),
  });
  state.index += 1;
  if (state.index === state.trials.length) {
    state.results.completed = true;
    state.results.finished_at = new Date().toISOString();
    save();
    $("progress").textContent = "Done";
    show("done-view");
    return;
  }
  save();
  const next = state.trials[state.index];
  if (next.block !== trial.block) showIntro(next.block);
  else loadTrial();
});

$("download").addEventListener("click", () => download(state.results));
