// TruthCheck popup: paste text (or grab the page selection) and analyze it.
const DEFAULT_API = "http://localhost:8000";
const $ = (id) => document.getElementById(id);

let apiBase = DEFAULT_API;

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

async function loadSettings() {
  const { apiBase: stored } = await chrome.storage.local.get("apiBase");
  apiBase = (stored || DEFAULT_API).replace(/\/+$/, "");
  $("api").value = apiBase;
}

async function saveSettings() {
  apiBase = ($("api").value || DEFAULT_API).replace(/\/+$/, "");
  await chrome.storage.local.set({ apiBase });
  checkHealth();
}

async function checkHealth() {
  const dot = $("health-dot");
  try {
    const resp = await fetch(`${apiBase}/health`, { signal: AbortSignal.timeout(5000) });
    const body = await resp.json();
    if (body.status === "ok") {
      dot.classList.add("ok");
      dot.classList.remove("down");
      const models = body.models || {};
      dot.title = `Backend online · text model: ${models.text ? "loaded" : "not trained"} · meso4: ${
        models.meso4 ? "loaded" : "not trained"
      }`;
      return;
    }
    throw new Error("bad status");
  } catch (_) {
    dot.classList.remove("ok");
    dot.classList.add("down");
    dot.title = "Backend unreachable - start it with: uvicorn app.main:app --reload --port 8000";
  }
}

function showError(message) {
  const el = $("error");
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function metaLines(result) {
  const d = (result && result.details) || {};
  const lines = [];
  if (d.model) lines.push(`Model: ${d.model}${d.model_loaded === false ? " (not trained - surface heuristic)" : ""}`);
  if (typeof d.p_fake === "number") {
    lines.push(`P(fake) = ${Math.round(d.p_fake * 100)}%${d.p_fake_ml != null && d.blended ? ` (ML: ${Math.round(d.p_fake_ml * 100)}%)` : ""}`);
  }
  if (typeof d.frames_analyzed === "number") lines.push(`Frames analyzed: ${d.frames_analyzed}`);
  if (d.image_size) lines.push(`Image: ${d.image_size.join("x")}`);
  if (d.verification) {
    const v = d.verification;
    lines.push(
      v.found
        ? `Verification (${v.method}): score ${v.score} — ${v.explanation || ""}`
        : `Verification: ${v.explanation || "no external signal"}`
    );
  }
  if (d.error) lines.push(`Error: ${d.error}`);
  if (d.reason) lines.push(d.reason);
  return lines;
}

function signalItems(result) {
  const d = (result && result.details) || {};
  const items = [];
  if (Array.isArray(d.signals)) items.push(...d.signals);
  if (d.verification) {
    (d.verification.contradicting || []).forEach((m) => items.push(`contradiction marker: "${m}"`));
    (d.verification.corroborating || []).forEach((m) => items.push(`corroboration marker: "${m}"`));
  }
  if (Array.isArray(d.frame_scores)) {
    const forged = d.frames_forged;
    items.push(`${forged} of ${d.frame_scores.length} sampled frames scored as forged`);
  }
  return items;
}

function renderResult(result) {
  const box = $("result");
  if (!result) {
    box.classList.add("hidden");
    return;
  }
  const verdict = result.verdict || "UNKNOWN";
  const pct = Math.round((result.confidence || 0) * 100);

  const badge = $("badge");
  badge.textContent = verdict;
  badge.className = `badge ${verdict}`;

  $("confidence").textContent = `${pct}%`;

  const fill = $("meter-fill");
  fill.style.width = `${pct}%`;
  fill.className = `meter-fill ${verdict}`;

  $("meta").innerHTML = metaLines(result).map((l) => `<div>${esc(l)}</div>`).join("");

  const items = signalItems(result);
  $("signals").innerHTML = items.map((s) => `<li>${esc(s)}</li>`).join("");

  box.classList.remove("hidden");
}

async function analyze() {
  const text = $("text").value.trim();
  const button = $("analyze");
  showError("");
  if (!text) {
    showError("Paste or select some text first.");
    return;
  }
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Analyzing';
  try {
    const resp = await fetch(`${apiBase}/analyze/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, verify: $("verify").checked }),
      signal: AbortSignal.timeout(60000),
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (body.detail) detail += `: ${typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)}`;
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    const result = await resp.json();
    renderResult(result);
    await chrome.storage.local.set({
      lastResult: { type: "TC_RESULT", kind: "text", result, at: Date.now() },
    });
  } catch (err) {
    showError(`Analysis failed: ${err && err.message ? err.message : err}`);
  } finally {
    button.disabled = false;
    button.textContent = "Analyze";
  }
}

async function useSelection() {
  showError("");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || tab.id == null) throw new Error("no active tab");
    const reply = await chrome.tabs.sendMessage(tab.id, { type: "TC_GET_SELECTION" });
    const selected = (reply && reply.text || "").trim();
    if (!selected) {
      throw new Error("Nothing selected on this page - select text first, or paste it below.");
    }
    $("text").value = selected;
    await analyze();
  } catch (err) {
    showError(err && err.message ? err.message : String(err));
  }
}

function clearAll() {
  $("text").value = "";
  showError("");
  renderResult(null);
}

async function restoreLastResult() {
  const { lastResult } = await chrome.storage.local.get("lastResult");
  if (lastResult && Date.now() - lastResult.at < 10 * 60 * 1000) {
    renderResult(lastResult.result);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  await checkHealth();
  await restoreLastResult();
  $("analyze").addEventListener("click", analyze);
  $("selection").addEventListener("click", useSelection);
  $("clear").addEventListener("click", clearAll);
  $("api").addEventListener("change", saveSettings);
  $("text").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
});
