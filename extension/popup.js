// TruthCheck popup: shows the latest analysis result (from the right-click
// menu) plus backend health. Settings live on the options page.
const DEFAULT_API = "http://localhost:8000";
const $ = (id) => document.getElementById(id);

let apiBase = DEFAULT_API;
const RING_CIRCUMFERENCE = 339.292; // 2 * PI * 54

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function setSubtitle(text) {
  const el = $("subtitle");
  if (el) el.textContent = text;
}

function tierOf(verdict, pct) {
  if (verdict === "FAKE" || verdict === "ERROR") return "high";
  if (verdict === "REAL") return "low";
  if (pct >= 75) return "high";
  if (pct >= 45) return "medium";
  return "low";
}

function resetRing() {
  const ring = $("circleProgress");
  ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
  ring.className = "circle-progress";
  $("confidenceValue").textContent = "0";
  $("confidenceLabel").textContent = "—";
  $("confidenceLabel").className = "confidence-label";
  const badge = $("badge");
  badge.textContent = "—";
  badge.className = "verdict-text";
  $("signalCount").textContent = "0";
}

async function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  await chrome.storage.local.set({ theme });
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
}

async function loadTheme() {
  const { theme } = await chrome.storage.local.get("theme");
  document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
}

async function loadSettings() {
  const { apiBase: stored } = await chrome.storage.local.get("apiBase");
  apiBase = (stored || DEFAULT_API).replace(/\/+$/, "");
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
      if ($("subtitle").textContent.startsWith("Backend")) setSubtitle("Verify before you trust.");
      return;
    }
    throw new Error("bad status");
  } catch (_) {
    dot.classList.remove("ok");
    dot.classList.add("down");
    dot.title = "Backend unreachable - start it with: uvicorn app.main:app --reload --port 8000";
    setSubtitle("Backend unreachable - start uvicorn on port 8000.");
  }
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

function showHint(show) {
  const hint = $("hint");
  if (hint) hint.classList.toggle("hidden", !show);
}

function renderResult(result) {
  const box = $("result");
  if (!result) {
    box.classList.add("hidden");
    resetRing();
    setSubtitle("Verify before you trust.");
    showHint(true);
    return;
  }
  const verdict = result.verdict || "UNKNOWN";
  const pct = Math.round((result.confidence || 0) * 100);

  const badge = $("badge");
  badge.textContent = verdict;
  badge.className = `verdict-text ${verdict}`;

  $("confidenceValue").textContent = String(pct);
  const ring = $("circleProgress");
  ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - pct / 100));
  const tier = tierOf(verdict, pct);
  ring.className = `circle-progress ${tier}`;

  const label = $("confidenceLabel");
  label.textContent = tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low";
  label.className = `confidence-label ${tier}`;

  $("meta").innerHTML = metaLines(result).map((l) => `<div>${esc(l)}</div>`).join("");

  const items = signalItems(result);
  $("signals").innerHTML = items.map((s) => `<li>${esc(s)}</li>`).join("");
  $("signalCount").textContent = String(items.length);

  box.classList.remove("hidden");
  showHint(false);
  setSubtitle(
    verdict === "FAKE"
      ? "Manipulation detected - review the signals below."
      : verdict === "REAL"
      ? "No manipulation found in this content."
      : verdict === "ERROR"
      ? "Analysis failed - see the details below."
      : "Inconclusive - not enough evidence either way."
  );
}

async function restoreLastResult() {
  const { lastResult } = await chrome.storage.local.get("lastResult");
  if (lastResult && Date.now() - lastResult.at < 10 * 60 * 1000) {
    renderResult(lastResult.result);
  } else {
    renderResult(null);
  }
}

function openSettings() {
  if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadTheme();
  await loadSettings();
  await checkHealth();
  await restoreLastResult();

  $("refreshButton").addEventListener("click", checkHealth);
  $("themeToggle").addEventListener("click", toggleTheme);
  $("optionsButton").addEventListener("click", openSettings);
  $("footerSettings").addEventListener("click", openSettings);

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    const key = e.key.toLowerCase();
    if (key === "t") toggleTheme();
    else if (key === "r") checkHealth();
    else if (key === "s") openSettings();
  });
});
