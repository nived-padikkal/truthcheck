// TruthCheck settings page: backend API URL + theme.
const DEFAULT_API = "http://localhost:8000";
const $ = (id) => document.getElementById(id);

function currentApi() {
  return ($("api").value || DEFAULT_API).replace(/\/+$/, "");
}

function setStatus(text, cls) {
  const el = $("statusLine");
  el.textContent = text || "";
  el.className = `status-line${cls ? ` ${cls}` : ""}`;
}

function renderModels(models) {
  const box = $("models");
  box.innerHTML = "";
  if (!models) return;
  ["text", "meso4", "cifake"].forEach((name) => {
    const chip = document.createElement("span");
    const ok = !!models[name];
    chip.className = `model-chip${ok ? " ok" : ""}`;
    chip.textContent = `${name}: ${ok ? "loaded" : "not trained"}`;
    box.appendChild(chip);
  });
}

function renderHealth(ok, body) {
  const pill = $("healthPill");
  if (ok) {
    pill.textContent = "online";
    pill.className = "status-pill ok";
    setStatus("Connected.", "ok");
    renderModels(body && body.models);
  } else {
    pill.textContent = "offline";
    pill.className = "status-pill down";
    setStatus("Cannot reach the backend - is uvicorn running on that URL?", "down");
    renderModels(null);
  }
}

async function checkHealth() {
  setStatus("Testing connection…");
  try {
    const resp = await fetch(`${currentApi()}/health`, { signal: AbortSignal.timeout(5000) });
    const body = await resp.json();
    if (body.status !== "ok") throw new Error("bad status");
    renderHealth(true, body);
  } catch (_) {
    renderHealth(false);
  }
}

async function save() {
  const apiBase = currentApi();
  $("api").value = apiBase;
  await chrome.storage.local.set({ apiBase });
  await checkHealth();
}

async function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  await chrome.storage.local.set({ theme });
}

document.addEventListener("DOMContentLoaded", async () => {
  const { apiBase, theme } = await chrome.storage.local.get(["apiBase", "theme"]);
  $("api").value = (apiBase || DEFAULT_API).replace(/\/+$/, "");
  $("theme").value = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = $("theme").value;

  $("save").addEventListener("click", save);
  $("test").addEventListener("click", checkHealth);
  $("api").addEventListener("keydown", (e) => {
    if (e.key === "Enter") save();
  });
  $("theme").addEventListener("change", (e) => applyTheme(e.target.value));

  await checkHealth();
});
