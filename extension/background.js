// TruthCheck background service worker: context menus + backend calls.
const API_DEFAULT = "http://localhost:8000";
const MEDIA_FETCH_TIMEOUT = 60000;
const API_TIMEOUT = 90000;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "tc-text",
      title: "TruthCheck: analyze text",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "tc-image",
      title: "TruthCheck: analyze image",
      contexts: ["image"],
    });
    chrome.contextMenus.create({
      id: "tc-video",
      title: "TruthCheck: analyze video",
      contexts: ["video"],
    });
  });
});

async function apiBase() {
  const { apiBase } = await chrome.storage.local.get("apiBase");
  return String(apiBase || API_DEFAULT).replace(/\/+$/, "");
}

async function fetchWithTimeout(url, options = {}, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function callApi(path, options) {
  const base = await apiBase();
  const resp = await fetchWithTimeout(`${base}${path}`, options, API_TIMEOUT);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body && body.detail) {
        detail += `: ${typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)}`;
      }
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return resp.json();
}

function analyzeText(text) {
  return callApi("/analyze/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, verify: true }),
  });
}

async function analyzeMedia(url, kind) {
  // Needs the <all_urls> host permission, otherwise cross-origin media is
  // blocked by CORS (a known limitation - failure is surfaced to the user).
  const resp = await fetchWithTimeout(url, {}, MEDIA_FETCH_TIMEOUT);
  if (!resp.ok) throw new Error(`could not download ${kind}: HTTP ${resp.status}`);
  const blob = await resp.blob();
  if (!blob || blob.size === 0) throw new Error(`${kind} download was empty`);
  const form = new FormData();
  const ext = kind === "image" ? "png" : "mp4";
  form.append("file", blob, `selection.${ext}`);
  return callApi(`/analyze/${kind}`, { method: "POST", body: form });
}

const KIND_BY_MENU_ID = { "tc-text": "text", "tc-image": "image", "tc-video": "video" };

function sendToTab(tabId, message) {
  if (tabId == null) return Promise.resolve(false);
  return chrome.tabs.sendMessage(tabId, message).then(
    () => true,
    () => false
  );
}

function updateBadge(tabId, verdict) {
  const colors = { FAKE: "#c62828", REAL: "#2e7d32", UNKNOWN: "#546e7a", ERROR: "#6a1b9a" };
  const text = verdict === "FAKE" ? "FAKE" : verdict === "REAL" ? "REAL" : verdict === "ERROR" ? "ERR" : "?";
  chrome.action.setBadgeBackgroundColor({ color: colors[verdict] || "#546e7a" });
  chrome.action.setBadgeText({ text, tabId });
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const kind = KIND_BY_MENU_ID[info.menuItemId];
  if (!kind) return;
  const tabId = tab && tab.id;

  sendToTab(tabId, { type: "TC_STATUS", kind, label: `Analyzing ${kind}...` });

  let result;
  try {
    if (kind === "text") {
      if (!info.selectionText || !info.selectionText.trim()) {
        throw new Error("nothing was selected");
      }
      result = await analyzeText(info.selectionText);
    } else {
      if (!info.srcUrl) throw new Error("media URL not available");
      // Image only: info.srcUrl is frequently a thumbnail (e.g. 707x434
      // instead of 1465x900). Ask the page for the largest candidate.
      let url = info.srcUrl;
      if (kind === "image") {
        try {
          const best = await chrome.tabs.sendMessage(tabId, {
            type: "TC_BEST_SRC",
            fallbackUrl: info.srcUrl,
          });
          if (best && best.src) url = best.src;
        } catch (_) {
          /* content script not available: use info.srcUrl */
        }
      }
      try {
        result = await analyzeMedia(url, kind);
      } catch (err) {
        if (kind === "image" && url !== info.srcUrl) {
          result = await analyzeMedia(info.srcUrl, kind); // retry with srcUrl
        } else {
          throw err;
        }
      }
    }
  } catch (err) {
    result = {
      verdict: "ERROR",
      confidence: 0,
      details: { error: String((err && err.message) || err) },
    };
  }

  const payload = { type: "TC_RESULT", kind, result, at: Date.now() };
  const delivered = await sendToTab(tabId, payload);
  await chrome.storage.local.set({ lastResult: payload, lastDelivered: delivered });
  updateBadge(tabId, result.verdict);
});

// Popup asks the background worker for the latest health snapshot.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "TC_HEALTH") {
    fetchWithTimeout(`${API_DEFAULT}/health`, {}, 5000)
      .then((r) => r.json())
      .then((body) => sendResponse({ ok: true, body }))
      .catch((err) => sendResponse({ ok: false, error: String(err && err.message ? err.message : err) }));
    return true; // async response
  }
  return false;
});
