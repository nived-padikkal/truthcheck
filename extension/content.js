// TruthCheck content script: shows analysis results as an overlay on the page.
(() => {
  const HOST_ID = "truthcheck-overlay";
  const COLORS = {
    FAKE: "#c62828",
    REAL: "#2e7d32",
    UNKNOWN: "#455a64",
    ERROR: "#6a1b9a",
    STATUS: "#37474f",
  };
  let hideTimer = null;

  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);
  }

  function detailLines(result) {
    const d = (result && result.details) || {};
    const lines = [];
    if (d.model) lines.push(d.model_loaded === false ? `model: ${d.model} (not trained)` : `model: ${d.model}`);
    if (Array.isArray(d.checks) && d.checks.length) {
      lines.push(`checks: ${d.checks.map((c) => `${c.model} ${(c.p_fake * 100).toFixed(0)}% fake`).join(" | ")}`);
    }
    if (typeof d.frames_analyzed === "number") lines.push(`frames analyzed: ${d.frames_analyzed}`);
    if (d.image_size) lines.push(`image: ${d.image_size.join("x")}`);
    if (typeof d.latency_ms === "number") lines.push(`${d.latency_ms} ms`);
    if (d.verification) {
      const v = d.verification;
      lines.push(v.found ? `verification: ${v.method} (score ${v.score})` : "verification: no external signal");
    }
    if (d.error) lines.push(`error: ${d.error}`);
    if (d.reason) lines.push(d.reason);
    return lines;
  }

  function render({ verdict, confidence, details }) {
    const host = ensureHost();
    const pct = Math.round((confidence || 0) * 100);
    const color = COLORS[verdict] || COLORS.UNKNOWN;
    const lines = detailLines({ verdict, details });
    host.innerHTML = `
      <div class="tc-card" style="background:${color}">
        <div class="tc-head">
          <span style="font-weight:700;letter-spacing:.4px">TRUTHCHECK</span>
          <button class="tc-close" title="Dismiss" aria-label="Dismiss"
            style="background:transparent;border:0;color:#fff;font-size:16px;line-height:1;cursor:pointer;padding:0 2px">&times;</button>
        </div>
        <div style="font-size:17px;font-weight:700;margin:2px 0 6px">Verdict: ${esc(verdict)}</div>
        <div style="background:rgba(255,255,255,.25);border-radius:4px;height:8px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:#fff;transition:width .4s ease"></div>
        </div>
        <div style="font-size:13px;margin-top:6px">Confidence: ${pct}%</div>
        ${lines.length ? `<div style="font-size:11px;opacity:.9;margin-top:6px;line-height:1.5">${lines
          .map((l) => esc(l))
          .join("<br>")}</div>` : ""}
      </div>`;
    host.querySelector(".tc-close").addEventListener("click", remove);
    scheduleHide(12000);
  }

  function renderStatus(label) {
    const host = ensureHost();
    host.innerHTML = `
      <div class="tc-card" style="background:${COLORS.STATUS}">
        <div class="tc-head"><span style="font-weight:700;letter-spacing:.4px">TRUTHCHECK</span></div>
        <div style="font-size:14px;margin-top:4px;display:flex;align-items:center;gap:8px">
          <span class="tc-spin" style="width:12px;height:12px;border:2px solid rgba(255,255,255,.35);
            border-top-color:#fff;border-radius:50%;display:inline-block;animation:tcspin .8s linear infinite"></span>
          ${esc(label || "Analyzing...")}
        </div>
      </div>`;
    if (!document.getElementById("tc-style")) {
      const style = document.createElement("style");
      style.id = "tc-style";
      style.textContent = "@keyframes tcspin{to{transform:rotate(360deg)}}";
      document.head.appendChild(style);
    }
    scheduleHide(60000);
  }

  function ensureHost() {
    let host = document.getElementById(HOST_ID);
    if (host) return host;
    host = document.createElement("div");
    host.id = HOST_ID;
    Object.assign(host.style, {
      position: "fixed",
      top: "16px",
      right: "16px",
      zIndex: String(2147483647),
      fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
      color: "#fff",
    });
    const style = document.createElement("style");
    style.textContent = `
      #${HOST_ID} .tc-card{min-width:230px;max-width:330px;padding:12px 14px;border-radius:10px;
        box-shadow:0 8px 28px rgba(0,0,0,.35);animation:tcin .18s ease-out}
      #${HOST_ID} .tc-head{display:flex;justify-content:space-between;align-items:center;
        font-size:11px;opacity:.9}
      @keyframes tcin{from{transform:translateX(24px);opacity:0}to{transform:none;opacity:1}}`;
    host.appendChild(style);
    (document.body || document.documentElement).appendChild(host);
    return host;
  }

  function scheduleHide(ms) {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(remove, ms);
  }

  function remove() {
    clearTimeout(hideTimer);
    const host = document.getElementById(HOST_ID);
    if (host) host.remove();
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type) return false;
    if (msg.type === "TC_RESULT") {
      render(msg.result || {});
      sendResponse && sendResponse({ ok: true });
    } else if (msg.type === "TC_STATUS") {
      renderStatus(msg.label);
      sendResponse && sendResponse({ ok: true });
    } else if (msg.type === "TC_GET_SELECTION") {
      const selection = window.getSelection ? String(window.getSelection()) : "";
      sendResponse && sendResponse({ text: selection });
    }
    return false;
  });
})();
