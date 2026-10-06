// TruthCheck content script: shows analysis results as an overlay on the page.
(() => {
  const HOST_ID = "truthcheck-overlay";
  const COLORS = {
    FAKE: "#ef4444",
    REAL: "#22c55e",
    UNKNOWN: "#b45309",
    ERROR: "#7c3aed",
    STATUS: "#0f2a85",
  };
  const BRAND = "#0f2a85";
  const RING_R = 52;
  const RING_CIRC = 2 * Math.PI * RING_R; // 326.73
  let hideTimer = null;

  const VERDICT_TEXT = {
    FAKE: "FAKE",
    REAL: "NO MANIPULATION FOUND",
    UNKNOWN: "INCONCLUSIVE",
    ERROR: "ERROR",
  };

  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);
  }

  function detailItems(d) {
    // model / frames / latency live in the footer chips, not the list.
    const items = [];
    if (Array.isArray(d.checks) && d.checks.length) {
      items.push(`checks: ${d.checks.map((c) => `${c.model} ${(c.p_fake * 100).toFixed(0)}% fake`).join(" | ")}`);
    }
    if (d.image_size) items.push(`image: ${d.image_size.join("x")}`);
    if (d.verification) {
      const v = d.verification;
      items.push(v.found ? `verification: ${v.method} (score ${v.score})` : "verification: no external signal");
    }
    if (d.error) items.push(`error: ${d.error}`);
    return items;
  }

  function chips(d) {
    const out = [];
    if (d.model) out.push(d.model_loaded === false ? `${d.model} (untrained)` : d.model);
    if (typeof d.frames_analyzed === "number") out.push(`${d.frames_analyzed} frames`);
    if (d.latency_ms != null) out.push(`${Math.round(d.latency_ms)} ms`);
    if (d.error && !out.length) out.push("failed");
    return out;
  }

  function tierOf(verdict, pct) {
    if (verdict === "FAKE" || verdict === "ERROR") return "high";
    if (verdict === "REAL") return "low";
    if (pct >= 75) return "high";
    if (pct >= 45) return "medium";
    return "low";
  }

  function render({ verdict, confidence, details }) {
    const host = ensureHost();
    const d = details || {};
    verdict = verdict || "UNKNOWN";
    const pct = Math.round((confidence || 0) * 100);
    const color = COLORS[verdict] || COLORS.UNKNOWN;
    const offset = RING_CIRC * (1 - pct / 100);
    const items = detailItems(d);
    const label = VERDICT_TEXT[verdict] || verdict;
    const tier = tierOf(verdict, pct);
    // A confidence meter only makes sense for a decisive verdict; UNKNOWN is
    // "inconclusive" and must never render as "Confidence: 0%".
    const showBar = (verdict === "REAL" || verdict === "FAKE") && pct > 0;
    const note =
      verdict === "UNKNOWN" ? "Inconclusive - not enough evidence either way" : "";
    const chipHtml = chips(d)
      .map((c) => `<span class="tc-chip">${esc(c)}</span>`)
      .join("");

    host.innerHTML = `
      <div class="tc-card">
        <div class="tc-head">
          <span class="tc-brand">TRUTHCHECK</span>
          <button class="tc-close" title="Dismiss" aria-label="Dismiss">&times;</button>
        </div>
        <div class="tc-body">
          <div class="tc-main">
            ${showBar ? `
            <div class="tc-ring">
              <svg viewBox="0 0 120 120" aria-hidden="true">
                <circle class="tc-ring-bg" cx="60" cy="60" r="${RING_R}"></circle>
                <circle class="tc-ring-fg" cx="60" cy="60" r="${RING_R}"
                  style="stroke:${color};stroke-dasharray:${RING_CIRC.toFixed(2)};
                  stroke-dashoffset:${offset.toFixed(2)}"></circle>
              </svg>
              <div class="tc-ring-value"><span>${pct}</span><i>%</i></div>
            </div>` : ""}
            <div class="tc-verdict-block">
              <div class="tc-verdict" style="color:${color};font-size:${label.length > 8 ? 18 : 26}px">${esc(label)}</div>
              ${showBar ? `<div class="tc-tier tc-tier-${tier}">${tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low"} confidence</div>` : ""}
              ${note ? `<div class="tc-note">${esc(note)}</div>` : ""}
            </div>
          </div>
          ${d.reason ? `<div class="tc-reason">${esc(d.reason)}</div>` : ""}
          ${items.length ? `
            <div class="tc-details">
              <div class="tc-details-head">Details <span class="tc-count">${items.length}</span></div>
              <ul class="tc-list">${items.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
            </div>` : ""}
        </div>
        ${chipHtml ? `<div class="tc-foot">${chipHtml}</div>` : ""}
      </div>`;

    host.querySelector(".tc-close").addEventListener("click", remove);
    const card = host.querySelector(".tc-card");
    card.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    card.addEventListener("mouseleave", () => scheduleHide(6000));
    scheduleHide(12000);
  }

  function renderStatus(label) {
    const host = ensureHost();
    host.innerHTML = `
      <div class="tc-card tc-card-status">
        <div class="tc-head" style="background:rgba(255,255,255,.12)">
          <span class="tc-brand">TRUTHCHECK</span>
        </div>
        <div class="tc-body tc-status-row">
          <span class="tc-spin"></span>
          <span>${esc(label || "Analyzing...")}</span>
        </div>
      </div>`;
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
      zIndex: "2147483647",
      fontFamily: "'Jost','Work Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif",
      color: "#fff",
    });

    const style = document.createElement("style");
    style.textContent = `
      #${HOST_ID} *{box-sizing:border-box}
      #${HOST_ID} .tc-card{
        width:300px;background:#fff;color:#1a1a1a;border-radius:12px;overflow:hidden;
        box-shadow:0 12px 36px rgba(0,0,0,.28);animation:tcin .22s cubic-bezier(.2,.8,.3,1)}
      #${HOST_ID} .tc-head{
        display:flex;justify-content:space-between;align-items:center;
        background:${BRAND};color:#fff;padding:9px 12px 9px 14px}
      #${HOST_ID} .tc-brand{font-size:11px;font-weight:700;letter-spacing:.14em}
      #${HOST_ID} .tc-close{
        background:transparent;border:0;color:#fff;font-size:17px;line-height:1;
        cursor:pointer;padding:0 4px;opacity:.85}
      #${HOST_ID} .tc-close:hover{opacity:1}
      #${HOST_ID} .tc-body{padding:14px}
      #${HOST_ID} .tc-main{display:flex;align-items:center;gap:14px}
      #${HOST_ID} .tc-ring{position:relative;width:74px;height:74px;flex:0 0 auto}
      #${HOST_ID} .tc-ring svg{width:100%;height:100%;transform:rotate(-90deg)}
      #${HOST_ID} .tc-ring-bg{fill:none;stroke:#f0f0f0;stroke-width:9}
      #${HOST_ID} .tc-ring-fg{fill:none;stroke-width:9;stroke-linecap:butt;
        transition:stroke-dashoffset .6s ease}
      #${HOST_ID} .tc-ring-value{
        position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
        gap:1px;color:#1a1a1a}
      #${HOST_ID} .tc-ring-value span{font-size:20px;font-weight:700;line-height:1}
      #${HOST_ID} .tc-ring-value i{font-size:11px;font-style:normal;font-weight:700;color:#6a6a6a}
      #${HOST_ID} .tc-verdict-block{min-width:0}
      #${HOST_ID} .tc-verdict{
        font-size:26px;font-weight:700;line-height:1.1;letter-spacing:.02em;
        font-family:'Jost',system-ui,sans-serif}
      #${HOST_ID} .tc-tier{font-size:11px;font-weight:700;margin-top:3px;letter-spacing:.03em}
      #${HOST_ID} .tc-tier-high{color:#ef4444}
      #${HOST_ID} .tc-tier-medium{color:#b45309}
      #${HOST_ID} .tc-tier-low{color:#16a34a}
      #${HOST_ID} .tc-note{font-size:12px;margin-top:4px;color:#6a6a6a;line-height:1.45}
      #${HOST_ID} .tc-reason{
        margin-top:12px;font-size:12px;line-height:1.5;color:#4a4a4a;
        background:#f8f8f8;border-left:3px solid ${BRAND};
        padding:7px 9px;border-radius:0 6px 6px 0}
      #${HOST_ID} .tc-details{margin-top:12px}
      #${HOST_ID} .tc-details-head{
        display:flex;justify-content:space-between;align-items:center;
        font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
        color:#6a6a6a;margin-bottom:6px}
      #${HOST_ID} .tc-count{
        background:#f0f0f0;color:#4a4a4a;border-radius:999px;padding:1px 7px;font-size:10px}
      #${HOST_ID} .tc-list{list-style:none;margin:0;padding:0;
        max-height:120px;overflow-y:auto}
      #${HOST_ID} .tc-list li{
        position:relative;padding:3px 0 3px 14px;font-size:11.5px;line-height:1.5;
        color:#4a4a4a;word-break:break-word}
      #${HOST_ID} .tc-list li::before{
        content:'';position:absolute;left:2px;top:9px;width:5px;height:5px;
        border-radius:50%;background:#c9ccd1}
      #${HOST_ID} .tc-list::-webkit-scrollbar{width:6px}
      #${HOST_ID} .tc-list::-webkit-scrollbar-thumb{background:#e0e0e0;border-radius:3px}
      #${HOST_ID} .tc-foot{
        display:flex;flex-wrap:wrap;gap:6px;padding:9px 14px;
        background:#f8f8f8;border-top:1px solid #eee}
      #${HOST_ID} .tc-chip{
        font-size:10px;font-weight:600;color:#4a4a4a;background:#fff;
        border:1px solid #e6e6e6;border-radius:999px;padding:2px 8px}
      #${HOST_ID} .tc-card-status{background:${BRAND};color:#fff}
      #${HOST_ID} .tc-status-row{display:flex;align-items:center;gap:10px;
        font-size:13px;font-weight:600}
      #${HOST_ID} .tc-spin{
        width:14px;height:14px;flex:0 0 auto;border-radius:50%;
        border:2px solid rgba(255,255,255,.35);border-top-color:#fff;
        animation:tcspin .8s linear infinite}
      @keyframes tcspin{to{transform:rotate(360deg)}}
      @keyframes tcin{from{transform:translateX(28px);opacity:0}to{transform:none;opacity:1}}`;
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

  // --- best image source (problem 5: info.srcUrl is often a thumbnail) -----
  let tcLastTarget = null;
  document.addEventListener("contextmenu", (e) => { tcLastTarget = e.target; }, true);

  function tcBestImageSource(fallbackUrl) {
    let el = tcLastTarget;
    const img =
      el &&
      (el.tagName === "IMG"
        ? el
        : ((el.closest && el.closest("picture, a, div")) || el).querySelector?.("img"));
    if (!img) return { src: fallbackUrl, reason: "no <img> found" };
    let best = img.currentSrc || img.src || fallbackUrl;
    let bestW = img.naturalWidth || 0;
    const srcset = img.getAttribute("srcset");
    if (srcset) {
      for (const part of srcset.split(",")) {
        const [u, d] = part.trim().split(/\s+/);
        const w = d && d.endsWith("w") ? parseInt(d, 10) : 0;
        if (u && w > bestW) { best = new URL(u, location.href).href; bestW = w; }
      }
    }
    return { src: best, naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight, reason: "ok" };
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
    } else if (msg.type === "TC_BEST_SRC") {
      sendResponse && sendResponse(tcBestImageSource(msg.fallbackUrl));
    }
    return false;
  });
})();
