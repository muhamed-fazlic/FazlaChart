const REFRESH_MS = 15000;
let refreshTimer = null;
let lastOverview = null;

function trendClass(trend) {
  if (trend === "bullish") return "trend-bullish";
  if (trend === "bearish") return "trend-bearish";
  return "trend-neutral";
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

function signClass(value) {
  if (value === null || value === undefined) return "";
  return value > 0 ? "pos" : value < 0 ? "neg" : "";
}

function fmtAge(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${(seconds / 3600).toFixed(1)}h ago`;
}

function renderBtcBanner(btc) {
  const el = document.getElementById("btc-banner");
  if (!btc) {
    el.textContent = "No BTC data yet — is the scanner running?";
    return;
  }
  const trends = Object.entries(btc.trend_by_timeframe || {})
    .map(([tf, trend]) => `${tf}: <span class="${trendClass(trend)}">${trend}</span>`)
    .join(" &nbsp;·&nbsp; ");
  const anomalies = (btc.anomalies || [])
    .map((a) => `<span class="anomaly-badge">${a}</span>`)
    .join("");
  el.innerHTML =
    `<span><b>BTC</b> ${fmt(btc.close, 2)}</span>` +
    `<span>${trends}</span>` +
    `<span>funding: ${fmt((btc.funding_rate || 0) * 100, 4)}%</span>` +
    `<span>OI Δ60m: <span class="${signClass(btc.open_interest_change_pct_60m)}">${fmt(btc.open_interest_change_pct_60m)}%</span></span>` +
    (anomalies ? `<span>${anomalies}</span>` : "");
}

function renderTable(symbols, anomaliesOnly) {
  const body = document.getElementById("scan-body");
  const rows = anomaliesOnly ? symbols.filter((s) => s.anomalies.length > 0) : symbols;

  if (rows.length === 0) {
    body.innerHTML = `<tr><td colspan="11" class="muted">No symbols match the current filter.</td></tr>`;
    return;
  }

  body.innerHTML = rows
    .map(
      (s, idx) => `
    <tr data-idx="${idx}" data-symbol="${s.symbol}">
      <td><b>${s.symbol}</b></td>
      <td class="${trendClass(s.trend)}">${s.trend}</td>
      <td>${fmt(s.close, 4)}</td>
      <td>${fmt(s.rsi, 1)}</td>
      <td>${fmt(s.ema20, 4)} / ${fmt(s.ema50, 4)}</td>
      <td class="${signClass(s.volume_change_pct)}">${fmt(s.volume_change_pct, 1)}%</td>
      <td>${fmt((s.funding_rate || 0) * 100, 4)}%</td>
      <td class="${signClass(s.open_interest_change_pct_60m)}">${fmt(s.open_interest_change_pct_60m, 2)}%</td>
      <td>${fmt(s.support, 4)} / ${fmt(s.resistance, 4)}</td>
      <td>${s.anomalies.map((a) => `<span class="anomaly-badge">${a}</span>`).join("") || "—"}</td>
      <td class="muted">${fmtAge(s.age_seconds)}</td>
    </tr>`
    )
    .join("");

  Array.from(body.querySelectorAll("tr[data-symbol]")).forEach((tr) => {
    tr.addEventListener("click", () => {
      const idx = Number(tr.dataset.idx);
      showDetail(rows[idx]);
    });
  });
}

function showDetail(symbolRow) {
  document.getElementById("detail-title").textContent = symbolRow.symbol;
  document.getElementById("detail-json").textContent = JSON.stringify(symbolRow.snapshot, null, 2);
  document.getElementById("detail-modal").classList.remove("hidden");
}

function hideDetail() {
  document.getElementById("detail-modal").classList.add("hidden");
}

async function refresh() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();
    lastOverview = data;
    document.getElementById("generated-at").textContent = data.generated_at
      ? `last scan: ${data.generated_at}`
      : "no scans yet";
    renderBtcBanner(data.btc);
    const anomaliesOnly = document.getElementById("anomalies-only").checked;
    renderTable(data.symbols, anomaliesOnly);
  } catch (err) {
    document.getElementById("generated-at").textContent = "connection error — is the dashboard server running?";
    console.error(err);
  }
}

function scheduleAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  if (document.getElementById("auto-refresh").checked) {
    refreshTimer = setInterval(refresh, REFRESH_MS);
  }
}

document.getElementById("anomalies-only").addEventListener("change", () => {
  if (lastOverview) renderTable(lastOverview.symbols, document.getElementById("anomalies-only").checked);
});
document.getElementById("auto-refresh").addEventListener("change", scheduleAutoRefresh);
document.getElementById("detail-close").addEventListener("click", hideDetail);
document.getElementById("detail-modal").addEventListener("click", (e) => {
  if (e.target.id === "detail-modal") hideDetail();
});

refresh();
scheduleAutoRefresh();
