const REFRESH_MS = 300000; // matches the scanner's 5-minute scan cycle — no point polling faster than the data changes
let refreshTimer = null;
let lastOverview = null;
let currentDetailSymbol = null;
let currentDetailTimeframe = null;

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
    body.innerHTML = `<tr><td colspan="12" class="muted">No symbols match the current filter.</td></tr>`;
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
      <td class="winrate-cell muted" data-wr-symbol="${s.symbol}">…</td>
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

  loadWinRates(rows.map((s) => s.symbol));
}

// Free, LLM-free win rate for every visible row — fetched with a small
// concurrency limit so 60 coins don't fire 60 simultaneous requests at once.
const WIN_RATE_CONCURRENCY = 6;

async function loadWinRates(symbols) {
  let cursor = 0;
  async function worker() {
    while (cursor < symbols.length) {
      const symbol = symbols[cursor++];
      await loadWinRateCell(symbol);
    }
  }
  const workers = Array.from({ length: Math.min(WIN_RATE_CONCURRENCY, symbols.length) }, worker);
  await Promise.all(workers);
}

async function loadWinRateCell(symbol) {
  const cell = document.querySelector(`.winrate-cell[data-wr-symbol="${symbol}"]`);
  if (!cell) return; // table may have re-rendered already
  try {
    const resp = await fetch(`/api/symbol/${symbol}/calibration`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const liveCell = document.querySelector(`.winrate-cell[data-wr-symbol="${symbol}"]`);
    if (!liveCell) return;
    if (!data.matches.length) {
      liveCell.className = "winrate-cell muted";
      liveCell.textContent = "n/a";
      return;
    }
    // Most statistically significant match (largest sample) leads the cell.
    const best = [...data.matches].sort((a, b) => b.n - a.n)[0];
    const winClass = best.win_rate_pct === null ? "" : best.win_rate_pct >= 55 ? "pos" : best.win_rate_pct <= 45 ? "neg" : "";
    const lowN = best.n < 30 ? " ⚠" : "";
    liveCell.className = "winrate-cell";
    liveCell.innerHTML = `<span class="${winClass}">${fmt(best.win_rate_pct, 1)}%</span> <span class="muted">(n=${best.n}${lowN})</span>`;
    liveCell.title = data.matches
      .map((m) => `${m.label} → ${m.direction}: ${fmt(m.win_rate_pct, 1)}% (n=${m.n}, avg net ${fmt(m.avg_net_return_pct, 2)}%)`)
      .join("\n");
  } catch (err) {
    const liveCell = document.querySelector(`.winrate-cell[data-wr-symbol="${symbol}"]`);
    if (liveCell) {
      liveCell.className = "winrate-cell muted";
      liveCell.textContent = "—";
    }
  }
}

function showDetail(symbolRow) {
  currentDetailSymbol = symbolRow.symbol;
  currentDetailTimeframe = symbolRow.reference_timeframe;
  document.getElementById("detail-title").textContent = symbolRow.symbol;
  document.getElementById("detail-json").textContent = JSON.stringify(symbolRow.snapshot, null, 2);
  document.getElementById("analysis-result").classList.add("hidden");
  document.getElementById("analysis-result").innerHTML = "";
  document.getElementById("analyze-status").textContent = "";
  document.getElementById("detail-modal").classList.remove("hidden");
  loadCalibrationPreview(symbolRow.symbol);
}

async function loadCalibrationPreview(symbol) {
  const el = document.getElementById("calibration-preview");
  el.className = "calibration-preview muted";
  el.textContent = "Checking historical win rate (free, no AI)…";
  try {
    const resp = await fetch(`/api/symbol/${symbol}/calibration`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    // A stale click (user already moved to a different row) must not paint
    // over the row now actually open.
    if (symbol !== currentDetailSymbol) return;

    if (!data.matches.length) {
      el.className = "calibration-preview muted";
      el.innerHTML = `No historical calibration data yet for: ${data.active_labels.join(", ")}.`;
      return;
    }

    el.className = "calibration-preview";
    el.innerHTML = data.matches
      .map((m) => {
        const lowN = m.n < 30;
        const winClass = m.win_rate_pct === null ? "" : m.win_rate_pct >= 55 ? "pos" : m.win_rate_pct <= 45 ? "neg" : "";
        return (
          `<div><b>${m.label}</b> → ${m.direction}: win rate ` +
          `<span class="${winClass}"><b>${fmt(m.win_rate_pct, 1)}%</b></span> ` +
          `(n=${m.n}${lowN ? ", small sample" : ""}, avg net ${fmt(m.avg_net_return_pct, 2)}%)</div>`
        );
      })
      .join("");
  } catch (err) {
    if (symbol !== currentDetailSymbol) return;
    el.className = "calibration-preview muted";
    el.textContent = "Could not load historical win rate.";
  }
}

function hideDetail() {
  document.getElementById("detail-modal").classList.add("hidden");
}

function renderAssessment(data) {
  const a = data.assessment;
  const biasClass = a.bias === "long" ? "trend-bullish" : a.bias === "short" ? "trend-bearish" : "trend-neutral";

  const riskLevelsHtml = data.risk_levels
    ? `<div class="risk-levels">
        <span>Entry: <b>${fmt(data.risk_levels.entry, 4)}</b></span>
        <span>SL: <b class="neg">${fmt(data.risk_levels.stop_loss, 4)}</b> (${fmt(data.risk_levels.risk_pct, 2)}%)</span>
        <span>TP: <b class="pos">${fmt(data.risk_levels.take_profit, 4)}</b> (${fmt(data.risk_levels.reward_pct, 2)}%)</span>
        <span>R:R <b>${fmt(data.risk_levels.risk_reward_ratio, 2)}</b></span>
      </div>`
    : `<p class="muted">No directional risk levels — trend is neutral.</p>`;

  const calibHtml = data.calibration_matches.length
    ? data.calibration_matches
        .map(
          (c) =>
            `<li>${c.label} → ${c.direction}: win rate <b>${fmt(c.win_rate_pct, 1)}%</b> (n=${c.n}, avg net ${fmt(c.avg_net_return_pct, 2)}%)</li>`
        )
        .join("")
    : `<li class="muted">No historical calibration data for the active label(s).</li>`;

  document.getElementById("analysis-result").innerHTML = `
    <div class="assessment-bias ${biasClass}">Bias: ${a.bias.toUpperCase()}</div>
    ${riskLevelsHtml}
    <h4>Historical basis</h4>
    <p>${a.historical_basis}</p>
    <h4>Calibration matches (ground truth, not a guess)</h4>
    <ul>${calibHtml}</ul>
    <h4>Key observations</h4>
    <ul>${a.key_observations.map((o) => `<li>${o}</li>`).join("")}</ul>
    <h4>Risk factors</h4>
    <ul>${a.risk_factors.map((o) => `<li>${o}</li>`).join("")}</ul>
    <h4>Risk/reward commentary</h4>
    <p>${a.risk_reward_commentary}</p>
    <h4>Caveats</h4>
    <p class="muted">${a.caveats}</p>
    <p class="muted" style="font-size: 11px; margin-top: 10px;">Diverges from BTC's reference trend: ${data.diverges_from_btc}</p>
  `;
  document.getElementById("analysis-result").classList.remove("hidden");
}

async function analyzeSymbol() {
  if (!currentDetailSymbol) return;
  const selected = document.getElementById("analyze-model").value;
  // Dropdown values are "provider:model" (e.g. "groq:llama-3.3-70b-versatile"
  // or "ollama:llama3.1:8b") — split on the first colon only, since Ollama
  // model names themselves contain a colon (tag separator).
  const sep = selected.indexOf(":");
  const provider = selected.slice(0, sep);
  const model = selected.slice(sep + 1);
  const btn = document.getElementById("analyze-btn");
  const status = document.getElementById("analyze-status");
  btn.disabled = true;
  status.textContent =
    provider === "ollama"
      ? "Thinking… (local CPU inference can take a few minutes)"
      : "Thinking… (cloud inference, usually a few seconds)";
  try {
    const url = `/api/analyze/${currentDetailSymbol}?timeframe=${encodeURIComponent(currentDetailTimeframe)}&provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`;
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderAssessment(data);
    status.textContent = "Done.";
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    btn.disabled = false;
  }
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
document.getElementById("analyze-btn").addEventListener("click", analyzeSymbol);

refresh();
scheduleAutoRefresh();
