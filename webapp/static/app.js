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

// --- Top Signal Now (statistically-grounded shortlist) --------------------
// Distinct from the heuristic "top movers" table: only ever shows a symbol
// here if its active label's calibration win rate clears 50% at the 95%
// confidence interval's conservative lower bound with n>=30 (see
// /api/scan/best-signal) — never a raw point estimate, never a guess.

function winRateRangeHtml(low, high) {
  if (low === null || low === undefined || high === null || high === undefined) return "";
  return `[${fmt(low, 1)}–${fmt(high, 1)}%]`;
}

function bestSignalEntryHtml(sig) {
  if (!sig.entry) return "";
  const kelly = sig.kelly_sizing;
  return `
    <div class="best-signal-entry">
      <span>Entry zona: <b>${fmt(sig.entry.entry_zone_low, 4)}–${fmt(sig.entry.entry_zone_high, 4)}</b></span>
      <span>SL: ${fmt(sig.entry.stop_loss, 4)}</span>
      <span>TP: ${fmt(sig.entry.take_profit, 4)}${sig.entry.take_profit_conservative !== undefined && sig.entry.take_profit_conservative !== null ? ` <span class="muted">(konzervativan: ${fmt(sig.entry.take_profit_conservative, 4)})</span>` : ""}</span>
      <span>R:R ${fmt(sig.entry.risk_reward_ratio, 2)}</span>
      ${kelly ? `<span>💰 Kelly: <b>${fmt(kelly.recommended_fraction_pct, 1)}%</b> kapitala</span>` : ""}
    </div>
    ${sig.entry.extension_warning ? `<div class="best-signal-warning">⏳ ${sig.entry.extension_warning}</div>` : ""}
    ${sig.entry.volatility_warning ? `<div class="best-signal-warning">⚠️ ${sig.entry.volatility_warning}</div>` : ""}
    ${kelly && kelly.warning ? `<div class="best-signal-warning">💰 ${kelly.warning}</div>` : ""}`;
}

function bestSignalCardHtml(sig, rank) {
  const dirClass = sig.direction === "long" ? "trend-bullish" : "trend-bearish";
  const dirWord = sig.direction === "long" ? "LONG" : "SHORT";
  return `
    <div class="best-signal-card${rank === 0 ? " best-signal-top" : ""}" data-symbol="${sig.symbol}">
      <div class="best-signal-head">
        <span class="best-signal-rank">#${rank + 1}</span>
        <b>${sig.symbol}</b>
        <span class="${dirClass}">${dirWord}</span>
        <span class="muted">${sig.timeframe}</span>
      </div>
      <div class="best-signal-stats">
        <span>Win rate: <b>${fmt(sig.win_rate_pct, 1)}%</b> ${winRateRangeHtml(sig.win_rate_ci_low_pct, sig.win_rate_ci_high_pct)}</span>
        <span class="muted">n=${sig.n}</span>
        <span class="${signClass(sig.avg_net_return_pct)}">avg net: ${fmt(sig.avg_net_return_pct, 2)}%</span>
        ${sig.regime ? `<span class="muted">regime: ${sig.regime}</span>` : ""}
      </div>
      <div class="muted best-signal-label">${sig.label}</div>
      ${sig.correlation ? `<div class="best-signal-warning">🔗 ${sig.correlation.note}</div>` : ""}
      ${bestSignalEntryHtml(sig)}
    </div>`;
}

function renderBestSignalPanel(data) {
  const el = document.getElementById("best-signal-panel");
  if (!data || !data.signals || !data.signals.length) {
    el.className = "best-signal-panel muted";
    el.innerHTML = data && data.candidates_considered
      ? `Nema trenutno statistički potvrđenog signala (pregledano ${data.candidates_considered} coina sa trendom, ` +
        `keš kalibracije spreman za ${data.cache_coverage || "0/0"}). Keš se puni u pozadini — pokušajte opet za koji minut.`
      : "Nema podataka još — sačekajte da skener i keš kalibracije završe prvi ciklus.";
    return;
  }
  el.className = "best-signal-panel";
  const [top, ...rest] = data.signals;
  el.innerHTML =
    `<div class="best-signal-title">🎯 Top signal sada — statistički potvrđen (n≥30, donja granica 95% CI &gt; 50%)</div>` +
    bestSignalCardHtml(top, 0) +
    (rest.length ? `<div class="best-signal-runners">${rest.map((s, i) => bestSignalCardHtml(s, i + 1)).join("")}</div>` : "");
  el.querySelectorAll(".best-signal-card").forEach((cardEl) => {
    cardEl.addEventListener("click", () => {
      document.getElementById("symbol-filter-input").value = cardEl.dataset.symbol;
      lookupSymbol(cardEl.dataset.symbol);
    });
  });
}

async function loadBestSignal() {
  try {
    const res = await fetch("/api/scan/best-signal?limit=4");
    const data = await res.json();
    renderBestSignalPanel(data);
  } catch (err) {
    console.error("Could not load best signal:", err);
  }
}

// --- Forward tracking (closed-loop validation: promised vs. realized) -----
// Every Top Signal Now candidate gets logged once (scanner/signal_tracking.py)
// and resolved against real price action as it plays out live — this panel
// shows whether the calibration's promised win rate is actually holding up,
// not just a backtest claim.

function renderForwardTrackingPanel(data) {
  const el = document.getElementById("forward-tracking-panel");
  if (!data) {
    el.className = "forward-tracking-panel muted";
    el.textContent = "Nema podataka o praćenju uživo još.";
    return;
  }
  if (data.resolved_count === 0) {
    el.className = "forward-tracking-panel muted";
    el.textContent = `📡 Praćenje uživo: ${data.open_count} signala u toku, još nema zatvorenih — rezultati stižu kako se SL/TP/timeout dešavaju.`;
    return;
  }
  el.className = "forward-tracking-panel";
  const byStatus = data.resolved_by_status || {};
  const realized = data.realized_win_rate_pct;
  const promised = data.avg_promised_win_rate_ci_low_pct;
  const gap = realized !== null && promised !== null ? realized - promised : null;
  const gapClass = gap === null ? "" : gap >= 0 ? "pos" : "neg";
  el.innerHTML =
    `<div class="forward-tracking-title">📡 Praćenje uživo (obećano vs. stvarno)</div>` +
    `<div class="forward-tracking-stats">` +
    `<span>Zatvoreno: <b>${data.resolved_count}</b> (TP ${byStatus.take_profit || 0} · SL ${byStatus.stop_loss || 0} · timeout ${byStatus.timeout || 0})</span>` +
    `<span>U toku: <b>${data.open_count}</b></span>` +
    `<span>Stvarni win rate: <b>${fmt(realized, 1)}%</b></span>` +
    `<span>Obećana donja granica (prosjek): <b>${fmt(promised, 1)}%</b></span>` +
    (gap !== null ? `<span class="${gapClass}">razlika: ${gap >= 0 ? "+" : ""}${fmt(gap, 1)}pp</span>` : "") +
    `</div>`;
}

async function loadForwardTracking() {
  try {
    const res = await fetch("/api/forward-tracking/summary");
    const data = await res.json();
    renderForwardTrackingPanel(data);
  } catch (err) {
    console.error("Could not load forward tracking summary:", err);
  }
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
      <td data-label="Symbol"><b>${s.symbol}</b></td>
      <td data-label="Trend" class="${trendClass(s.trend)}">${s.trend}</td>
      <td data-label="Close">${fmt(s.close, 4)}</td>
      <td data-label="RSI">${fmt(s.rsi, 1)}</td>
      <td data-label="EMA20 / EMA50">${fmt(s.ema20, 4)} / ${fmt(s.ema50, 4)}</td>
      <td data-label="Vol Δ%" class="${signClass(s.volume_change_pct)}">${fmt(s.volume_change_pct, 1)}%</td>
      <td data-label="Funding">${fmt((s.funding_rate || 0) * 100, 4)}%</td>
      <td data-label="OI Δ60m%" class="${signClass(s.open_interest_change_pct_60m)}">${fmt(s.open_interest_change_pct_60m, 2)}%</td>
      <td data-label="Support / Resistance">${fmt(s.support, 4)} / ${fmt(s.resistance, 4)}</td>
      <td data-label="Anomalies">${s.anomalies.map((a) => `<span class="anomaly-badge">${a}</span>`).join("") || "—"}</td>
      <td data-label="Win rate" class="winrate-cell muted" data-wr-symbol="${s.symbol}">…</td>
      <td data-label="Age" class="muted">${fmtAge(s.age_seconds)}</td>
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

// renderTable() (and therefore loadWinRates()) can fire again — e.g. toggling
// "anomalies only", or auto-refresh landing — before a previous pass has
// finished fetching all rows. Without a guard, each new pass starts its own
// full set of workers on top of whatever's still in flight, so the number of
// concurrent calibration requests keeps growing instead of staying capped at
// WIN_RATE_CONCURRENCY. This generation counter makes any in-progress pass
// stop issuing further fetches as soon as a newer pass starts.
let winRateLoadGeneration = 0;

async function loadWinRates(symbols) {
  const generation = ++winRateLoadGeneration;
  let cursor = 0;
  async function worker() {
    while (cursor < symbols.length) {
      if (generation !== winRateLoadGeneration) return; // a newer pass superseded this one
      const symbol = symbols[cursor++];
      await loadWinRateCell(symbol);
    }
  }
  const workers = Array.from({ length: Math.min(WIN_RATE_CONCURRENCY, symbols.length) }, worker);
  await Promise.all(workers);
}

// Builds the win-rate cell's class + inner HTML + tooltip from a list of
// calibration matches. Shared by the per-row background loader
// (loadWinRateCell, fetched from /api/symbol/{symbol}/calibration) and the
// single-symbol filter (which already gets matches bundled in the
// /api/symbol/{symbol}/lookup response, so it never needs this extra fetch).
function winRateCellHtml(matches) {
  if (!matches || !matches.length) {
    return { className: "winrate-cell muted", html: "n/a", title: "" };
  }
  // Most statistically significant match (largest sample) leads the cell.
  const best = [...matches].sort((a, b) => b.n - a.n)[0];
  const lowN = best.n < 30 ? " ⚠" : "";
  const title = matches
    .map((m) =>
      m.win_rate_pct === null
        ? `${m.label}: nema definisan smjer (long/short), pa se ne računa win rate — prosječan pomak ${fmt(m.avg_net_return_pct, 2)}% (n=${m.n})`
        : `${m.label} → ${m.direction}: ${fmt(m.win_rate_pct, 1)}% [95% CI ${fmt(m.win_rate_ci_low_pct, 1)}–${fmt(m.win_rate_ci_high_pct, 1)}%] (n=${m.n}, avg net ${fmt(m.avg_net_return_pct, 2)}%)`
    )
    .join("\n");
  if (best.win_rate_pct === null) {
    // Undirected label (e.g. neutral trend): no win/loss concept applies, so
    // show the sample size and average magnitude of the move instead of an
    // invented win rate.
    return {
      className: "winrate-cell muted",
      html: `bez smjera <span class="muted">(n=${best.n}${lowN})</span>`,
      title,
    };
  }
  // Colour the cell by the confidence interval, not the raw point estimate:
  // a 65% win rate whose 95% CI still straddles 50% (coin-flip) is not
  // trustworthy evidence of an edge, even though the point number "looks
  // good" — only colour it pos/neg once the whole interval clears 50%.
  const ciLow = best.win_rate_ci_low_pct;
  const ciHigh = best.win_rate_ci_high_pct;
  const winClass = ciLow !== null && ciLow >= 50 ? "pos" : ciHigh !== null && ciHigh <= 50 ? "neg" : "";
  const ciLabel = ciLow !== null ? ` <span class="muted">[${fmt(ciLow, 0)}–${fmt(ciHigh, 0)}]</span>` : "";
  return {
    className: "winrate-cell",
    html: `<span class="${winClass}">${fmt(best.win_rate_pct, 1)}%</span>${ciLabel} <span class="muted">(n=${best.n}${lowN})</span>`,
    title,
  };
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
    const { className, html, title } = winRateCellHtml(data.matches);
    liveCell.className = className;
    liveCell.innerHTML = html;
    liveCell.title = title;
  } catch (err) {
    const liveCell = document.querySelector(`.winrate-cell[data-wr-symbol="${symbol}"]`);
    if (liveCell) {
      liveCell.className = "winrate-cell muted";
      liveCell.textContent = "—";
    }
  }
}

function fmtPct(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  const cls = signClass(value);
  return `<span class="${cls}">${fmt(value, digits)}%</span>`;
}

function renderTimeframeCard(tf, data) {
  const anomalies = (data.anomalies || []).map((a) => `<span class="anomaly-badge">${a}</span>`).join("") || `<span class="muted">—</span>`;
  const fib = data.fibonacci || {};
  const retracement = Object.entries(fib.retracement || {})
    .map(([ratio, level]) => `<span>${ratio}: <b>${fmt(level, 4)}</b></span>`)
    .join(" &nbsp; ");
  const extension = Object.entries(fib.extension || {})
    .map(([ratio, level]) => `<span>${ratio}: <b>${fmt(level, 4)}</b></span>`)
    .join(" &nbsp; ");

  return `
    <div class="tf-card">
      <div class="tf-card-header">
        <b>${tf}</b>
        <span class="${trendClass(data.trend)}">${data.trend}</span>
        <span class="muted">close: ${fmt(data.close, 4)}</span>
      </div>
      <table class="tf-table">
        <tr><td>RSI</td><td>${fmt(data.rsi, 1)}</td><td>EMA cross</td><td>${data.ema_cross || "—"}</td></tr>
        <tr><td>EMA9 / 20 / 21 / 50 / 200</td><td colspan="3">${fmt(data.ema9, 4)} / ${fmt(data.ema20, 4)} / ${fmt(data.ema21, 4)} / ${fmt(data.ema50, 4)} / ${fmt(data.ema200, 4)}</td></tr>
        <tr><td>MACD / signal / hist</td><td colspan="3">${fmt(data.macd, 5)} / ${fmt(data.macd_signal, 5)} / ${fmt(data.macd_hist, 5)}</td></tr>
        <tr><td>ATR</td><td>${fmt(data.atr, 4)}</td><td>Bollinger lower/upper</td><td>${fmt(data.bollinger_lower, 4)} / ${fmt(data.bollinger_upper, 4)}</td></tr>
        <tr><td>Volume</td><td>${fmt(data.volume, 1)}</td><td>Rel. volume / Δ%</td><td>${fmt(data.relative_volume, 2)}x / ${fmtPct(data.volume_change_pct, 1)}</td></tr>
        <tr><td>Support / Resistance</td><td colspan="3">${fmt(data.support, 4)} / ${fmt(data.resistance, 4)}</td></tr>
        <tr><td>Drawdown / Momentum fading</td><td colspan="3">${fmtPct(data.drawdown_pct, 1)} / ${data.momentum_fading === null || data.momentum_fading === undefined ? "—" : data.momentum_fading}</td></tr>
      </table>
      <div class="anomaly-badges">${anomalies}</div>
      ${
        fib.swing_high
          ? `<details class="fib-toggle"><summary class="muted">Fibonacci (swing ${fib.swing_direction || "—"}: ${fmt(fib.swing_low, 4)} - ${fmt(fib.swing_high, 4)})</summary>
              <div class="fib-levels"><b>Retracement:</b> ${retracement || "—"}</div>
              <div class="fib-levels"><b>Extension:</b> ${extension || "—"}</div>
            </details>`
          : ""
      }
    </div>`;
}

function renderDerivativesAndPivots(snapshot) {
  const d = snapshot.derivatives || {};
  const p = snapshot.pivot_points || {};
  const derivativesHtml = `
    <table class="tf-table">
      <tr><td>Funding rate</td><td>${fmtPct((d.funding_rate || 0) * 100, 4)}</td><td>Mark / Index price</td><td>${fmt(d.mark_price, 4)} / ${fmt(d.index_price, 4)}</td></tr>
      <tr><td>Open interest</td><td>${fmt(d.open_interest, 1)}</td><td>OI Δ 5m / 15m / 60m</td><td>${fmtPct(d.open_interest_change_pct_5m, 2)} / ${fmtPct(d.open_interest_change_pct_15m, 2)} / ${fmtPct(d.open_interest_change_pct_60m, 2)}</td></tr>
    </table>`;
  const pivotHtml = Object.keys(p).length
    ? `<table class="tf-table">
        <tr><td>P</td><td>${fmt(p.pivot, 4)}</td><td>R1 / S1</td><td>${fmt(p.r1, 4)} / ${fmt(p.s1, 4)}</td></tr>
        <tr><td>R2 / S2</td><td>${fmt(p.r2, 4)} / ${fmt(p.s2, 4)}</td><td>R3 / S3</td><td>${fmt(p.r3, 4)} / ${fmt(p.s3, 4)}</td></tr>
      </table>`
    : `<p class="muted">No pivot points yet (needs a completed daily candle).</p>`;
  return `${derivativesHtml}<h4 class="muted" style="margin-top: 10px;">Daily pivot points</h4>${pivotHtml}`;
}

function renderSnapshotDetails(snapshot) {
  const timeframes = snapshot.timeframes || {};
  document.getElementById("detail-timeframes").innerHTML = Object.entries(timeframes)
    .map(([tf, data]) => renderTimeframeCard(tf, data))
    .join("");
  document.getElementById("detail-derivatives").innerHTML = renderDerivativesAndPivots(snapshot);
}

function renderHeatmap(data) {
  const bins = [...(data.price_bins || [])].sort((a, b) => b.price_low - a.price_low); // high price first
  const maxPct = Math.max(...bins.map((b) => b.pct_of_total), 0.0001);
  const current = data.current_price;

  const rows = bins
    .filter((b) => b.pct_of_total > 0.5) // skip near-empty bins so the list stays readable
    .map((b) => {
      const isPoc = Math.abs(((b.price_low + b.price_high) / 2) - data.poc_price) < (b.price_high - b.price_low);
      const isCurrent = current >= b.price_low && current <= b.price_high;
      const width = Math.max(2, (b.pct_of_total / maxPct) * 100);
      return `<div class="heatmap-bar-row${isPoc ? " is-poc" : ""}${isCurrent ? " is-current" : ""}">
        <span>${fmt(b.price_low, 4)}–${fmt(b.price_high, 4)}</span>
        <span class="heatmap-bar-track"><span class="heatmap-bar-fill" style="width:${width}%"></span></span>
        <span>${fmt(b.pct_of_total, 1)}%</span>
      </div>`;
    })
    .join("");

  const stopZones = (data.likely_stop_zones || [])
    .map((z) => `<li>${fmt(z.price, 4)} — ${z.reason}</li>`)
    .join("");

  const walls = data.orderbook_walls;
  const wallItem = (w) =>
    `<li class="${w.side === "bid" ? "pos" : "neg"}">${fmt(w.price, 4)} (${w.distance_pct > 0 ? "+" : ""}${fmt(w.distance_pct, 2)}%) — ${w.side === "bid" ? "bid" : "ask"} zid, ${fmt(w.notional_usdt / 1000, 1)}k USDT</li>`;
  const wallRows = walls
    ? [...walls.bid_walls, ...walls.ask_walls]
        .sort((a, b) => b.notional_usdt - a.notional_usdt)
        .slice(0, 6)
        .map(wallItem)
        .join("")
    : "";
  const orderbookSection = walls
    ? `
    <div class="muted" style="font-size: 11px; margin: 8px 0 2px;">Order-book zidovi (live, ne proxy)</div>
    <ul class="heatmap-stop-zones">${wallRows || "<li>Nema izraženih zidova u trenutnom snapshotu.</li>"}</ul>`
    : data.orderbook_walls_error
    ? `<div class="muted" style="font-size: 10px; margin-top: 4px;">Order-book podaci trenutno nedostupni — prikazan samo proxy.</div>`
    : "";

  return `
    <div class="muted" style="font-size: 11px; margin-bottom: 4px;">
      Current: <b>${fmt(current, 4)}</b> · POC (najprometnija zona): <b>${fmt(data.poc_price, 4)}</b> ·
      Value area: ${fmt(data.value_area_low, 4)} – ${fmt(data.value_area_high, 4)}
    </div>
    ${rows}
    <ul class="heatmap-stop-zones">${stopZones}</ul>
    ${orderbookSection}
    <div class="muted" style="font-size: 10px; margin-top: 4px;">${data.note || ""}</div>
  `;
}

async function loadLiquidityHeatmap(symbol, timeframe) {
  const el = document.getElementById("detail-heatmap");
  el.className = "heatmap-wrap muted";
  el.textContent = "Loading heatmap…";
  try {
    const resp = await fetch(`/api/symbol/${symbol}/heatmap?timeframe=${encodeURIComponent(timeframe)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (symbol !== currentDetailSymbol) return; // stale click guard, same pattern as calibration preview
    el.className = "heatmap-wrap";
    el.innerHTML = renderHeatmap(data);
  } catch (err) {
    if (symbol !== currentDetailSymbol) return;
    el.className = "heatmap-wrap muted";
    el.textContent = "Could not load heatmap.";
    console.error(err);
  }
}

function showDetail(symbolRow) {
  currentDetailSymbol = symbolRow.symbol;
  currentDetailTimeframe = symbolRow.reference_timeframe;
  document.getElementById("detail-title").textContent = symbolRow.symbol;
  renderSnapshotDetails(symbolRow.snapshot);
  document.getElementById("detail-json").textContent = JSON.stringify(symbolRow.snapshot, null, 2);
  document.getElementById("analysis-result").classList.add("hidden");
  document.getElementById("analysis-result").innerHTML = "";
  document.getElementById("analyze-status").textContent = "";
  document.getElementById("detail-modal").classList.remove("hidden");
  loadCalibrationPreview(symbolRow.symbol);
  loadLiquidityHeatmap(symbolRow.symbol, symbolRow.reference_timeframe);
}

// Out-of-sample validation: a label's historical win rate was measured over
// its whole history, but the edge could be an artifact of an older regime
// that's since changed. This re-checks the win rate on just the most recent
// held-out slice of trades (never used to compute the headline number) and
// flags plainly when that recent slice does NOT confirm the edge — this is
// a stronger trust signal than the overall win rate alone.
function oosValidationHtml(oosResults) {
  if (!oosResults || !oosResults.length) return "";
  const rows = oosResults
    .map((o) => {
      if (o.test_n < 20) {
        return (
          `<div class="muted">🧪 <b>${o.label}</b> → ${o.direction}: premalo novijih podataka ` +
          `za out-of-sample provjeru (n=${o.test_n}).</div>`
        );
      }
      const icon = o.holds_up ? "✅" : "⚠️";
      const cls = o.holds_up ? "pos" : "neg";
      return (
        `<div class="${cls}">${icon} <b>${o.label}</b> → ${o.direction}: train ${fmt(o.train_win_rate_pct, 1)}% ` +
        `(n=${o.train_n}) vs. najnoviji test ${fmt(o.test_win_rate_pct, 1)}% (n=${o.test_n}, CI-low ${fmt(o.test_win_rate_ci_low_pct, 1)}%)` +
        `${o.warning ? ` — ${o.warning}` : ""}</div>`
      );
    })
    .join("");
  return `<div class="oos-validation"><div class="muted">Out-of-sample provjera (edge na najnovijim, neviđenim trejdovima):</div>${rows}</div>`;
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
        const ciLow = m.win_rate_ci_low_pct;
        const ciHigh = m.win_rate_ci_high_pct;
        const winClass =
          m.win_rate_pct === null ? "" : ciLow !== null && ciLow >= 50 ? "pos" : ciHigh !== null && ciHigh <= 50 ? "neg" : "";
        const ciLabel = ciLow !== null ? ` [95% CI ${fmt(ciLow, 0)}–${fmt(ciHigh, 0)}%]` : "";
        return (
          `<div><b>${m.label}</b> → ${m.direction}: win rate ` +
          `<span class="${winClass}"><b>${fmt(m.win_rate_pct, 1)}%</b></span>${ciLabel} ` +
          `(n=${m.n}${lowN ? ", small sample" : ""}, avg net ${fmt(m.avg_net_return_pct, 2)}%)</div>`
        );
      })
      .join("") + oosValidationHtml(data.out_of_sample);
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

  const mtfRows = Object.entries(data.multi_timeframe_risk_levels || {})
    .map(([tf, lv]) => {
      if (lv.bias === "neutral") {
        return `<tr><td><b>${tf}</b></td><td class="trend-neutral" colspan="5">neutralno — nema smjera, nema SL/TP</td></tr>`;
      }
      const biasCls = lv.bias === "long" ? "trend-bullish" : "trend-bearish";
      const extensionWarnRow = lv.extension_warning
        ? `<tr><td></td><td class="neg" colspan="5" style="font-size:11px;">⏳ ${lv.extension_warning}</td></tr>`
        : "";
      const warnRow = lv.volatility_warning
        ? `<tr><td></td><td class="neg" colspan="5" style="font-size:11px;">⚠️ ${lv.volatility_warning}</td></tr>`
        : "";
      const chopWarnRow = lv.chop_zone_warning
        ? `<tr><td></td><td class="neg" colspan="5" style="font-size:11px;">🧲 ${lv.chop_zone_warning}</td></tr>`
        : "";
      const wallWarnRow = lv.orderbook_wall_warning
        ? `<tr><td></td><td class="neg" colspan="5" style="font-size:11px;">🧱 ${lv.orderbook_wall_warning}</td></tr>`
        : "";
      const kellyRow = lv.kelly_sizing
        ? `<tr><td></td><td colspan="5" style="font-size:11px;">💰 Kelly veličina pozicije: <b>${fmt(lv.kelly_sizing.recommended_fraction_pct, 1)}%</b> kapitala` +
          (lv.kelly_sizing.warning ? ` <span class="muted">— ${lv.kelly_sizing.warning}</span>` : "") +
          `</td></tr>`
        : "";
      const conservativeTp =
        lv.take_profit_conservative !== undefined && lv.take_profit_conservative !== null
          ? ` <span class="muted">(konzervativan: ${fmt(lv.take_profit_conservative, 4)})</span>`
          : "";
      const patientEntry =
        lv.patient_entry !== undefined && lv.patient_entry !== null
          ? ` <span class="muted">(strpljiv: ${fmt(lv.patient_entry, 4)})</span>`
          : "";
      return `<tr>
        <td><b>${tf}</b></td>
        <td class="${biasCls}">${lv.bias === "long" ? "LONG" : "SHORT"}</td>
        <td>${fmt(lv.entry, 4)}${patientEntry}</td>
        <td class="neg">${fmt(lv.stop_loss, 4)} <span class="muted">(${fmt(lv.risk_pct, 2)}%)</span></td>
        <td class="pos">${fmt(lv.take_profit, 4)} <span class="muted">(${fmt(lv.reward_pct, 2)}%)</span>${conservativeTp}</td>
        <td>${fmt(lv.risk_reward_ratio, 2)}</td>
      </tr>${extensionWarnRow}${warnRow}${chopWarnRow}${wallWarnRow}${kellyRow}`;
    })
    .join("");
  const riskLevelsHtml = mtfRows
    ? `<table class="tf-table risk-levels-table">
        <thead><tr><th>TF</th><th>Bias</th><th>Entry</th><th>SL</th><th>TP</th><th>R:R</th></tr></thead>
        <tbody>${mtfRows}</tbody>
      </table>`
    : `<p class="muted">No directional risk levels — all timeframes are neutral.</p>`;

  const calibHtml = data.calibration_matches.length
    ? data.calibration_matches
        .map((c) => {
          const ciLabel =
            c.win_rate_ci_low_pct !== null && c.win_rate_ci_low_pct !== undefined
              ? ` [95% CI ${fmt(c.win_rate_ci_low_pct, 0)}–${fmt(c.win_rate_ci_high_pct, 0)}%]`
              : "";
          return `<li>${c.label} → ${c.direction}: win rate <b>${fmt(c.win_rate_pct, 1)}%</b>${ciLabel} (n=${c.n}, avg net ${fmt(c.avg_net_return_pct, 2)}%)</li>`;
        })
        .join("")
    : `<li class="muted">No historical calibration data for the active label(s).</li>`;

  const s = data.sentiment;
  const sentimentHtml = s
    ? s.label
      ? `<span class="${s.label === "bullish" ? "pos" : s.label === "bearish" ? "neg" : "muted"}">${s.label.toUpperCase()}</span>` +
        ` — StockTwits: ${s.bullish_count} bullish / ${s.bearish_count} bearish / ${s.unlabeled_count} unlabeled (${s.total_messages} msgs)`
      : `<span class="muted">Nema dovoljno podataka — ${s.insufficient_reason}</span>`
    : `<span class="muted">Sentiment nije dohvaćen.</span>`;

  document.getElementById("analysis-result").innerHTML = `
    <div class="assessment-bias ${biasClass}">Bias: ${a.bias.toUpperCase()}</div>
    <h4>Entry / SL / TP po timeframe-u (ATR-based)</h4>
    ${riskLevelsHtml}
    <h4>Historical basis</h4>
    <p>${a.historical_basis}</p>
    <h4>Calibration matches (ground truth, not a guess)</h4>
    <ul>${calibHtml}</ul>
    <h4>Social sentiment (StockTwits, raw tags — not a prediction)</h4>
    <p>${sentimentHtml}</p>
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

// Which data source currently feeds the table + auto-refresh:
// "top10"  — default, the N strongest bullish/bearish symbols (cheap, DB-only)
// "all"    — show-all-toggle checked, every tracked symbol
// "single" — user looked up one specific symbol, no auto-refresh
let currentMode = "top10";

function scheduleAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  if (!document.getElementById("auto-refresh").checked) return;
  const loader = currentMode === "all" ? refresh : currentMode === "top10" ? loadTopMovers : null;
  if (loader) refreshTimer = setInterval(loader, REFRESH_MS);
}

// --- Top movers (default view) --------------------------------------------
// Instead of showing nothing until the user types a symbol, default to the
// N tracked coins with the strongest *current* bullish/bearish structure
// (/api/scan/top) — still DB-only, no live Binance calls or extra
// backtests beyond the win-rate for the rows actually shown.
const TOP_MOVERS_LIMIT = 10;

async function loadTopMovers() {
  try {
    const res = await fetch(`/api/scan/top?limit=${TOP_MOVERS_LIMIT}`);
    const data = await res.json();
    lastOverview = data;
    document.getElementById("generated-at").textContent = data.generated_at
      ? `last scan: ${data.generated_at}`
      : "no scans yet";
    renderBtcBanner(data.btc);
    document.getElementById("symbol-filter-status").textContent = data.symbols.length
      ? `Top ${data.symbols.length} coina sa najjačom bullish/bearish strukturom (osvježava se automatski). Ukucajte simbol za pojedinačnu provjeru.`
      : "Nema dovoljno podataka još — sačekajte da skener završi barem jedan ciklus.";
    const anomaliesOnly = document.getElementById("anomalies-only").checked;
    renderTable(data.symbols, anomaliesOnly);
  } catch (err) {
    document.getElementById("generated-at").textContent = "connection error — is the dashboard server running?";
    console.error(err);
  }
}

// --- Single-symbol filter -------------------------------------------------
// The user can always type one specific symbol to look it up directly via
// /api/symbol/{symbol}/lookup, which already bundles the calibration
// (win-rate) matches in the same response — no second round-trip.

async function populateSymbolSuggestions() {
  try {
    const res = await fetch("/api/symbols");
    const data = await res.json();
    const datalist = document.getElementById("symbol-suggestions");
    datalist.innerHTML = data.symbols.map((s) => `<option value="${s}"></option>`).join("");
  } catch (err) {
    console.error("Could not load symbol suggestions:", err);
  }
}

function renderSingleSymbolRow(s) {
  const body = document.getElementById("scan-body");
  const { className: wrClass, html: wrHtml, title: wrTitle } = winRateCellHtml(s.calibration_matches);
  const trackedNote = s.tracked
    ? ""
    : `<span class="muted" title="Nije u redovnom skeniranju (ispod top-N po volumenu) — podaci su dohvaćeni uživo.">(uživo, nije praćen)</span>`;
  body.innerHTML = `
    <tr data-idx="0" data-symbol="${s.symbol}">
      <td data-label="Symbol"><b>${s.symbol}</b> ${trackedNote}</td>
      <td data-label="Trend" class="${trendClass(s.trend)}">${s.trend}</td>
      <td data-label="Close">${fmt(s.close, 4)}</td>
      <td data-label="RSI">${fmt(s.rsi, 1)}</td>
      <td data-label="EMA20 / EMA50">${fmt(s.ema20, 4)} / ${fmt(s.ema50, 4)}</td>
      <td data-label="Vol Δ%" class="${signClass(s.volume_change_pct)}">${fmt(s.volume_change_pct, 1)}%</td>
      <td data-label="Funding">${fmt((s.funding_rate || 0) * 100, 4)}%</td>
      <td data-label="OI Δ60m%" class="${signClass(s.open_interest_change_pct_60m)}">${fmt(s.open_interest_change_pct_60m, 2)}%</td>
      <td data-label="Support / Resistance">${fmt(s.support, 4)} / ${fmt(s.resistance, 4)}</td>
      <td data-label="Anomalies">${(s.anomalies || []).map((a) => `<span class="anomaly-badge">${a}</span>`).join("") || "—"}</td>
      <td data-label="Win rate" class="${wrClass}" title="${wrTitle.replace(/"/g, "&quot;")}">${wrHtml}</td>
      <td data-label="Age" class="muted">${fmtAge(s.age_seconds)}</td>
    </tr>`;
  document.querySelector(`tr[data-symbol="${s.symbol}"]`).addEventListener("click", () => showDetail(s));
}

async function lookupSymbol(rawSymbol) {
  const symbol = rawSymbol.trim().toUpperCase();
  const status = document.getElementById("symbol-filter-status");
  const body = document.getElementById("scan-body");
  if (!symbol) return;
  currentMode = "single";
  if (refreshTimer) clearInterval(refreshTimer); // a looked-up symbol doesn't auto-refresh/get replaced by top movers
  status.textContent = `Učitavam ${symbol}…`;
  body.innerHTML = `<tr><td colspan="12" class="muted">Učitavam ${symbol}…</td></tr>`;
  try {
    const res = await fetch(`/api/symbol/${encodeURIComponent(symbol)}/lookup`);
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    lastOverview = { symbols: [data] };
    renderSingleSymbolRow(data);
    status.textContent = data.tracked ? "" : "Napomena: coin nije u redovnom skeniranju, podaci su dohvaćeni uživo.";
  } catch (err) {
    body.innerHTML = `<tr><td colspan="12" class="muted">Greška: ${err.message}</td></tr>`;
    status.textContent = "";
  }
}

function setFullTableMode(enabled) {
  if (enabled) {
    currentMode = "all";
    document.getElementById("symbol-filter-status").textContent = "";
    refresh();
  } else {
    currentMode = "top10";
    loadTopMovers();
  }
  scheduleAutoRefresh();
}

document.getElementById("symbol-filter-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (document.getElementById("show-all-toggle").checked) return; // full table already showing everything
  lookupSymbol(document.getElementById("symbol-filter-input").value);
});
document.getElementById("show-all-toggle").addEventListener("change", (e) => {
  setFullTableMode(e.target.checked);
});

document.getElementById("anomalies-only").addEventListener("change", () => {
  if (lastOverview && currentMode !== "single") {
    renderTable(lastOverview.symbols, document.getElementById("anomalies-only").checked);
  }
});
document.getElementById("auto-refresh").addEventListener("change", scheduleAutoRefresh);
document.getElementById("detail-close").addEventListener("click", hideDetail);
document.getElementById("detail-modal").addEventListener("click", (e) => {
  if (e.target.id === "detail-modal") hideDetail();
});
document.getElementById("analyze-btn").addEventListener("click", analyzeSymbol);

async function loadHealth() {
  const el = document.getElementById("health-banner");
  try {
    const resp = await fetch("/api/health");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (data.status === "stale") {
      const minutes = data.staleness_seconds != null ? Math.round(data.staleness_seconds / 60) : "?";
      el.className = "health-banner neg";
      el.textContent = `⚠️ Scanner izgleda zaglavljen — nema novih podataka ${minutes} min (zadnji scan: ${data.last_scanned_at || "nikad"}). Provjeri servis na VPS-u.`;
    } else if (data.status === "no_data") {
      el.className = "health-banner neg";
      el.textContent = "⚠️ Scanner još nije upisao nijedan rezultat.";
    } else {
      el.className = "health-banner hidden";
      el.textContent = "";
    }
  } catch (err) {
    // A failed health check itself is not alarming enough to show a banner
    // (could just be a blip reaching the dashboard's own API) — stay quiet.
    console.error(err);
  }
}

populateSymbolSuggestions();
setFullTableMode(document.getElementById("show-all-toggle").checked);
loadBestSignal();
setInterval(loadBestSignal, REFRESH_MS);
loadForwardTracking();
setInterval(loadForwardTracking, REFRESH_MS);
loadHealth();
setInterval(loadHealth, 60000); // check more often than the data refresh itself — staleness matters sooner
