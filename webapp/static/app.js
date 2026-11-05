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

// Prices on this dashboard range from BTC (tens of thousands) down to
// sub-cent altcoins (e.g. 0.02465) — a fixed decimal count either wastes
// space on expensive coins or silently truncates the only digits that
// matter on cheap ones (0.0248 hides exactly the 0.02483 vs 0.02489
// difference a trader needs to see). Scale precision to roughly 5
// significant figures instead of a fixed decimal count.
function fmtPrice(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (!isFinite(num)) return "—";
  if (num === 0) return "0";
  const magnitude = Math.floor(Math.log10(Math.abs(num)));
  const digits = Math.max(2, Math.min(10, 4 - magnitude));
  return num.toFixed(digits);
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
// Every timeframe (5m/15m/1h/4h) is ranked and calibrated independently, so
// this renders as one tab per timeframe rather than one blended list.

const BEST_SIGNAL_TIMEFRAMES = ["5m", "15m", "1h", "4h"];
const BEST_SIGNAL_TIMEFRAME_LABELS = {
  "5m": "5m · Scalp",
  "15m": "15m · Short-term",
  "1h": "1h · Intraday",
  "4h": "4h · Swing",
};

// Timeframes managed with TP1/TP2/TP3 instead of one single target — must
// mirror scanner.signal_tracking.MULTI_TARGET_TIMEFRAMES (any timeframe of
// 1h or longer).
const MULTI_TARGET_TIMEFRAMES = new Set(["1h", "4h"]);
let activeBestSignalTimeframe = "1h";

function winRateRangeHtml(low, high) {
  if (low === null || low === undefined || high === null || high === undefined) return "";
  return `[${fmt(low, 1)}–${fmt(high, 1)}%]`;
}

function bestSignalEntryHtml(sig) {
  if (!sig.entry) return "";
  const kelly = sig.kelly_sizing;
  const e = sig.entry;
  // 4h candidates are managed as multi-target trades (TP1/TP2/TP3, stop to
  // breakeven after TP1) — show all three so the plan matches how it will
  // actually be tracked, instead of just the single TP a 1h/15m trade gets.
  const isMultiTarget = MULTI_TARGET_TIMEFRAMES.has(sig.timeframe) && e.take_profit_extended !== undefined && e.take_profit_extended !== null;
  const tpHtml = isMultiTarget
    ? `<span>TP1: ${fmtPrice(e.take_profit_conservative)} (40%) · TP2: ${fmtPrice(e.take_profit)} (35%) · TP3: ${fmtPrice(e.take_profit_extended)} (25%)</span>`
    : `<span>TP: ${fmtPrice(e.take_profit)}${e.take_profit_conservative !== undefined && e.take_profit_conservative !== null ? ` <span class="muted">(konzervativan: ${fmtPrice(e.take_profit_conservative)})</span>` : ""}</span>`;
  return `
    <div class="best-signal-entry">
      <span>Entry zona: <b>${fmtPrice(e.entry_zone_low)}–${fmtPrice(e.entry_zone_high)}</b></span>
      <span>SL: ${fmtPrice(e.stop_loss)}</span>
      ${tpHtml}
      <span>R:R ${fmt(e.risk_reward_ratio, 2)}</span>
      ${kelly ? `<span>💰 Kelly: <b>${fmt(kelly.recommended_fraction_pct, 1)}%</b> kapitala</span>` : ""}
    </div>
    ${isMultiTarget ? `<div class="best-signal-note">🎯 Nakon TP1 stop ide na breakeven, nakon TP2 na cijenu TP1 — trejd se broji pozitivnim i ako se kasnije zatvori na pomjerenom stopu.</div>` : ""}
    ${sig.entry.extension_warning ? `<div class="best-signal-warning">⏳ ${sig.entry.extension_warning}</div>` : ""}
    ${sig.entry.volatility_warning ? `<div class="best-signal-warning">⚠️ ${sig.entry.volatility_warning}</div>` : ""}
    ${kelly && kelly.warning ? `<div class="best-signal-warning">💰 ${kelly.warning}</div>` : ""}`;
}

const PATTERN_LABELS = { trending: "📈 Trending", breakout: "🚀 Breakout", reversal: "↩️ Reversal" };
const PATTERN_CLASSES = { trending: "pattern-trending", breakout: "pattern-breakout", reversal: "pattern-reversal" };

function patternBadgeHtml(pattern) {
  if (!pattern) return "";
  const text = PATTERN_LABELS[pattern] || pattern;
  const cls = PATTERN_CLASSES[pattern] || "";
  return `<span class="pattern-badge ${cls}">${text}</span>`;
}

function qualityScoreBadgeHtml(qs) {
  if (!qs || qs.total === undefined || qs.total === null) return "";
  const cls = qs.total >= 70 ? "pos" : qs.total >= 50 ? "" : "neg";
  const tooltip =
    `Trend: ${fmt(qs.trend_strength, 0)} · Momentum: ${fmt(qs.momentum, 0)} · Volumen: ${fmt(qs.volume_confirmation, 0)} · ` +
    `MTF slaganje: ${fmt(qs.multi_timeframe_alignment, 0)} · Volatilnost: ${fmt(qs.volatility_conditions, 0)} · ` +
    `Historija: ${fmt(qs.historical_performance, 0)}`;
  return `<span class="quality-score-badge ${cls}" title="${tooltip}">⚡ Kvalitet: <b>${fmt(qs.total, 0)}</b>/100</span>`;
}

function regimeBadgeHtml(sig) {
  if (!sig.current_market_regime) return "";
  const label = { low: "mirno", normal: "normalno", high: "volatilno" }[sig.current_market_regime] || sig.current_market_regime;
  const matched = sig.regime_fallback === false;
  const title = matched
    ? `Statistika uparena s trenutnim volatility režimom (${label}) -- win rate odgovara uslovima kakvi su TRENUTNO, ne cjelokupnom prosjeku.`
    : `Nema dovoljno historije baš za "${label}" režim -- koristi se blend svih režima kao fallback.`;
  return `<span class="muted regime-badge" title="${title}">📊 ${label}${matched ? "" : " (blend)"}</span>`;
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
        ${patternBadgeHtml(sig.pattern)}
        ${qualityScoreBadgeHtml(sig.quality_score)}
      </div>
      <div class="best-signal-stats">
        <span>Win rate: <b>${fmt(sig.win_rate_pct, 1)}%</b> ${winRateRangeHtml(sig.win_rate_ci_low_pct, sig.win_rate_ci_high_pct)}</span>
        <span class="muted">n=${sig.n}</span>
        <span class="${signClass(sig.avg_net_return_pct)}">avg net: ${fmt(sig.avg_net_return_pct, 2)}%</span>
        ${regimeBadgeHtml(sig)}
      </div>
      <div class="muted best-signal-label">${sig.label}</div>
      ${sig.correlation ? `<div class="best-signal-warning">🔗 ${sig.correlation.note}</div>` : ""}
      ${bestSignalEntryHtml(sig)}
    </div>`;
}

function bestSignalTabsHtml(signalsByTimeframe) {
  return BEST_SIGNAL_TIMEFRAMES.map((tf) => {
    const count = (signalsByTimeframe[tf] || []).length;
    const activeClass = tf === activeBestSignalTimeframe ? " best-signal-tab-active" : "";
    return `<button type="button" class="best-signal-tab${activeClass}" data-tf="${tf}">${BEST_SIGNAL_TIMEFRAME_LABELS[tf]} (${count})</button>`;
  }).join("");
}

function renderBestSignalPanel(data) {
  const el = document.getElementById("best-signal-panel");
  const byTimeframe = (data && data.signals_by_timeframe) || {};
  const totalSignals = Object.values(byTimeframe).reduce((sum, list) => sum + list.length, 0);
  if (!data || !totalSignals) {
    el.className = "best-signal-panel muted";
    el.innerHTML = data && data.candidates_considered
      ? `Nema trenutno statistički potvrđenog signala ni na jednom timeframe-u (pregledano ${data.candidates_considered} ` +
        `kombinacija coin+timeframe sa trendom, keš kalibracije spreman za ${data.cache_coverage || "0/0"}). Keš se puni ` +
        "u pozadini — pokušajte opet za koji minut."
      : "Nema podataka još — sačekajte da skener i keš kalibracije završe prvi ciklus.";
    return;
  }
  el.className = "best-signal-panel";
  // Never land the user on a tab that's currently empty if another one has signals.
  if (!(byTimeframe[activeBestSignalTimeframe] || []).length) {
    activeBestSignalTimeframe = BEST_SIGNAL_TIMEFRAMES.find((tf) => (byTimeframe[tf] || []).length) || activeBestSignalTimeframe;
  }
  const signals = byTimeframe[activeBestSignalTimeframe] || [];
  const [top, ...rest] = signals;
  const bodyHtml = signals.length
    ? bestSignalCardHtml(top, 0) +
      (rest.length ? `<div class="best-signal-runners">${rest.map((s, i) => bestSignalCardHtml(s, i + 1)).join("")}</div>` : "")
    : `<div class="muted best-signal-empty-tf">Nema statistički potvrđenog signala na ${activeBestSignalTimeframe} timeframe-u trenutno.</div>`;

  el.innerHTML =
    `<div class="best-signal-title">🎯 Top signal sada — statistički potvrđen (n≥30, donja granica 95% CI &gt; 50%), odvojeno po timeframe-u</div>` +
    `<div class="best-signal-tabs">${bestSignalTabsHtml(byTimeframe)}</div>` +
    bodyHtml;

  el.querySelectorAll(".best-signal-tab").forEach((tabEl) => {
    tabEl.addEventListener("click", () => {
      activeBestSignalTimeframe = tabEl.dataset.tf;
      renderBestSignalPanel(data);
    });
  });
  el.querySelectorAll(".best-signal-card").forEach((cardEl) => {
    cardEl.addEventListener("click", () => {
      document.getElementById("symbol-filter-input").value = cardEl.dataset.symbol;
      lookupSymbol(cardEl.dataset.symbol);
    });
  });
}

async function loadBestSignal() {
  try {
    const res = await fetch("/api/scan/best-signal?limit=5");
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

function renderForwardTrackingPanel(data, driftStatus) {
  const el = document.getElementById("forward-tracking-panel");
  if (!data) {
    el.className = "forward-tracking-panel muted";
    el.textContent = "Nema podataka o praćenju uživo još.";
    return;
  }
  const mutedNote = driftNoteHtml(driftStatus);
  if (data.resolved_count === 0) {
    el.className = "forward-tracking-panel muted";
    el.innerHTML =
      `📡 Praćenje uživo: ${data.open_count} signala u toku, još nema zatvorenih — rezultati stižu kako se SL/TP/timeout dešavaju.` +
      mutedNote;
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
    `<span>Zatvoreno: <b>${data.resolved_count}</b> (${statusBreakdownText(byStatus)})</span>` +
    `<span>U toku: <b>${data.open_count}</b></span>` +
    `<span>Stvarni win rate: <b>${fmt(realized, 1)}%</b></span>` +
    `<span>Obećana donja granica (prosjek): <b>${fmt(promised, 1)}%</b></span>` +
    (gap !== null ? `<span class="${gapClass}">razlika: ${gap >= 0 ? "+" : ""}${fmt(gap, 1)}pp</span>` : "") +
    `</div>` +
    mutedNote;
}

// A bucket the live calibration-drift detector has muted (see
// scanner/drift_detector.py) is simply absent from Top Signal Now — this
// note exists so that absence is explained instead of silently invisible.
function driftNoteHtml(driftStatus) {
  if (!driftStatus || !driftStatus.muted_buckets || !driftStatus.muted_buckets.length) return "";
  const items = driftStatus.muted_buckets
    .map((b) => `${b.timeframe}/${b.direction.toUpperCase()} (${fmt(b.realized_win_rate_pct, 1)}% vs. obećanih ${fmt(b.promised_win_rate_ci_low_pct_avg, 1)}%)`)
    .join(", ");
  return `<div class="forward-tracking-warning">⏸️ Privremeno isključeno iz preporuka (loš live rezultat): ${items}</div>`;
}

async function loadForwardTracking() {
  try {
    const [res, driftRes] = await Promise.all([fetch("/api/forward-tracking/summary"), fetch("/api/drift-status")]);
    const data = await res.json();
    const driftStatus = await driftRes.json();
    renderForwardTrackingPanel(data, driftStatus);
  } catch (err) {
    console.error("Could not load forward tracking summary:", err);
  }
}

// --- Tab switching (Scanner vs. Analitika trejdova) -----------------------

let currentTradesStatusFilter = "";
let currentTradesRange = "";
let currentTradesSince = null;
let currentTradesUntil = null;

function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  document.getElementById("tab-scanner").classList.toggle("active", tab === "scanner");
  document.getElementById("tab-trades").classList.toggle("active", tab === "trades");
  document.getElementById("tab-backtest").classList.toggle("active", tab === "backtest");
  document.getElementById("tab-research").classList.toggle("active", tab === "research");
  if (tab === "trades") {
    loadTradesTab();
  } else if (tab === "backtest") {
    loadBacktestTab();
  } else if (tab === "research") {
    loadResearchTab();
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// --- Analitika trejdova: every tracked signal, in full detail -------------
// Distinct from the compact forward-tracking-panel summary strip: this tab
// shows every individual tracked trade (open and resolved) with its exact
// entry/SL/TP/exit and promised-vs-realized numbers, not just aggregate
// counts, so a user can see exactly *what* was traded, not just a tally.

const STATUS_LABELS = {
  open: "U toku",
  take_profit: "TP",
  stop_loss: "SL",
  timeout: "Timeout",
  tp1_be: "TP1→BE",
  tp2_be: "TP2→BE",
  full_run: "Full run",
};

function statusBadgeHtml(status) {
  const label = STATUS_LABELS[status] || status;
  return `<span class="status-badge status-${status}">${label}</span>`;
}

// Small "2/3" progress badge shown next to the status badge for still-open
// multi-target (4h) trades that have already banked at least one target --
// lets a user see at a glance that a trade is trailing on a moved stop
// rather than sitting at its original risk.
function multiTargetProgressHtml(t) {
  if (!t.multi_target || t.status !== "open" || !t.targets_hit) return "";
  return ` <span class="muted">(${t.targets_hit}/3 targets)</span>`;
}

// Shared "(TP x · TP1→BE y · SL z · timeout w)" breakdown string, used by
// every summary/bucket card. Builds from whatever non-open statuses are
// present in a fixed, readable order instead of hardcoding each status in
// every call site — so adding a new status (e.g. a future TP4) only means
// adding it to STATUS_LABELS and this order list, not every render spot.
const STATUS_BREAKDOWN_ORDER = ["take_profit", "full_run", "tp2_be", "tp1_be", "stop_loss", "timeout"];

function statusBreakdownText(byStatus) {
  return STATUS_BREAKDOWN_ORDER.filter((s) => (byStatus[s] || 0) > 0)
    .map((s) => `${STATUS_LABELS[s]} ${byStatus[s]}`)
    .join(" · ");
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// Trade-table rows come straight from the signal_outcomes SQLite row shape
// (plain `pattern` string, plain `quality_score` number, plain `regime` +
// `regime_fallback` columns) -- distinct from the best-signal candidate
// shape the Top Signal Now badges above are built for (nested
// `quality_score.total`, `current_market_regime`), so these small adapters
// translate a trade row into the same badge markup instead of duplicating it.
function tradeQualityBadgeHtml(qualityScore) {
  if (qualityScore === null || qualityScore === undefined) return "—";
  const cls = qualityScore >= 70 ? "pos" : qualityScore >= 50 ? "" : "neg";
  return `<span class="quality-score-badge ${cls}">${fmt(qualityScore, 0)}</span>`;
}

const REGIME_LABELS = { low: "mirno", normal: "normalno", high: "volatilno" };

function tradeRegimeBadgeHtml(regime, regimeFallback) {
  if (!regime) return "—";
  const label = REGIME_LABELS[regime] || regime;
  const matched = regimeFallback === 0 || regimeFallback === false;
  return `<span class="muted regime-badge">📊 ${label}${matched ? "" : " (blend)"}</span>`;
}


function renderTradesSummary(summary, trades) {
  const el = document.getElementById("trades-summary");
  if (!summary) {
    el.className = "trades-summary muted";
    el.textContent = "Nema podataka o praćenju trejdova još.";
    return;
  }
  const byStatus = summary.resolved_by_status || {};
  const resolvedTrades = trades.filter((t) => t.status !== "open" && t.realized_return_pct !== null && t.realized_return_pct !== undefined);
  const avgNetReturn = resolvedTrades.length
    ? resolvedTrades.reduce((sum, t) => sum + t.realized_return_pct, 0) / resolvedTrades.length
    : null;
  const gap =
    summary.realized_win_rate_pct !== null && summary.avg_promised_win_rate_ci_low_pct !== null
      ? summary.realized_win_rate_pct - summary.avg_promised_win_rate_ci_low_pct
      : null;

  el.className = "trades-summary";
  el.innerHTML = `
    <div class="trades-summary-cards">
      <div class="trades-summary-card"><span class="label">Ukupno trejdova</span><span class="value">${summary.open_count + summary.resolved_count}</span></div>
      <div class="trades-summary-card"><span class="label">U toku</span><span class="value">${summary.open_count}</span></div>
      <div class="trades-summary-card"><span class="label">Take profit</span><span class="value pos">${byStatus.take_profit || 0}</span></div>
      <div class="trades-summary-card"><span class="label">Full run (TP1+TP2+TP3)</span><span class="value pos">${byStatus.full_run || 0}</span></div>
      <div class="trades-summary-card"><span class="label">TP2 → BE</span><span class="value pos">${byStatus.tp2_be || 0}</span></div>
      <div class="trades-summary-card"><span class="label">TP1 → BE</span><span class="value pos">${byStatus.tp1_be || 0}</span></div>
      <div class="trades-summary-card"><span class="label">Stop loss</span><span class="value neg">${byStatus.stop_loss || 0}</span></div>
      <div class="trades-summary-card"><span class="label">Timeout</span><span class="value">${byStatus.timeout || 0}</span></div>
      <div class="trades-summary-card"><span class="label">Stvarni win rate</span><span class="value">${fmt(summary.realized_win_rate_pct, 1)}%</span></div>
      <div class="trades-summary-card"><span class="label">Prosječan net return</span><span class="value ${signClass(avgNetReturn)}">${avgNetReturn !== null ? fmt(avgNetReturn, 2) + "%" : "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Obećano vs. stvarno</span><span class="value ${gap !== null ? (gap >= 0 ? "pos" : "neg") : ""}">${gap !== null ? (gap >= 0 ? "+" : "") + fmt(gap, 1) + "pp" : "—"}</span></div>
    </div>`;
}

function renderTradesTable(trades) {
  const body = document.getElementById("trades-body");
  if (!trades.length) {
    body.innerHTML = `<tr><td colspan="15" class="muted">Nema trejdova za ovaj filter.</td></tr>`;
    return;
  }
  body.innerHTML = trades
    .map((t) => {
      const dirClass = t.direction === "long" ? "trend-bullish" : "trend-bearish";
      const dirWord = t.direction === "long" ? "LONG" : "SHORT";
      return `
    <tr>
      <td data-label="Symbol"><b>${t.symbol}</b></td>
      <td data-label="TF">${t.timeframe}</td>
      <td data-label="Smjer" class="${dirClass}">${dirWord}</td>
      <td data-label="Pattern">${patternBadgeHtml(t.pattern) || "—"}</td>
      <td data-label="Kvalitet">${tradeQualityBadgeHtml(t.quality_score)}</td>
      <td data-label="Regime">${tradeRegimeBadgeHtml(t.regime, t.regime_fallback)}</td>
      <td data-label="Status">${statusBadgeHtml(t.status)}${multiTargetProgressHtml(t)}</td>
      <td data-label="Entry">${fmtPrice(t.entry_price)}</td>
      <td data-label="SL">${fmtPrice(t.stop_loss)}</td>
      <td data-label="TP">${fmtPrice(t.take_profit)}</td>
      <td data-label="Exit">${t.exit_price !== null && t.exit_price !== undefined ? fmtPrice(t.exit_price) : "—"}</td>
      <td data-label="Net %" class="${signClass(t.realized_return_pct)}">${t.realized_return_pct !== null && t.realized_return_pct !== undefined ? fmt(t.realized_return_pct, 2) + "%" : "—"}</td>
      <td data-label="Obećano (CI low)">${fmt(t.promised_win_rate_ci_low_pct, 1)}%</td>
      <td data-label="Otvoren">${fmtDateTime(t.logged_at)}</td>
      <td data-label="Zatvoren">${fmtDateTime(t.resolved_at)}</td>
    </tr>`;
    })
    .join("");
}

function renderBucketBreakdown(buckets) {
  const el = document.getElementById("bucket-breakdown");
  if (!buckets.length) {
    el.className = "bucket-breakdown muted";
    el.innerHTML = "Nema zatvorenih trejdova za raspodjelu po bucket-ima.";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = buckets
    .map((b) => {
      const dirWord = b.direction === "long" ? "LONG" : "SHORT";
      const dirClass = b.direction === "long" ? "pos" : "neg";
      const gapClass = b.gap_pp === null ? "" : b.gap_pp >= 0 ? "pos" : "neg";
      const netClass = signClass(b.avg_net_return_pct);
      return `
    <tr>
      <td><b>${b.timeframe}</b></td>
      <td class="${dirClass}">${dirWord}</td>
      <td>${b.resolved_count} <span class="muted">(${statusBreakdownText(b.by_status)})</span></td>
      <td>${fmt(b.win_rate_pct, 1)}%</td>
      <td class="${netClass}">${b.avg_net_return_pct !== null ? fmt(b.avg_net_return_pct, 2) + "%" : "—"}</td>
      <td>${fmt(b.avg_promised_win_rate_ci_low_pct, 1)}%</td>
      <td class="${gapClass}">${b.gap_pp !== null ? (b.gap_pp >= 0 ? "+" : "") + fmt(b.gap_pp, 1) + "pp" : "—"}</td>
    </tr>`;
    })
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">Po bucket-u (timeframe + smjer) — ko zapravo vuče prosjek</div>
    <table class="bucket-breakdown-table">
      <thead>
        <tr><th>TF</th><th>Smjer</th><th>N (zatvoreno)</th><th>Win rate</th><th>Avg net %</th><th>Obećano (CI low)</th><th>Razlika</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// Shared row renderer for the pattern/regime/quality breakdowns below --
// they all compute the exact same win-rate/avg-net/gap statistics
// server-side (see webapp/main.py::_bucket_stats), just grouped by a
// different key, so the row markup only differs in its leading cell(s).
function _breakdownRowHtml(leadingCells, stats) {
  const gapClass = stats.gap_pp === null ? "" : stats.gap_pp >= 0 ? "pos" : "neg";
  const netClass = signClass(stats.avg_net_return_pct);
  const byStatus = stats.by_status || {};
  return `
    <tr>
      ${leadingCells}
      <td>${stats.resolved_count} <span class="muted">(${statusBreakdownText(byStatus)})</span></td>
      <td>${fmt(stats.win_rate_pct, 1)}%</td>
      <td class="${netClass}">${stats.avg_net_return_pct !== null ? fmt(stats.avg_net_return_pct, 2) + "%" : "—"}</td>
      <td>${fmt(stats.avg_promised_win_rate_ci_low_pct, 1)}%</td>
      <td class="${gapClass}">${stats.gap_pp !== null ? (stats.gap_pp >= 0 ? "+" : "") + fmt(stats.gap_pp, 1) + "pp" : "—"}</td>
    </tr>`;
}

function renderPatternBreakdown(patterns) {
  const el = document.getElementById("pattern-breakdown");
  if (!patterns.length) {
    el.className = "bucket-breakdown muted";
    el.innerHTML = "Nema zatvorenih trejdova za raspodjelu po pattern-u.";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = patterns.map((p) => _breakdownRowHtml(`<td>${patternBadgeHtml(p.pattern) || p.pattern}</td>`, p)).join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">Po pattern-u — da li BREAKOUT/TRENDING/REVERSAL stvarno radi bolje?</div>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Pattern</th><th>N (zatvoreno)</th><th>Win rate</th><th>Avg net %</th><th>Obećano (CI low)</th><th>Razlika</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

const REGIME_MATCH_LABELS = { "regime-matched": "uparen", blend: "blend", unknown: "—" };

function renderRegimeBreakdown(regimes) {
  const el = document.getElementById("regime-breakdown");
  if (!regimes.length) {
    el.className = "bucket-breakdown muted";
    el.innerHTML = "Nema zatvorenih trejdova za raspodjelu po regime-u.";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = regimes
    .map((r) => {
      const label = REGIME_LABELS[r.regime] || r.regime;
      const matchLabel = REGIME_MATCH_LABELS[r.match_status] || r.match_status;
      return _breakdownRowHtml(`<td>📊 ${label}</td><td>${matchLabel}</td>`, r);
    })
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">Po volatility regime-u — da li regime-matched kalibracija stvarno pomaže?</div>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Regime</th><th>Status</th><th>N (zatvoreno)</th><th>Win rate</th><th>Avg net %</th><th>Obećano (CI low)</th><th>Razlika</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderQualityBreakdown(buckets) {
  const el = document.getElementById("quality-breakdown");
  if (!buckets.length) {
    el.className = "bucket-breakdown muted";
    el.innerHTML = "Nema zatvorenih trejdova za raspodjelu po Quality Score-u.";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = buckets.map((q) => _breakdownRowHtml(`<td>${q.quality_bucket}</td>`, q)).join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">Po Signal Quality Score-u — da li viši skor stvarno znači bolji win rate?</div>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Quality bucket</th><th>N (zatvoreno)</th><th>Win rate</th><th>Avg net %</th><th>Obećano (CI low)</th><th>Razlika</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// --- Backtest tab: cached retrospective portfolio-simulation report -------
// Reads the pre-generated report written by
// `python -m scanner.backtesting.portfolio_backtest --report-json ...`
// (see webapp/main.py's /api/backtest/report). Deliberately a one-shot load
// on tab switch, not a polled auto-refresh like the live tabs -- the report
// is a static snapshot that only changes when someone reruns the backtest
// by hand, so polling it on the scanner's 5-minute cadence would be noise.

// Which strategy variant (see scanner/backtesting/strategy_variants.py) the
// tab currently shows -- defaults to "original" and is only ever changed by
// the user via the "Strategija" dropdown below. Kept as module state (not
// re-read from the DOM every call) so every fetch in this file -- report,
// refresh, refresh/status -- stays in lockstep with whichever variant the
// dropdown is showing right now, without threading a parameter through
// every call site.
let currentBacktestVariant = "original";
let backtestVariantsLoaded = false;

// Populates the "Strategija" dropdown from /api/backtest/variants exactly
// once (its options are static for the life of the page -- the registry in
// strategy_variants.py doesn't change without a code deploy). Re-running
// this after the user has picked a variant would reset their selection, so
// loadBacktestTab() only calls it the first time.
async function loadBacktestVariantsDropdown() {
  const select = document.getElementById("backtest-variant-select");
  if (!select) return;
  try {
    const res = await fetch("/api/backtest/variants");
    const data = await res.json();
    currentBacktestVariant = data.default_variant_id || "original";
    select.innerHTML = (data.variants || [])
      .map((v) => {
        const reportFlag = v.has_report ? "" : " (nema izvještaja)";
        return `<option value="${v.variant_id}">${v.label}${reportFlag}</option>`;
      })
      .join("");
    select.value = currentBacktestVariant;
    backtestVariantsLoaded = true;
  } catch (err) {
    console.error("Could not load backtest variants:", err);
  }
}

// --- CPU-throttle slider ("koliko % servera smije koristiti ovaj run") --
// Persisted server-side (scanner_data/backtest_cpu_limit.json via
// GET/POST /api/backtest/cpu-limit, see webapp/main.py) so it survives
// page reloads AND is reused by the next refresh regardless of what
// triggers it -- the manual button here, or any future automated trigger,
// since both ultimately go through the same POST /api/backtest/refresh
// handler that reads this exact same setting. Loaded once per page load
// (like the variants dropdown) so a user mid-drag on the slider doesn't
// get overwritten by a routine tab reload.
let backtestCpuLimitLoaded = false;

function renderBacktestCpuLimitValue(pct) {
  const valueEl = document.getElementById("backtest-cpu-limit-value");
  if (valueEl) valueEl.textContent = `${pct}%`;
}

async function loadBacktestCpuLimit() {
  const rangeEl = document.getElementById("backtest-cpu-limit-range");
  if (!rangeEl) return;
  try {
    const res = await fetch("/api/backtest/cpu-limit");
    const data = await res.json();
    const pct = Number(data.cpu_limit_pct) || 30;
    rangeEl.value = String(pct);
    renderBacktestCpuLimitValue(pct);
    backtestCpuLimitLoaded = true;
  } catch (err) {
    console.error("Could not load backtest CPU limit:", err);
  }
}

async function saveBacktestCpuLimit(pct) {
  try {
    await fetch(`/api/backtest/cpu-limit?pct=${encodeURIComponent(pct)}`, { method: "POST" });
  } catch (err) {
    console.error("Could not save backtest CPU limit:", err);
  }
}

async function loadBacktestTab() {
  if (!backtestVariantsLoaded) {
    await loadBacktestVariantsDropdown();
  }
  if (!backtestCpuLimitLoaded) {
    await loadBacktestCpuLimit();
  }
  try {
    const res = await fetch(`/api/backtest/report?variant=${encodeURIComponent(currentBacktestVariant)}`);
    const report = await res.json();
    if (!report.available) {
      renderBacktestUnavailable();
    } else {
      renderBacktestMeta(report);
      renderBacktestFullConfig(report.config);
      renderBacktestSummary(report.metrics);
      renderBacktestEquityCurve(report.equity_curve, report.config ? report.config.initial_capital : 10000);
      const breakdown = report.breakdown || {};
      renderBacktestBreakdownTable("backtest-breakdown-timeframe", "Po timeframe-u", breakdown.by_timeframe, "TF");
      renderBacktestBreakdownTable("backtest-breakdown-direction", "Po smjeru", breakdown.by_direction, "Smjer");
      renderBacktestBreakdownTable("backtest-breakdown-label", "Po label-u (setup tipu)", breakdown.by_label, "Label");
      renderStatisticalValidation(report.statistical_validation);
      renderOosValidation(report.oos_validation);
      renderIndependenceValidation(report.independence_validation);
      renderWalkForwardValidation(report.walk_forward_validation);
    }
  } catch (err) {
    console.error("Could not load backtest report:", err);
    document.getElementById("backtest-meta").textContent = `Greška: ${err.message}`;
  }
  await loadExperimentHistory();
  // Only reflects current state in the button/status text -- does NOT
  // itself re-trigger a reload on completion, since this same function is
  // what a completed refresh calls to re-render (see
  // startBacktestRefreshPolling()). Re-triggering here would recurse.
  await renderBacktestRefreshStatus();
}

// --- Manual "🔄 Osvježi cijeli test" button -------------------------------
// Fires POST /api/backtest/refresh (detached background job on the server,
// see webapp/main.py), then polls GET /api/backtest/refresh/status every
// 15s until it leaves the "running" state, at which point it reloads this
// whole tab once. A plain page load / tab switch also calls
// renderBacktestRefreshStatus() (via loadBacktestTab() above) so the
// button correctly shows "already running" if a refresh triggered from
// another tab/session (or from the VPS shell directly) is still going.

let backtestRefreshPollTimer = null;

function applyBacktestRefreshStatus(status) {
  const btn = document.getElementById("backtest-refresh-btn");
  const statusEl = document.getElementById("backtest-refresh-status");
  if (!btn || !statusEl) return;
  if (status.state === "running") {
    btn.disabled = true;
    statusEl.textContent = "⏳ Test u toku (obično 30-60+ min s funding cost + point-in-time fetch-om)…";
  } else if (status.state === "done") {
    btn.disabled = false;
    statusEl.textContent = "✅ Posljednji test završen uspješno.";
  } else if (status.state === "error") {
    btn.disabled = false;
    statusEl.textContent = "⚠️ Posljednji test je završio bez validnog izvještaja — provjeri log na serveru.";
  } else {
    btn.disabled = false;
    statusEl.textContent = "";
  }
}

async function renderBacktestRefreshStatus() {
  try {
    const res = await fetch(`/api/backtest/refresh/status?variant=${encodeURIComponent(currentBacktestVariant)}`);
    applyBacktestRefreshStatus(await res.json());
  } catch (err) {
    console.error("Could not read backtest refresh status:", err);
  }
}

// `variant` is captured at the moment the refresh was triggered (not
// re-read from `currentBacktestVariant` on every tick) so that switching
// the dropdown to a different strategy while a refresh is still polling
// doesn't make this poll start asking about -- and eventually reload the
// tab for -- the wrong variant's status.
async function pollBacktestRefreshUntilDone(variant) {
  try {
    const res = await fetch(`/api/backtest/refresh/status?variant=${encodeURIComponent(variant)}`);
    const status = await res.json();
    if (variant === currentBacktestVariant) {
      applyBacktestRefreshStatus(status);
    }
    if (status.state === "running") return;
    if (backtestRefreshPollTimer) {
      clearInterval(backtestRefreshPollTimer);
      backtestRefreshPollTimer = null;
    }
    if (status.state === "done" && variant === currentBacktestVariant) {
      await loadBacktestTab();
    }
  } catch (err) {
    console.error("Could not poll backtest refresh status:", err);
  }
}

async function triggerBacktestRefresh() {
  const btn = document.getElementById("backtest-refresh-btn");
  const statusEl = document.getElementById("backtest-refresh-status");
  const variant = currentBacktestVariant;
  btn.disabled = true;
  statusEl.textContent = "Pokrećem test na serveru…";
  try {
    const res = await fetch(`/api/backtest/refresh?variant=${encodeURIComponent(variant)}`, {
      method: "POST",
    });
    const data = await res.json();
    if (!data.started && data.reason === "already_running") {
      statusEl.textContent = "Test je već u toku (pokrenut ranije).";
    }
    if (!backtestRefreshPollTimer) {
      backtestRefreshPollTimer = setInterval(() => pollBacktestRefreshUntilDone(variant), 15000);
    }
    await pollBacktestRefreshUntilDone(variant);
  } catch (err) {
    statusEl.textContent = `Greška: ${err.message}`;
    btn.disabled = false;
  }
}

// --- Phase 8 Korak 3: Deflated Sharpe Ratio + experiment registry --------
// These two sections are a statistical check ON the backtest report above,
// not a separate dataset -- they belong in this tab because they answer
// "how much should I actually trust the numbers above, given how many
// times this (or a similar) strategy has been tried?" rather than
// presenting a new, independent metric.

function renderStatisticalValidation(dsr) {
  const el = document.getElementById("backtest-statistical-validation");
  if (!dsr) {
    el.className = "trades-summary muted";
    el.innerHTML =
      "Statistička validacija (Deflated Sharpe Ratio) nije dostupna za ovaj izvještaj " +
      "(premalo trejdova ili izvještaj je generisan prije dodavanja ove funkcije).";
    return;
  }
  el.className = "trades-summary";
  const skillBadge = dsr.likely_skill
    ? '<span class="dsr-badge dsr-badge-pos">vjerovatno pravi rezultat</span>'
    : '<span class="dsr-badge dsr-badge-neg">vjerovatno slučajnost/overfitting</span>';
  const bb = dsr.block_bootstrap_mean_return;
  const bbHtml = bb
    ? `
    <div class="bucket-breakdown-title" style="margin-top:14px;">
      🎲 Block bootstrap CI za prosječan povrat po trejdu
      ${
        bb.edge_survives_lower_bound
          ? '<span class="dsr-badge dsr-badge-pos">edge preživi i donju granicu</span>'
          : '<span class="dsr-badge dsr-badge-neg">donja granica ≤ 0%</span>'
      }
    </div>
    <p class="dsr-explainer muted">
      Umjesto da svaki trejd tretira kao nezavisan dokaz (kao prosti bootstrap), ovo preuzorkuje
      <b>cjeline uzastopnih trejdova</b> (blok veličine ${bb.block_size}) da sačuva kratkoročnu korelaciju
      između susjednih trejdova (npr. isti tržišni režim). ${bb.confidence_pct}% interval povjerenja za prosječan
      povrat po trejdu je <b>[${fmt(bb.ci_low_pct, 3)}%, ${fmt(bb.ci_high_pct, 3)}%]</b> preko ${bb.n_resamples}
      preuzorkovanja — ako donja granica nije iznad 0%, edge nije pouzdano odvojen od šuma čak ni uz ovu konzervativniju provjeru.
    </p>
    <div class="trades-summary-cards">
      <div class="trades-summary-card"><span class="label">Observed prosjek</span><span class="value ${signClass(bb.observed_pct)}">${fmt(bb.observed_pct, 3)}%</span></div>
      <div class="trades-summary-card"><span class="label">CI donja granica</span><span class="value ${signClass(bb.ci_low_pct)}">${fmt(bb.ci_low_pct, 3)}%</span></div>
      <div class="trades-summary-card"><span class="label">CI gornja granica</span><span class="value ${signClass(bb.ci_high_pct)}">${fmt(bb.ci_high_pct, 3)}%</span></div>
      <div class="trades-summary-card"><span class="label">Blok veličina</span><span class="value">${bb.block_size}</span></div>
    </div>`
    : `
    <div class="bucket-breakdown-title" style="margin-top:14px;">🎲 Block bootstrap CI</div>
    <p class="muted">Nije dostupno (isključeno preko --no-block-bootstrap, ili premalo trejdova — potrebno je najmanje 4).</p>`;
  el.innerHTML = `
    <div class="bucket-breakdown-title">
      📐 Statistička validacija (Deflated Sharpe Ratio) ${skillBadge}
    </div>
    <p class="dsr-explainer muted">
      Observed Sharpe je obični Sharpe ratio ovog izvještaja. Deflated benchmark je koliko visok Sharpe
      bismo <b>očekivali da vidimo čisto slučajno</b> nakon <b>${dsr.n_trials}</b> pokušaja (ovaj + prethodni, iz
      eksperimentalnog registra ispod). Probabilistic Sharpe Ratio (PSR) je procjena vjerovatnoće da je stvarni
      Sharpe ove strategije iznad tog benchmarka — PSR blizu 100% znači da rezultat vjerovatno nije slučajnost;
      blizu 0% znači da i dalje može biti.
    </p>
    <div class="trades-summary-cards">
      <div class="trades-summary-card"><span class="label">Observed Sharpe</span><span class="value ${signClass(dsr.observed_sharpe)}">${fmt(dsr.observed_sharpe, 3)}</span></div>
      <div class="trades-summary-card"><span class="label">Deflated benchmark</span><span class="value">${fmt(dsr.deflated_sharpe_benchmark, 3)}</span></div>
      <div class="trades-summary-card"><span class="label">Probabilistic Sharpe Ratio</span><span class="value">${fmt(dsr.probabilistic_sharpe_ratio * 100, 1)}%</span></div>
      <div class="trades-summary-card"><span class="label">Broj pokušaja (n_trials)</span><span class="value">${dsr.n_trials}</span></div>
      <div class="trades-summary-card"><span class="label">Broj trejdova</span><span class="value">${dsr.n_observations}</span></div>
      <div class="trades-summary-card"><span class="label">Skewness</span><span class="value">${fmt(dsr.skewness, 3)}</span></div>
      <div class="trades-summary-card"><span class="label">Kurtosis</span><span class="value">${fmt(dsr.kurtosis, 3)}</span></div>
    </div>
    ${bbHtml}`;
}

// Phase 8 Korak 6 (audit follow-up): per (label, horizon) check of whether
// the win rate measured on the most-recent held-out test window still
// clears 50% -- a label can look great pooled over the whole history and
// still be a pure in-sample artifact. This is a check ON individual labels'
// edges, same spirit as the DSR check above being a check on the portfolio
// Sharpe as a whole.
function renderOosValidation(oos) {
  const el = document.getElementById("backtest-oos-validation");
  if (!oos || !oos.length) {
    el.className = "bucket-breakdown muted";
    el.textContent =
      "Nema in/out-of-sample provjere u ovom izvještaju (isključena preko --no-oos-validation, ili nema dovoljno slučajeva).";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = oos
    .map((r) => {
      const badge = r.holds_up
        ? '<span class="pos">drži se</span>'
        : `<span class="neg">${r.warning ? "ne drži se" : "slabije na test prozoru"}</span>`;
      return `
    <tr>
      <td><b>${r.label}</b></td>
      <td>${r.direction ?? "—"}</td>
      <td>${r.horizon_candles}</td>
      <td>${r.train_n} (${r.train_win_rate_pct != null ? fmt(r.train_win_rate_pct, 1) + "%" : "—"})</td>
      <td>${r.test_n} (${r.test_win_rate_pct != null ? fmt(r.test_win_rate_pct, 1) + "%" : "—"}, CI low ${r.test_win_rate_ci_low_pct != null ? fmt(r.test_win_rate_ci_low_pct, 1) + "%" : "—"})</td>
      <td>${badge}</td>
      <td class="muted">${r.warning ?? ""}</td>
    </tr>`;
    })
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">
      🔁 In/Out-of-sample provjera (po label-u) — da li se edge drži i na najnovijim, do sada neviđenim slučajevima
    </div>
    <p class="dsr-explainer muted">
      "Train" je sve osim posljednjeg test prozora; "Test" je samo taj najnoviji, nikad-korišten dio. Ako je
      train win rate dobar a test loš, taj label je vjerovatno in-sample artefakt, ne pravi edge.
    </p>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Label</th><th>Smjer</th><th>Horizon</th><th>Train N (win%)</th><th>Test N (win%, CI low)</th><th>Status</th><th>Napomena</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// Phase 8 Korak 6: per (label, horizon) comparison of the raw win-rate CI
// (today's number, one row per historical case) against one recomputed
// after collapsing time-overlapping cases -- any symbol -- into single
// independent events first. A label can look confirmed on raw N and still
// be a handful of correlated market moves counted many times over; this is
// the check for that specific failure mode, distinct from (and upstream
// of) the in/out-of-sample check above.
function renderIndependenceValidation(independence) {
  const el = document.getElementById("backtest-independence-validation");
  if (!independence || !independence.length) {
    el.className = "bucket-breakdown muted";
    el.textContent =
      "Nema provjere nezavisnosti uzorka u ovom izvještaju (isključena preko --no-independence-validation, " +
      "ili nema dovoljno slučajeva, ili je izvještaj generisan prije dodavanja ove funkcije).";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = independence
    .map((r) => {
      const badge = r.holds_up_independent
        ? '<span class="pos">drži se</span>'
        : `<span class="neg">${r.warning ? "ne drži se" : "slabije nakon korekcije"}</span>`;
      return `
    <tr>
      <td><b>${r.label}</b></td>
      <td>${r.direction ?? "—"}</td>
      <td>${r.horizon_candles}</td>
      <td>${r.n_raw} (${fmt(r.win_rate_raw_pct, 1)}%, CI low ${r.win_rate_raw_ci_low_pct != null ? fmt(r.win_rate_raw_ci_low_pct, 1) + "%" : "—"})</td>
      <td>${r.n_independent_events} (${r.win_rate_independent_pct != null ? fmt(r.win_rate_independent_pct, 1) + "%" : "—"}, CI low ${r.win_rate_independent_ci_low_pct != null ? fmt(r.win_rate_independent_ci_low_pct, 1) + "%" : "—"})</td>
      <td>${badge}</td>
      <td class="muted">${r.warning ?? ""}</td>
    </tr>`;
    })
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">
      🧩 Provjera nezavisnosti uzorka (po label-u) — da li su preklapajući trejdovi (bilo kog simbola) naduvali broj "dokaza"
    </div>
    <p class="dsr-explainer muted">
      Jedan tržišni pokret (npr. BTC pumpa) može upaliti isti label na više povezanih altcoin-a gotovo istovremeno —
      to je JEDAN događaj, ne nekoliko nezavisnih potvrda. "Sirovo N" broji svaki slučaj posebno; "Nezavisni događaji"
      spaja vremenski preklapajuće slučajeve (bilo kog simbola) u jedan prije nego što se ponovo računa 95% CI. Ako
      je "drži se" kolona "ne drži se" ovdje a gore (OOS) je bila "drži se", edge je vjerovatno samo par korelisanih
      pokreta koji su se brojali više puta, ne širok, stvaran uzorak.
    </p>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Label</th><th>Smjer</th><th>Horizon</th><th>Sirovo N (win%, CI low)</th><th>Nezavisni događaji (win%, CI low)</th><th>Status</th><th>Napomena</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// Phase 8 Korak 7 Test 2: walk-forward verdict per (label, horizon) --
// expands the single train/test split above (renderOosValidation) into
// several chronological folds, requiring EVERY evaluated fold to confirm
// the edge, not just one lucky (or unlucky) split. Deliberately operates on
// raw case counts per fold, same as the OOS check -- not independence-
// corrected (see scanner/validation/walk_forward.py's module docstring),
// so it is complementary to, not a replacement for, the independence check
// above.
function renderWalkForwardValidation(walkForward) {
  const el = document.getElementById("backtest-walk-forward-validation");
  if (!walkForward || !walkForward.length) {
    el.className = "bucket-breakdown muted";
    el.textContent =
      "Nema walk-forward provjere u ovom izvještaju (isključena preko --no-walk-forward-validation, " +
      "ili nema dovoljno slučajeva/historije za foldove, ili je izvještaj generisan prije dodavanja ove funkcije).";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = walkForward
    .map((r) => {
      const badge = r.holds_up_walk_forward
        ? '<span class="pos">drži se u svim foldovima</span>'
        : `<span class="neg">${r.warning ? "ne drži se konzistentno" : "slabije kroz vrijeme"}</span>`;
      const foldsSummary = (r.folds || [])
        .map((f) => {
          const foldBadge = !f.evaluated ? "—" : f.holds_up ? "✅" : "❌";
          const winPct = f.test_win_rate_pct != null ? fmt(f.test_win_rate_pct, 1) + "%" : "—";
          return `#${f.fold_index}: n=${f.test_n}, ${winPct} ${foldBadge}`;
        })
        .join(" &middot; ");
      return `
    <tr>
      <td><b>${r.label}</b></td>
      <td>${r.direction ?? "—"}</td>
      <td>${r.horizon_candles}</td>
      <td>${r.n_folds_holding_up}/${r.n_folds_evaluated} (od ${r.n_folds_total} ukupno)</td>
      <td class="muted">${foldsSummary}</td>
      <td>${badge}</td>
      <td class="muted">${r.warning ?? ""}</td>
    </tr>`;
    })
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">
      📆 Walk-forward provjera (po label-u) — da li se edge drži konzistentno kroz VIŠE uzastopnih vremenskih perioda
    </div>
    <p class="dsr-explainer muted">
      Historija je podijeljena u nekoliko uzastopnih foldova (expanding/anchored train prozor, svaki sljedeći fold
      trenira na svemu prije njega). Edge "drži se" samo ako SVAKI ocijenjeni fold pojedinačno potvrdi 95% CI donju
      granicu iznad 50% — jedan dobar fold ne znači ništa ako su ostali loši. Ovo je stroža, višeperiodna verzija
      gornje in/out-of-sample provjere, ali i dalje koristi sirovo N po foldu (bez korekcije nezavisnosti uzorka).
    </p>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Label</th><th>Smjer</th><th>Horizon</th><th>Foldova drži se</th><th>Po foldu (N, win%, status)</th><th>Status</th><th>Napomena</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// Deliberately a separate fetch from loadBacktestTab's try/catch -- the
// experiment registry can be populated (and interesting to see) even when
// no dashboard report has been generated yet, and vice versa, so one
// section being unavailable should never hide the other.
async function loadExperimentHistory() {
  try {
    const res = await fetch("/api/backtest/experiments?limit=50");
    const data = await res.json();
    renderExperimentHistory(data.available ? data.experiments : null);
  } catch (err) {
    console.error("Could not load experiment history:", err);
    renderExperimentHistory(null);
  }
}

function renderExperimentHistory(experiments) {
  const el = document.getElementById("backtest-experiments");
  if (!experiments || !experiments.length) {
    el.className = "bucket-breakdown muted";
    el.textContent = "Još nema zabilježenih eksperimenata u registru.";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = experiments
    .map(
      (e) => `
    <tr>
      <td>${fmtDateTime(e.recorded_at)}</td>
      <td><b>${e.strategy_family}</b>${e.strategy_version ? ` <span class="muted">(${e.strategy_version})</span>` : ""}</td>
      <td>${e.timeframe ?? "—"}</td>
      <td>${e.n_trades ?? "—"}</td>
      <td class="${signClass(e.sharpe_ratio)}">${e.sharpe_ratio !== null && e.sharpe_ratio !== undefined ? fmt(e.sharpe_ratio, 3) : "—"}</td>
      <td class="${signClass(e.net_profit_pct)}">${e.net_profit_pct !== null && e.net_profit_pct !== undefined ? fmt(e.net_profit_pct, 2) + "%" : "—"}</td>
      <td class="muted">${e.notes ?? ""}</td>
    </tr>`
    )
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">📋 Historija eksperimenata (svaki pokušaj, ne samo najbolji)</div>
    <table class="bucket-breakdown-table">
      <thead><tr><th>Vrijeme</th><th>Familija</th><th>TF</th><th>N</th><th>Sharpe</th><th>Net profit</th><th>Napomene</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderBacktestUnavailable() {
  document.getElementById("backtest-meta").className = "backtest-meta muted";
  document.getElementById("backtest-meta").innerHTML =
    "Još nema generisanog backtest izvještaja. Pokrenite ga ručno na serveru:" +
    '<pre style="white-space: pre-wrap;">python -m scanner.backtesting.portfolio_backtest --symbols ... --report-json scanner_data/portfolio_backtest_report.json</pre>';
  document.getElementById("backtest-summary").className = "trades-summary muted";
  document.getElementById("backtest-summary").textContent = "";
  document.getElementById("backtest-equity-curve").className = "backtest-equity-curve muted";
  document.getElementById("backtest-equity-curve").textContent = "";
  ["backtest-breakdown-timeframe", "backtest-breakdown-direction", "backtest-breakdown-label"].forEach((id) => {
    document.getElementById(id).className = "bucket-breakdown muted";
    document.getElementById(id).textContent = "";
  });
  document.getElementById("backtest-statistical-validation").className = "trades-summary muted";
  document.getElementById("backtest-statistical-validation").textContent = "";
  document.getElementById("backtest-full-config").className = "bucket-breakdown muted";
  document.getElementById("backtest-full-config").textContent = "";
  document.getElementById("backtest-oos-validation").className = "bucket-breakdown muted";
  document.getElementById("backtest-oos-validation").textContent = "";
  document.getElementById("backtest-independence-validation").className = "bucket-breakdown muted";
  document.getElementById("backtest-independence-validation").textContent = "";
  document.getElementById("backtest-walk-forward-validation").className = "bucket-breakdown muted";
  document.getElementById("backtest-walk-forward-validation").textContent = "";
}

function renderBacktestMeta(report) {
  const el = document.getElementById("backtest-meta");
  const cfg = report.config || {};
  el.className = "backtest-meta";
  let html = `Generisano: <b>${fmtDateTime(report.generated_at)}</b> &middot; ${(cfg.symbols || []).length} simbola &middot; TF: ${(cfg.timeframes || []).join(", ") || "—"} &middot; warmup: ${cfg.warmup_candles ?? "—"} svijeća &middot; trošak/trejd: ${cfg.round_trip_cost_pct ?? "—"}%`;
  if (report.skipped_symbols && report.skipped_symbols.length) {
    html += ` &middot; <span class="neg">preskočeno (vendor greška): ${report.skipped_symbols.join(", ")}</span>`;
  }
  const pit = report.point_in_time_universe;
  if (pit && pit.enabled) {
    html += ` &middot; <span class="pos">point-in-time univerzum: aktivan</span> (isključeno ${pit.trades_excluded ?? 0} trejdova koji ne bi bili u top-N na dan ulaska)`;
  } else {
    html += ` &middot; <span class="muted">point-in-time univerzum: nije korišten (simboli birani unazad na osnovu današnje likvidnosti — survivorship bias nije ispravljen)</span>`;
  }
  el.innerHTML = html;
}

// Full execution-assumption transparency: every config value that actually
// affects the simulated numbers above, not just the headline few (symbols/
// timeframes/warmup/cost) shown in the compact meta line. Anything added to
// `build_dashboard_report()`'s config dict in portfolio_backtest.py should
// also get a row here, or it's silently unauditable from the dashboard.
function renderBacktestFullConfig(cfg) {
  const el = document.getElementById("backtest-full-config");
  if (!cfg) {
    el.className = "bucket-breakdown muted";
    el.textContent = "";
    return;
  }
  el.className = "bucket-breakdown";
  const rows = [
    ["Početni kapital", `$${fmt(cfg.initial_capital, 0)}`],
    ["Rizik po trejdu", `${cfg.risk_per_trade_pct ?? "—"}%`],
    ["Max konkurentnih pozicija", cfg.max_concurrent_positions ?? "—"],
    ["Max isti smjer odjednom", cfg.max_concurrent_per_direction ?? "—"],
    ["Broj svijeća po simbolu (kline_limit)", cfg.kline_limit ?? "—"],
    ["Dodatni slippage", `${cfg.extra_slippage_pct ?? 0}%`],
    [
      "Per-simbol cost tier-ovi",
      cfg.use_symbol_cost_tiers
        ? '<span class="pos">uključeno</span> (tier-1 majori po base trošku, ostali s nadoplatom)'
        : '<span class="muted">isključeno</span> (svaki simbol plaća isti flat trošak)',
    ],
    [
      "Funding cost (perpetual swap)",
      cfg.include_funding_cost
        ? '<span class="pos">uključen</span> (stvarna historija funding rate-a naplaćena po trejdu)'
        : '<span class="muted">isključen</span> (trošak finansiranja pozicije nije uračunat)',
    ],
    [
      "SL/TP tie-break",
      cfg.sl_tp_tie_break === "optimistic"
        ? '<span class="neg">optimistic</span> (pretpostavlja da je TP pogođen prvi kad su oba u istoj svijeći — gornja granica)'
        : '<span class="pos">conservative</span> (pretpostavlja da je SL pogođen prvi kad su oba u istoj svijeći — nikad ne precjenjuje edge)',
    ],
    [
      "In/out-of-sample provjera",
      cfg.include_oos_validation
        ? `<span class="pos">uključena</span> (test prozor: posljednjih ${Math.round((cfg.oos_test_fraction ?? 0) * 100)}% slučajeva, min. ${cfg.oos_min_test_n ?? "—"} za pouzdanost)`
        : '<span class="muted">isključena</span>',
    ],
    [
      "Provjera nezavisnosti uzorka",
      cfg.include_independence_validation
        ? `<span class="pos">uključena</span> (preklapajući trejdovi bilo kog simbola broje se kao jedan događaj; min. ${cfg.independence_min_n ?? "—"} nezavisnih događaja za pouzdanost)`
        : '<span class="muted">isključena</span>',
    ],
    [
      "Walk-forward provjera",
      cfg.include_walk_forward_validation
        ? `<span class="pos">uključena</span> (${cfg.walk_forward_n_folds ?? "—"} uzastopnih foldova, min. ${cfg.walk_forward_min_test_n ?? "—"} po foldu, min. ${cfg.walk_forward_min_folds_evaluated ?? "—"} ocijenjenih foldova za verdikt)`
        : '<span class="muted">isključena</span>',
    ],
    [
      "Block bootstrap CI (prosječan povrat)",
      cfg.include_block_bootstrap
        ? `<span class="pos">uključen</span> (${cfg.block_bootstrap_n_resamples ?? "—"} preuzorkovanja, ${cfg.block_bootstrap_confidence_pct ?? "—"}% interval povjerenja)`
        : '<span class="muted">isključen</span>',
    ],
  ];
  const tableRows = rows.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">⚙️ Puna konfiguracija ovog backtest-a (sve pretpostavke koje utiču na brojeve iznad)</div>
    <table class="bucket-breakdown-table">
      <tbody>${tableRows}</tbody>
    </table>`;
}

function renderBacktestSummary(metrics) {
  const el = document.getElementById("backtest-summary");
  if (!metrics || !metrics.n_trades) {
    el.className = "trades-summary muted";
    el.textContent = (metrics && metrics.note) || "Nema trejdova u ovom backtest izvještaju.";
    return;
  }
  el.className = "trades-summary";
  el.innerHTML = `
    <div class="trades-summary-cards">
      <div class="trades-summary-card"><span class="label">Broj trejdova</span><span class="value">${metrics.n_trades}</span></div>
      <div class="trades-summary-card"><span class="label">Nezavisne epizode</span><span class="value">${metrics.n_independent_episodes ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Win rate</span><span class="value">${fmt(metrics.win_rate_pct, 1)}%</span></div>
      <div class="trades-summary-card"><span class="label">Net profit</span><span class="value ${signClass(metrics.net_profit_pct)}">${fmt(metrics.net_profit_pct, 2)}%</span></div>
      <div class="trades-summary-card"><span class="label">Profit factor</span><span class="value">${metrics.profit_factor ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Expectancy (R)</span><span class="value ${signClass(metrics.expectancy_r_multiple)}">${metrics.expectancy_r_multiple !== null && metrics.expectancy_r_multiple !== undefined ? fmt(metrics.expectancy_r_multiple, 3) : "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Max drawdown</span><span class="value neg">${fmt(metrics.max_drawdown_pct, 2)}%</span></div>
      <div class="trades-summary-card"><span class="label">Sharpe</span><span class="value ${signClass(metrics.sharpe_ratio)}">${metrics.sharpe_ratio ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Sortino</span><span class="value ${signClass(metrics.sortino_ratio)}">${metrics.sortino_ratio ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">CAGR</span><span class="value ${signClass(metrics.cagr_pct)}">${metrics.cagr_pct !== null && metrics.cagr_pct !== undefined ? fmt(metrics.cagr_pct, 2) + "%" : "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Trejdova / godina</span><span class="value">${metrics.trades_per_year ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Prosj. konkurentnih pozicija</span><span class="value">${metrics.avg_concurrent_positions ?? "—"}</span></div>
      <div class="trades-summary-card"><span class="label">Ukupni troškovi (pp)</span><span class="value neg">${fmt(metrics.total_cost_pct_points, 2)}</span></div>
    </div>`;
}

function renderBacktestBreakdownTable(elId, title, rows, keyLabel) {
  const el = document.getElementById(elId);
  if (!rows || !rows.length) {
    el.className = "bucket-breakdown muted";
    el.innerHTML = `Nema podataka za "${title}".`;
    return;
  }
  el.className = "bucket-breakdown";
  const body = rows
    .map(
      (r) => `
    <tr>
      <td><b>${r.key}</b></td>
      <td>${r.n}</td>
      <td>${fmt(r.win_rate_pct, 1)}%</td>
      <td class="${signClass(r.avg_r_multiple)}">${r.avg_r_multiple !== null && r.avg_r_multiple !== undefined ? fmt(r.avg_r_multiple, 3) : "—"}</td>
      <td class="${signClass(r.net_profit)}">${fmt(r.net_profit, 2)}</td>
    </tr>`
    )
    .join("");
  el.innerHTML = `
    <div class="bucket-breakdown-title">${title}</div>
    <table class="bucket-breakdown-table">
      <thead><tr><th>${keyLabel}</th><th>N</th><th>Win rate</th><th>Avg R</th><th>Net profit</th></tr></thead>
      <tbody>${body}</tbody>
    </table>`;
}

// A minimal hand-rolled inline-SVG sparkline -- the frontend has no
// charting library (no Chart.js/D3), and one data series on a tab that's
// rarely open doesn't justify adding a new dependency.
function renderBacktestEquityCurve(curve, initialCapital) {
  const el = document.getElementById("backtest-equity-curve");
  if (!curve || curve.length < 2) {
    el.className = "backtest-equity-curve muted";
    el.textContent = "Nedovoljno tačaka za equity krivu.";
    return;
  }
  el.className = "backtest-equity-curve";
  const width = 600;
  const height = 140;
  const padX = 4;
  const padY = 8;
  const values = curve.map((p) => p.equity);
  const baseline = initialCapital ?? values[0];
  const min = Math.min(...values, baseline);
  const max = Math.max(...values, baseline);
  const range = max - min || 1;
  const stepX = (width - 2 * padX) / (curve.length - 1);
  const toY = (equity) => height - padY - ((equity - min) / range) * (height - 2 * padY);
  const points = curve.map((p, i) => `${(padX + i * stepX).toFixed(1)},${toY(p.equity).toFixed(1)}`).join(" ");
  const last = values[values.length - 1];
  const strokeVar = last >= baseline ? "var(--green)" : "var(--red)";
  el.innerHTML = `
    <div class="bucket-breakdown-title">Equity kriva (${curve.length} tačaka)</div>
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <line x1="${padX}" y1="${toY(baseline).toFixed(1)}" x2="${width - padX}" y2="${toY(baseline).toFixed(1)}"
            style="stroke: var(--border); stroke-dasharray: 4 3; stroke-width: 1" />
      <polyline points="${points}" fill="none" style="stroke: ${strokeVar}; stroke-width: 2" />
    </svg>`;
}

// Builds the shared range/since/until query params every forward-tracking
// endpoint accepts -- an explicit custom since/until always wins over a
// preset range token (see webapp/main.py::_resolve_time_range).
function tradesRangeQueryParams() {
  const params = new URLSearchParams();
  if (currentTradesSince || currentTradesUntil) {
    if (currentTradesSince) params.set("since", currentTradesSince);
    if (currentTradesUntil) params.set("until", currentTradesUntil);
  } else if (currentTradesRange) {
    params.set("range", currentTradesRange);
  }
  return params;
}

// Monotonically increasing token guarding against slow, overlapping loads
// (the VPS can take a while under load) -- without this, an older in-flight
// request (e.g. the unfiltered initial load) can resolve *after* a newer one
// (e.g. clicking the "SL" filter) and silently clobber the table back to the
// stale, unfiltered result even though the filter button shows "active".
let _tradesLoadToken = 0;

async function loadTradesTab(status) {
  if (status !== undefined) currentTradesStatusFilter = status;
  const token = ++_tradesLoadToken;
  try {
    const rangeParams = tradesRangeQueryParams();
    const tradesParams = new URLSearchParams(rangeParams);
    if (currentTradesStatusFilter) tradesParams.set("status", currentTradesStatusFilter);

    const [summaryRes, tradesRes, bucketRes, patternRes, regimeRes, qualityRes] = await Promise.all([
      fetch(`/api/forward-tracking/summary?${rangeParams}`),
      fetch(`/api/forward-tracking/trades?${tradesParams}`),
      fetch(`/api/forward-tracking/by-bucket?${rangeParams}`),
      fetch(`/api/forward-tracking/by-pattern?${rangeParams}`),
      fetch(`/api/forward-tracking/by-regime?${rangeParams}`),
      fetch(`/api/forward-tracking/by-quality?${rangeParams}`),
    ]);
    const summary = await summaryRes.json();
    const tradesData = await tradesRes.json();
    const bucketData = await bucketRes.json();
    const patternData = await patternRes.json();
    const regimeData = await regimeRes.json();
    const qualityData = await qualityRes.json();
    if (token !== _tradesLoadToken) return; // a newer load (e.g. a filter click) already started -- discard this stale one
    renderTradesSummary(summary, tradesData.trades || []);
    renderTradesTable(tradesData.trades || []);
    renderBucketBreakdown(bucketData.buckets || []);
    renderPatternBreakdown(patternData.patterns || []);
    renderRegimeBreakdown(regimeData.regimes || []);
    renderQualityBreakdown(qualityData.quality_buckets || []);
  } catch (err) {
    if (token !== _tradesLoadToken) return;
    console.error("Could not load trades analytics:", err);
    document.getElementById("trades-body").innerHTML = `<tr><td colspan="15" class="muted">Greška: ${err.message}</td></tr>`;
  }
}

document.querySelectorAll(".trades-filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".trades-filter-btn").forEach((b) => b.classList.toggle("active", b === btn));
    loadTradesTab(btn.dataset.status);
  });
});

// --- Analitika trejdova: time-range filter (presets + custom) -------------

document.querySelectorAll(".trades-range-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const customBar = document.getElementById("trades-custom-range");
    if (btn.dataset.range === "custom") {
      customBar.classList.toggle("hidden");
      return;
    }
    customBar.classList.add("hidden");
    document.querySelectorAll(".trades-range-btn").forEach((b) => b.classList.toggle("active", b === btn));
    currentTradesRange = btn.dataset.range;
    currentTradesSince = null;
    currentTradesUntil = null;
    loadTradesTab();
  });
});

document.getElementById("trades-custom-apply").addEventListener("click", () => {
  const sinceInput = document.getElementById("trades-custom-since").value;
  const untilInput = document.getElementById("trades-custom-until").value;
  if (!sinceInput && !untilInput) return;
  currentTradesSince = sinceInput ? new Date(sinceInput).toISOString() : null;
  currentTradesUntil = untilInput ? new Date(untilInput).toISOString() : null;
  currentTradesRange = "";
  document.querySelectorAll(".trades-range-btn").forEach((b) => b.classList.toggle("active", b.dataset.range === "custom"));
  loadTradesTab();
});

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
      <td data-label="Close">${fmtPrice(s.close)}</td>
      <td data-label="RSI">${fmt(s.rsi, 1)}</td>
      <td data-label="EMA20 / EMA50">${fmtPrice(s.ema20)} / ${fmtPrice(s.ema50)}</td>
      <td data-label="Vol Δ%" class="${signClass(s.volume_change_pct)}">${fmt(s.volume_change_pct, 1)}%</td>
      <td data-label="Funding">${fmt((s.funding_rate || 0) * 100, 4)}%</td>
      <td data-label="OI Δ60m%" class="${signClass(s.open_interest_change_pct_60m)}">${fmt(s.open_interest_change_pct_60m, 2)}%</td>
      <td data-label="Support / Resistance">${fmtPrice(s.support)} / ${fmtPrice(s.resistance)}</td>
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
    .map(([ratio, level]) => `<span>${ratio}: <b>${fmtPrice(level)}</b></span>`)
    .join(" &nbsp; ");
  const extension = Object.entries(fib.extension || {})
    .map(([ratio, level]) => `<span>${ratio}: <b>${fmtPrice(level)}</b></span>`)
    .join(" &nbsp; ");

  return `
    <div class="tf-card">
      <div class="tf-card-header">
        <b>${tf}</b>
        <span class="${trendClass(data.trend)}">${data.trend}</span>
        <span class="muted">close: ${fmtPrice(data.close)}</span>
      </div>
      <table class="tf-table">
        <tr><td>RSI</td><td>${fmt(data.rsi, 1)}</td><td>EMA cross</td><td>${data.ema_cross || "—"}</td></tr>
        <tr><td>EMA9 / 20 / 21 / 50 / 200</td><td colspan="3">${fmtPrice(data.ema9)} / ${fmtPrice(data.ema20)} / ${fmtPrice(data.ema21)} / ${fmtPrice(data.ema50)} / ${fmtPrice(data.ema200)}</td></tr>
        <tr><td>MACD / signal / hist</td><td colspan="3">${fmt(data.macd, 5)} / ${fmt(data.macd_signal, 5)} / ${fmt(data.macd_hist, 5)}</td></tr>
        <tr><td>ATR</td><td>${fmtPrice(data.atr)}</td><td>Bollinger lower/upper</td><td>${fmtPrice(data.bollinger_lower)} / ${fmtPrice(data.bollinger_upper)}</td></tr>
        <tr><td>Volume</td><td>${fmt(data.volume, 1)}</td><td>Rel. volume / Δ%</td><td>${fmt(data.relative_volume, 2)}x / ${fmtPct(data.volume_change_pct, 1)}</td></tr>
        <tr><td>Support / Resistance</td><td colspan="3">${fmtPrice(data.support)} / ${fmtPrice(data.resistance)}</td></tr>
        <tr><td>Drawdown / Momentum fading</td><td colspan="3">${fmtPct(data.drawdown_pct, 1)} / ${data.momentum_fading === null || data.momentum_fading === undefined ? "—" : data.momentum_fading}</td></tr>
      </table>
      <div class="anomaly-badges">${anomalies}</div>
      ${
        fib.swing_high
          ? `<details class="fib-toggle"><summary class="muted">Fibonacci (swing ${fib.swing_direction || "—"}: ${fmtPrice(fib.swing_low)} - ${fmtPrice(fib.swing_high)})</summary>
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
      <tr><td>Funding rate</td><td>${fmtPct((d.funding_rate || 0) * 100, 4)}</td><td>Mark / Index price</td><td>${fmtPrice(d.mark_price)} / ${fmtPrice(d.index_price)}</td></tr>
      <tr><td>Open interest</td><td>${fmt(d.open_interest, 1)}</td><td>OI Δ 5m / 15m / 60m</td><td>${fmtPct(d.open_interest_change_pct_5m, 2)} / ${fmtPct(d.open_interest_change_pct_15m, 2)} / ${fmtPct(d.open_interest_change_pct_60m, 2)}</td></tr>
    </table>`;
  const pivotHtml = Object.keys(p).length
    ? `<table class="tf-table">
        <tr><td>P</td><td>${fmtPrice(p.pivot)}</td><td>R1 / S1</td><td>${fmtPrice(p.r1)} / ${fmtPrice(p.s1)}</td></tr>
        <tr><td>R2 / S2</td><td>${fmtPrice(p.r2)} / ${fmtPrice(p.s2)}</td><td>R3 / S3</td><td>${fmtPrice(p.r3)} / ${fmtPrice(p.s3)}</td></tr>
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
        <span>${fmtPrice(b.price_low)}–${fmtPrice(b.price_high)}</span>
        <span class="heatmap-bar-track"><span class="heatmap-bar-fill" style="width:${width}%"></span></span>
        <span>${fmt(b.pct_of_total, 1)}%</span>
      </div>`;
    })
    .join("");

  const stopZones = (data.likely_stop_zones || [])
    .map((z) => `<li>${fmtPrice(z.price)} — ${z.reason}</li>`)
    .join("");

  const walls = data.orderbook_walls;
  const wallItem = (w) =>
    `<li class="${w.side === "bid" ? "pos" : "neg"}">${fmtPrice(w.price)} (${w.distance_pct > 0 ? "+" : ""}${fmt(w.distance_pct, 2)}%) — ${w.side === "bid" ? "bid" : "ask"} zid, ${fmt(w.notional_usdt / 1000, 1)}k USDT</li>`;
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
      Current: <b>${fmtPrice(current)}</b> · POC (najprometnija zona): <b>${fmtPrice(data.poc_price)}</b> ·
      Value area: ${fmtPrice(data.value_area_low)} – ${fmtPrice(data.value_area_high)}
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
          ? ` <span class="muted">(konzervativan: ${fmtPrice(lv.take_profit_conservative)})</span>`
          : "";
      const patientEntry =
        lv.patient_entry !== undefined && lv.patient_entry !== null
          ? ` <span class="muted">(strpljiv: ${fmtPrice(lv.patient_entry)})</span>`
          : "";
      return `<tr>
        <td><b>${tf}</b></td>
        <td class="${biasCls}">${lv.bias === "long" ? "LONG" : "SHORT"}</td>
        <td>${fmtPrice(lv.entry)}${patientEntry}</td>
        <td class="neg">${fmtPrice(lv.stop_loss)} <span class="muted">(${fmt(lv.risk_pct, 2)}%)</span></td>
        <td class="pos">${fmtPrice(lv.take_profit)} <span class="muted">(${fmt(lv.reward_pct, 2)}%)</span>${conservativeTp}</td>
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
      <td data-label="Close">${fmtPrice(s.close)}</td>
      <td data-label="RSI">${fmt(s.rsi, 1)}</td>
      <td data-label="EMA20 / EMA50">${fmtPrice(s.ema20)} / ${fmtPrice(s.ema50)}</td>
      <td data-label="Vol Δ%" class="${signClass(s.volume_change_pct)}">${fmt(s.volume_change_pct, 1)}%</td>
      <td data-label="Funding">${fmt((s.funding_rate || 0) * 100, 4)}%</td>
      <td data-label="OI Δ60m%" class="${signClass(s.open_interest_change_pct_60m)}">${fmt(s.open_interest_change_pct_60m, 2)}%</td>
      <td data-label="Support / Resistance">${fmtPrice(s.support)} / ${fmtPrice(s.resistance)}</td>
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
document.getElementById("backtest-refresh-btn").addEventListener("click", triggerBacktestRefresh);
document.getElementById("backtest-variant-select").addEventListener("change", (e) => {
  currentBacktestVariant = e.target.value;
  loadBacktestTab();
});
document.getElementById("backtest-cpu-limit-range").addEventListener("input", (e) => {
  renderBacktestCpuLimitValue(e.target.value);
});
document.getElementById("backtest-cpu-limit-range").addEventListener("change", (e) => {
  saveBacktestCpuLimit(e.target.value);
});

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
setInterval(() => {
  if (document.getElementById("tab-trades").classList.contains("active")) loadTradesTab();
}, REFRESH_MS);

// --- Istraživanje: read-only summary of the Phase 8 research registry -----
// Distinct from the live "🧪 Backtest" tab: this renders the curated,
// human-maintained scanner_data/research_registry.json — a report card of
// every closed/ongoing pre-registered hypothesis-testing family. One-time
// fetch on first click (mirrors how loadBacktestTab() is only triggered by
// switchTab()); the backing file only changes when a human closes a new
// hypothesis, so there's no auto-refresh and no refresh button here.

let researchLoaded = false;

async function loadResearchTab() {
  if (researchLoaded) return;
  researchLoaded = true;
  try {
    const res = await fetch("/api/research/registry");
    const data = await res.json();
    if (!data.available) {
      renderResearchUnavailable();
    } else {
      renderResearchHoldoutBanner(data.shared_holdout);
      renderResearchFamilies(data.families || []);
      renderResearchOverallNote(data.overall_note);
    }
  } catch (err) {
    console.error("Could not load research registry:", err);
    researchLoaded = false; // allow a retry on the next tab click
    document.getElementById("research-holdout-banner").textContent = `Greška: ${err.message}`;
  }
}

function renderResearchUnavailable() {
  document.getElementById("research-holdout-banner").textContent = "Istraživački registar još nije dostupan.";
  document.getElementById("research-holdout-banner").className = "backtest-meta muted";
  document.getElementById("research-families").innerHTML = "";
  document.getElementById("research-overall-note").textContent = "";
  document.getElementById("research-overall-note").className = "trades-summary muted";
}

function renderResearchHoldoutBanner(sharedHoldout) {
  const el = document.getElementById("research-holdout-banner");
  if (!sharedHoldout) {
    el.textContent = "";
    el.className = "backtest-meta muted";
    return;
  }
  el.className = "backtest-meta";
  const endIso = sharedHoldout.end_timestamp_iso || "—";
  const note = sharedHoldout.note || "";
  el.innerHTML = `📅 Zajednička holdout granica: <b>${endIso}</b>${note ? ` — ${note}` : ""}`;
}

function renderResearchOverallNote(note) {
  const el = document.getElementById("research-overall-note");
  if (!note) {
    el.textContent = "";
    el.className = "trades-summary muted";
    return;
  }
  el.className = "trades-summary";
  el.innerHTML = `<b>Zaključak:</b> ${note}`;
}

// "no_edge"/"no_edge_decisive" render as a neutral/red "closed, no edge"
// badge; anything else (not yet seen in today's data, but the schema
// allows e.g. a future "edge_confirmed") falls back to a green "edge
// potvrđen" style so new decisions don't need a code change to render
// sensibly instead of falling into the closed-red bucket by default.
const RESEARCH_DECISION_BADGES = {
  no_edge: { label: "Zatvoreno — nema edge-a", cls: "research-badge-noedge" },
  no_edge_decisive: { label: "Zatvoreno — odlučno nema edge-a", cls: "research-badge-noedge" },
};

function researchDecisionBadgeHtml(decision) {
  const known = RESEARCH_DECISION_BADGES[decision];
  if (known) return `<span class="research-badge ${known.cls}">${known.label}</span>`;
  if (!decision) return `<span class="research-badge research-badge-unknown">U toku</span>`;
  return `<span class="research-badge research-badge-confirmed">${decision}</span>`;
}

function researchCiRangeHtml(low, high, survives) {
  if (low === null || low === undefined || high === null || high === undefined) return "—";
  const indicator = survives === true ? " ✅" : survives === false ? " ❌" : "";
  return `[${fmt(low)}%, ${fmt(high)}%]${indicator}`;
}

// Renders a labeled stat only if the value is actually present — the
// variants array (family 1) deliberately has null fields per variant
// (e.g. a frozen_baseline with no n_independent), and printing
// "null"/"N/A" for every one of those would be noisier than just omitting
// the row, per the task's "skip null fields gracefully" requirement.
function researchStatRow(label, value) {
  if (value === null || value === undefined || value === "") return "";
  return `<tr><td class="muted">${label}</td><td>${value}</td></tr>`;
}

function renderResearchFlatFamilyStats(family) {
  const rows = [
    researchStatRow("Nezavisnih uzoraka (n)", family.n_independent),
    researchStatRow("Potreban n (pre-registrovano)", family.n_required),
    researchStatRow(
      "Neto povrat (observed / CI)",
      family.net_observed_pct !== null && family.net_observed_pct !== undefined
        ? `${fmt(family.net_observed_pct)}% &nbsp; ${researchCiRangeHtml(family.net_ci_low_pct, family.net_ci_high_pct, family.net_edge_survives_lower_bound)}`
        : null
    ),
    researchStatRow(
      "Bruto povrat (observed / CI)",
      family.gross_observed_pct !== null && family.gross_observed_pct !== undefined
        ? `${fmt(family.gross_observed_pct)}% &nbsp; ${researchCiRangeHtml(family.gross_ci_low_pct, family.gross_ci_high_pct, family.gross_edge_survives_lower_bound)}`
        : null
    ),
    researchStatRow("Mjerena sigma", family.measured_sigma_pct !== null && family.measured_sigma_pct !== undefined ? `${fmt(family.measured_sigma_pct)}%` : null),
    researchStatRow("Win rate", family.win_rate_pct !== null && family.win_rate_pct !== undefined ? `${fmt(family.win_rate_pct)}%` : null),
    researchStatRow("Prosječan dnevni turnover", family.avg_daily_turnover_pct !== null && family.avg_daily_turnover_pct !== undefined ? `${fmt(family.avg_daily_turnover_pct)}%` : null),
    researchStatRow("BTC beta korelacija", family.btc_beta_correlation !== null && family.btc_beta_correlation !== undefined ? fmt(family.btc_beta_correlation, 4) : null),
    researchStatRow("Nedecidirano?", family.undecidable === true ? "Da" : family.undecidable === false ? "Ne" : null),
  ].join("");
  if (!rows) return "";
  return `<table class="bucket-breakdown-table research-stats-table"><tbody>${rows}</tbody></table>`;
}

function renderResearchVariant(variant) {
  const rows = [
    researchStatRow("n (nezavisni)", variant.n_independent),
    researchStatRow(
      "Neto povrat (observed / CI)",
      variant.net_observed_pct !== null && variant.net_observed_pct !== undefined
        ? `${fmt(variant.net_observed_pct)}% ${variant.net_ci_low_pct == null && variant.net_ci_high_pct == null ? "" : researchCiRangeHtml(variant.net_ci_low_pct, variant.net_ci_high_pct, variant.edge_survives_lower_bound)}`
        : variant.net_ci_low_pct !== null && variant.net_ci_low_pct !== undefined
          ? researchCiRangeHtml(variant.net_ci_low_pct, variant.net_ci_high_pct, variant.edge_survives_lower_bound)
          : null
    ),
    researchStatRow("OOS pass count", variant.oos_pass_count),
    researchStatRow("Independence pass count", variant.independence_pass_count),
    researchStatRow("Walk-forward pass count", variant.walk_forward_pass_count),
  ].join("");
  return `
    <div class="research-variant-card">
      <div class="research-variant-name">${variant.name}</div>
      <table class="bucket-breakdown-table research-stats-table"><tbody>${rows}</tbody></table>
    </div>`;
}

function renderResearchFamilyCard(family) {
  const statsHtml = Array.isArray(family.variants)
    ? `<div class="research-variants-grid">${family.variants.map(renderResearchVariant).join("")}</div>`
    : renderResearchFlatFamilyStats(family);
  return `
    <section class="bucket-breakdown research-family-card">
      <div class="research-family-header">
        <span class="bucket-breakdown-title">#${family.id} ${family.name}</span>
        ${researchDecisionBadgeHtml(family.decision)}
      </div>
      <div class="muted research-family-doc">${family.doc || ""}</div>
      ${statsHtml}
      <p class="research-decision-text">${family.decision_text || ""}</p>
    </section>`;
}

function renderResearchFamilies(families) {
  const el = document.getElementById("research-families");
  if (!families.length) {
    el.innerHTML = `<p class="muted">Nema zabilježenih porodica u registru.</p>`;
    return;
  }
  el.innerHTML = families.map(renderResearchFamilyCard).join("");
}
