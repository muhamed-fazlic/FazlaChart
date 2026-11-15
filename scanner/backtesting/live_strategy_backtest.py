"""Backtest of the live main strategy (``scanner.live_strategy``) for the
dashboard's "🧪 Backtest" tab.

A basket strategy (weekly long top-5 / short bottom-5 by 14d momentum)
has no SL/TP trades, so it doesn't fit ``portfolio_backtest``. It runs
through the same ``cross_sectional_engine.run_backtest`` the Strategy Lab
used to rank it, with the live spec's parameters, and adds what the tab
needs on top: an equity curve, drawdown, period stats and which symbols
the basket picked most often.

Universe:

* With a ranked point-in-time universe cache (``--ranked-universe-cache``,
  the same file the Strategy Lab scheduler uses): ``all`` = every ranked
  symbol (ranks 1..N), ``top`` = ranks 1-57, point-in-time (no
  survivorship bias). This is what the Lab measured.
* Without one: today's top ``--top-n`` USDT perpetuals by 24h volume
  (``top`` = first 57 of them). That carries SURVIVORSHIP BIAS (coins that
  died or got delisted are missing), and the report says so.

Run: ``python -m scanner.backtesting.live_strategy_backtest --variant momentum_30d_all``
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scanner import live_strategy as ls
from scanner.backtesting import cross_sectional_engine as cse

INITIAL_CAPITAL = 10_000.0
DEFAULT_TOP_N = 170  # same width as the Lab's ranked universe (3 terciles of 57/56/57)


@dataclass(frozen=True)
class BasketBacktest:
    variant_id: str
    label: str
    description: str
    tercile: str  # "all" | "top" | "mid" | "bottom"


BASKET_BACKTESTS: dict[str, BasketBacktest] = {
    "momentum_30d_all": BasketBacktest(
        "momentum_30d_all",
        "🎯 Glavna strategija: momentum 14d, svi coini",
        "Live glavna strategija: svake sedmice LONG top 5 / SHORT bottom 5 po 14d momentumu, svi coini.",
        "all",
    ),
    "momentum_30d_top": BasketBacktest(
        "momentum_30d_top",
        "Momentum 14d, top tercil likvidnosti (Lab #1)",
        "Ista strategija, ali samo na 57 najlikvidnijih coina, tačno ono što je Strategy Lab rangirao #1.",
        "top",
    ),
    "momentum_30d_mid": BasketBacktest(
        "momentum_30d_mid",
        "Momentum 14d, srednji tercil likvidnosti",
        "Ista strategija na coinima rangiranim 58-113 po likvidnosti.",
        "mid",
    ),
    "momentum_30d_bottom": BasketBacktest(
        "momentum_30d_bottom",
        "Momentum 14d, donji tercil likvidnosti",
        "Ista strategija na najmanje likvidnim coinima (rang 114-170).",
        "bottom",
    ),
}
MAIN_BASKET_VARIANT_ID = "momentum_30d_all"


def report_path(base_dir: str, variant_id: str) -> str:
    return str(Path(base_dir) / f"basket_backtest_report__{variant_id}.json")


def _max_drawdown_pct(equity: list[float]) -> float | None:
    if not equity:
        return None
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, (value / peak - 1.0) * 100.0)
    return round(worst, 4)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Equity curve, metrics and symbol-pick frequency from an engine
    report. Pure function, unit-tested."""
    periods = report.get("rebalance_periods", [])
    nets = [p["net_return_pct"] for p in periods]
    equity_values = [INITIAL_CAPITAL]
    curve = []
    for p in periods:
        equity_values.append(equity_values[-1] * (1.0 + p["net_return_pct"] / 100.0))
        curve.append({"time": p["exit_time"], "equity": round(equity_values[-1], 2)})
    if periods:
        curve.insert(0, {"time": periods[0]["rebalance_time"], "equity": INITIAL_CAPITAL})

    btc = [p["btc_return_pct"] for p in periods if p.get("btc_return_pct") is not None]
    btc_compounded = None
    if btc:
        acc = 1.0
        for r in btc:
            acc *= 1.0 + r / 100.0
        btc_compounded = round((acc - 1.0) * 100.0, 4)

    weeks = len(periods)
    metrics = {
        "n_periods": weeks,
        "win_rate_pct": report.get("win_rate_pct"),
        "avg_net_pct": round(sum(nets) / weeks, 4) if nets else None,
        "median_net_pct": round(statistics.median(nets), 4) if nets else None,
        "avg_gross_pct": round(sum(p["gross_return_pct"] for p in periods) / weeks, 4) if periods else None,
        "avg_cost_pct": round(sum(p["cost_pct"] for p in periods) / weeks, 4) if periods else None,
        "avg_funding_pct": round(sum(p["funding_cost_pct"] for p in periods) / weeks, 4) if periods else None,
        "best_period_pct": round(max(nets), 4) if nets else None,
        "worst_period_pct": round(min(nets), 4) if nets else None,
        "compounded_net_pct": round((equity_values[-1] / INITIAL_CAPITAL - 1.0) * 100.0, 4) if nets else None,
        "max_drawdown_pct": _max_drawdown_pct(equity_values),
        "sharpe_per_period": (
            round(statistics.mean(nets) / statistics.stdev(nets), 4)
            if len(nets) >= 2 and statistics.stdev(nets) > 0
            else None
        ),
        "avg_turnover": report.get("avg_turnover"),
        "btc_compounded_pct": btc_compounded,
        "net_ci": report.get("net_block_bootstrap"),
        "beta_check": report.get("beta_check"),
        "undecidable": report.get("undecidable_threshold"),
    }

    picks: dict[str, Counter] = {"long": Counter(), "short": Counter()}
    for p in periods:
        for symbol, side in p["basket"].items():
            picks[side][symbol] += 1
    top_picks = {
        side: [{"symbol": s, "count": c, "pct_of_periods": round(c / weeks * 100.0, 1)} for s, c in counter.most_common(15)]
        for side, counter in picks.items()
    } if weeks else {"long": [], "short": []}

    recent = [
        {
            "rebalance_time": p["rebalance_time"],
            "exit_time": p["exit_time"],
            "longs": sorted(s for s, side in p["basket"].items() if side == "long"),
            "shorts": sorted(s for s, side in p["basket"].items() if side == "short"),
            "gross_return_pct": p["gross_return_pct"],
            "cost_pct": p["cost_pct"],
            "funding_cost_pct": p["funding_cost_pct"],
            "net_return_pct": p["net_return_pct"],
            "btc_return_pct": p.get("btc_return_pct"),
            "turnover": p["turnover"],
        }
        for p in reversed(periods)
    ]
    return {"metrics": metrics, "equity_curve": curve, "top_picks": top_picks, "periods": recent}


# --- Per-leg stop-loss sweep ----------------------------------------------
# "What would the win rate be with a stop loss?" -- replays the SAME baskets
# the engine formed, adding an X% stop on every leg, checked against each
# 1h candle's high/low during the holding week. Rules (documented, shown on
# the dashboard):
# * stop = entry * (1 - X%) for longs, entry * (1 + X%) for shorts;
# * a candle that opens beyond the stop (gap) fills at its open, not the stop;
# * a stopped leg's capital sits in cash until the next rebalance;
# * a stopped leg that the next basket keeps on the same side must be
#   re-opened -> one extra round-trip cost (the engine assumed it was held);
# * funding is charged only while the leg is actually open.
# ``None`` = no stop; it reproduces the engine's own period returns, which
# the tests check, so every row is directly comparable.

SL_LEVELS_PCT: tuple[float | None, ...] = (None, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)


def _leg_path(history, side: str, t: datetime, t_exit: datetime, sl_pct: float | None):
    """Returns (entry, exit_price, exit_time, stopped) or None if unpriced."""
    import pandas as pd

    from scanner.backtesting.cross_sectional_momentum_backtest import _price_at

    entry = _price_at(history, t)
    final = _price_at(history, t_exit)
    if entry is None or final is None or entry == 0:
        return None
    if sl_pct is not None:
        stop = entry * (1 - sl_pct / 100.0) if side == "long" else entry * (1 + sl_pct / 100.0)
        window = history[(history["open_time"] > pd.Timestamp(t)) & (history["open_time"] <= pd.Timestamp(t_exit))]
        if side == "long":
            hit = window[window["low"].astype(float) <= stop]
        else:
            hit = window[window["high"].astype(float) >= stop]
        if not hit.empty:
            row = hit.iloc[0]
            o = float(row["open"])
            fill = min(o, stop) if side == "long" else max(o, stop)
            exit_time = row["open_time"].to_pydatetime() + timedelta(hours=1)
            return entry, fill, min(exit_time, t_exit), True
    return entry, final, t_exit, False


def stop_loss_sweep(
    periods: list[dict[str, Any]],
    histories: dict[str, Any],
    funding_by_symbol: dict[str, list],
    spec: ls.LiveStrategySpec,
    levels: tuple[float | None, ...] = SL_LEVELS_PCT,
) -> list[dict[str, Any]]:
    from scanner.backtesting.cross_sectional_momentum_backtest import _leg_weight
    from scanner.backtesting.funding_simulator import funding_cost_pct
    from scanner.calibration import round_trip_cost_pct_for

    weight = _leg_weight(spec.basket_size)
    rows = []
    for level in levels:
        period_nets: list[float] = []
        leg_nets: list[float] = []
        legs_stopped = 0
        stopped_prev: dict[str, str] = {}  # symbol -> side stopped out last period
        prev_basket: dict[str, str] | None = None
        for p in periods:
            t = datetime.fromisoformat(p["rebalance_time"])
            t_exit = datetime.fromisoformat(p["exit_time"])
            basket = p["basket"]
            gross = funding = 0.0
            extra_cost = 0.0
            stopped_now: dict[str, str] = {}
            valid = True
            leg_results = []
            for symbol, side in basket.items():
                path = _leg_path(histories[symbol], side, t, t_exit, level)
                if path is None:
                    valid = False
                    break
                entry, exit_price, exit_time, stopped = path
                raw = (exit_price - entry) / entry * 100.0
                ret = raw if side == "long" else -raw
                fund = funding_cost_pct(side, t, exit_time, funding_by_symbol.get(symbol, []))
                new_entry = prev_basket is None or prev_basket.get(symbol) != side
                reentry = stopped_prev.get(symbol) == side and not new_entry
                leg_cost = round_trip_cost_pct_for(symbol, spec.base_cost_pct) if (new_entry or reentry) else 0.0
                if reentry:
                    extra_cost += weight * leg_cost
                if stopped:
                    stopped_now[symbol] = side
                gross += weight * ret
                funding += weight * fund
                leg_results.append((ret - fund - leg_cost, stopped))
            if not valid:
                continue
            period_nets.append(gross - p["cost_pct"] - extra_cost - funding)
            for net, stopped in leg_results:
                leg_nets.append(net)
                legs_stopped += int(stopped)
            stopped_prev = stopped_now
            prev_basket = basket

        equity = [1.0]
        for n in period_nets:
            equity.append(equity[-1] * (1 + n / 100.0))
        rows.append(
            {
                "stop_loss_pct": level,
                "n_periods": len(period_nets),
                "period_win_rate_pct": (
                    round(sum(1 for n in period_nets if n > 0) / len(period_nets) * 100, 1) if period_nets else None
                ),
                "avg_period_net_pct": round(sum(period_nets) / len(period_nets), 4) if period_nets else None,
                "compounded_net_pct": round((equity[-1] - 1) * 100, 4) if period_nets else None,
                "max_drawdown_pct": _max_drawdown_pct(equity),
                "worst_period_pct": round(min(period_nets), 4) if period_nets else None,
                "n_legs": len(leg_nets),
                "leg_win_rate_pct": (
                    round(sum(1 for n in leg_nets if n > 0) / len(leg_nets) * 100, 1) if leg_nets else None
                ),
                "avg_leg_net_pct": round(sum(leg_nets) / len(leg_nets), 4) if leg_nets else None,
                "legs_stopped_pct": round(legs_stopped / len(leg_nets) * 100, 1) if leg_nets else None,
                "period_nets": [round(n, 6) for n in period_nets],
            }
        )
    return rows


# --- Account simulation in dollars ----------------------------------------
# "I start with $2000, $200 per position": the same baskets, each leg a fixed
# dollar size on ISOLATED 1x margin -> a leg can never lose more than its own
# size (liquidation at -100%; Binance actually liquidates slightly earlier
# and charges a fee -- ignored). Optional per-leg TP/SL; when TP and SL are
# both inside one 1h candle the SL is assumed first (conservative). Costs:
# round-trip cost when a leg is (re)opened, funding while it is open.

ACCOUNT_START_USD = 2000.0
ACCOUNT_LEG_USD = 200.0
ACCOUNT_SCENARIOS: tuple[tuple[str, float | None, float | None], ...] = (
    ("Bez TP/SL", None, None),
    ("SL 30%", None, 30.0),
    ("SL 20%", None, 20.0),
    ("TP 10%", 10.0, None),
    ("TP 10% + SL 30%", 10.0, 30.0),
)


def _leg_outcome(history, side: str, t: datetime, t_exit: datetime, tp: float | None, sl: float | None):
    """(return %, exit time, reason) with liquidation cap at -100%, or None."""
    import pandas as pd

    from scanner.backtesting.cross_sectional_momentum_backtest import _price_at

    entry = _price_at(history, t)
    final = _price_at(history, t_exit)
    if entry is None or final is None or entry == 0:
        return None
    sign = 1.0 if side == "long" else -1.0
    w = history[(history["open_time"] > pd.Timestamp(t)) & (history["open_time"] <= pd.Timestamp(t_exit))]
    fav_col, adv_col = ("high", "low") if side == "long" else ("low", "high")
    opens = (w["open"].astype(float).to_numpy() / entry - 1) * 100 * sign
    fav = (w[fav_col].astype(float).to_numpy() / entry - 1) * 100 * sign
    adv = (w[adv_col].astype(float).to_numpy() / entry - 1) * 100 * sign
    times = w["open_time"].tolist()
    for i in range(len(opens)):
        exit_time = min(times[i].to_pydatetime() + timedelta(hours=1), t_exit)
        if adv[i] <= -100.0:
            return -100.0, exit_time, "liq"
        if sl is not None and adv[i] <= -sl:
            return float(opens[i] if opens[i] <= -sl else -sl), exit_time, "sl"
        if tp is not None and fav[i] >= tp:
            return float(opens[i] if opens[i] >= tp else tp), exit_time, "tp"
    return max((final / entry - 1) * 100 * sign, -100.0), t_exit, "end"


def account_simulation(
    periods: list[dict[str, Any]],
    histories: dict[str, Any],
    funding_by_symbol: dict[str, list],
    spec: ls.LiveStrategySpec,
    scenarios: tuple[tuple[str, float | None, float | None], ...] = ACCOUNT_SCENARIOS,
    start_usd: float = ACCOUNT_START_USD,
    leg_usd: float = ACCOUNT_LEG_USD,
) -> list[dict[str, Any]]:
    from scanner.backtesting.funding_simulator import funding_cost_pct
    from scanner.calibration import round_trip_cost_pct_for

    out = []
    for name, tp, sl in scenarios:
        equity = start_usd
        peak = equity
        max_dd = 0.0
        weekly: list[float] = []
        curve = []
        counts = {"tp": 0, "sl": 0, "liq": 0, "end": 0}
        prev: dict[str, str] = {}
        exited_prev: set[str] = set()
        for p in periods:
            t = datetime.fromisoformat(p["rebalance_time"])
            t_exit = datetime.fromisoformat(p["exit_time"])
            pnl = 0.0
            exited: set[str] = set()
            for symbol, side in p["basket"].items():
                res = _leg_outcome(histories[symbol], side, t, t_exit, tp, sl)
                if res is None:
                    continue
                ret, exit_time, reason = res
                counts[reason] += 1
                fund = funding_cost_pct(side, t, exit_time, funding_by_symbol.get(symbol, []))
                reopened = prev.get(symbol) != side or symbol in exited_prev
                cost = round_trip_cost_pct_for(symbol, spec.base_cost_pct) if reopened else 0.0
                pnl += leg_usd * (ret - fund - cost) / 100.0
                if reason != "end":
                    exited.add(symbol)
            equity += pnl
            weekly.append(pnl)
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
            curve.append({"time": p["exit_time"], "equity": round(equity, 2)})
            prev = dict(p["basket"])
            exited_prev = exited
        n_legs = sum(counts.values())
        out.append(
            {
                "name": name,
                "take_profit_pct": tp,
                "stop_loss_pct": sl,
                "start_usd": start_usd,
                "leg_usd": leg_usd,
                "end_usd": round(equity, 2),
                "return_pct": round((equity / start_usd - 1) * 100, 2),
                "weeks_positive": sum(1 for w in weekly if w > 0),
                "n_weeks": len(weekly),
                "best_week_usd": round(max(weekly), 2) if weekly else None,
                "worst_week_usd": round(min(weekly), 2) if weekly else None,
                "max_drawdown_usd": round(max_dd, 2),
                "tp_hit_pct": round(counts["tp"] / n_legs * 100, 1) if n_legs else None,
                "sl_hit_pct": round(counts["sl"] / n_legs * 100, 1) if n_legs else None,
                "liquidated_legs": counts["liq"],
                "curve": curve,
            }
        )
    return out


def _universe(variant: BasketBacktest, ranked_cache: str | None, top_n: int) -> tuple[tuple[str, ...], str | None, dict]:
    """Returns (symbols, temp PIT cache path or None, universe metadata)."""
    if ranked_cache and Path(ranked_cache).exists():
        from scanner.backtesting.historical_universe import (
            load_ranked_universe_cache,
            windows_for_rank_range,
        )
        from scanner.backtesting.strategy_miner import _write_tercile_universe_cache

        ranked = load_ranked_universe_cache(ranked_cache)
        width = max((len(w.ranked_symbols) for w in ranked), default=0)
        lo, hi = (1, width) if variant.tercile == "all" else ls.build_spec(variant.tercile).tercile_rank_range
        windows = windows_for_rank_range(ranked, lo, hi)
        symbols = tuple(sorted(set().union(*(w.symbols for w in windows))))
        tmp = _write_tercile_universe_cache(windows, variant.tercile)
        return symbols, str(tmp), {
            "source": "point_in_time_ranked_cache",
            "ranked_cache_path": ranked_cache,
            "rank_range": [lo, hi],
            "survivorship_bias": False,
        }

    from tradingagents.dataflows.vendors.binance import client

    current = client.get_top_symbols_by_quote_volume(top_n=top_n)
    if variant.tercile != "all":
        lo, hi = ls.build_spec(variant.tercile).tercile_rank_range
        current = current[lo - 1 : hi]
    return tuple(current), None, {
        "source": "current_top_by_24h_volume",
        "top_n": top_n,
        "survivorship_bias": True,
        "note": "Današnji top coini primijenjeni unazad: coini koji su propali/delistovani fale, pa rezultat izgleda bolje nego što bi bio.",
    }


def run(
    variant_id: str,
    *,
    ranked_universe_cache: str | None = None,
    top_n: int = DEFAULT_TOP_N,
    end_timestamp_ms: int | None = None,
) -> dict[str, Any]:
    from scanner.strategy_lab_scheduler import compute_rolling_as_of_ts

    variant = BASKET_BACKTESTS[variant_id]
    spec = ls.build_spec(variant.tercile)
    end_ms = end_timestamp_ms or compute_rolling_as_of_ts()
    symbols, pit_path, universe_meta = _universe(variant, ranked_universe_cache, top_n)
    config = cse.CrossSectionalConfig(
        hypothesis=spec.template,
        ranking_fn=cse.momentum_ranking_fn,
        lookback_hours=spec.lookback_hours,
        rebalance_hours=spec.rebalance_hours,
        basket_size=spec.basket_size,
    )
    try:
        engine_report = cse.run_backtest(
            config, symbols, end_timestamp_ms=end_ms, point_in_time_universe_cache=pit_path
        )
    finally:
        if pit_path:
            Path(pit_path).unlink(missing_ok=True)

    summary = summarize(engine_report)
    periods = engine_report.get("rebalance_periods", [])
    sl_sweep = None
    if periods:
        from scanner.backtesting.data_snapshot import (
            cached_fetch_full_funding_history,
            cached_load_extended_history,
        )

        needed = sorted({s for p in periods for s in p["basket"]})
        # Same cached files the engine just used -> no new network calls.
        histories = {
            s: cached_load_extended_history(s, cse.DEFAULT_PRICE_TIMEFRAME, cse.DEFAULT_PRICE_CANDLES, end_ms)
            for s in needed
        }
        funding = {s: cached_fetch_full_funding_history(s, end_ms) for s in needed}
        sl_sweep = stop_loss_sweep(periods, histories, funding, spec)
        account = account_simulation(periods, histories, funding, spec)
    return {
        "stop_loss_sweep": sl_sweep,
        "account_simulation": account if periods else None,
        "kind": "basket",
        "variant_id": variant.variant_id,
        "label": variant.label,
        "description": variant.description,
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": spec.as_dict(),
        "universe": {**universe_meta, "n_symbols_requested": len(symbols),
                     "n_symbols_used": len(engine_report.get("symbols_used", []))},
        "period_start": periods[0]["rebalance_time"] if periods else None,
        "period_end": periods[-1]["exit_time"] if periods else None,
        "end_timestamp_iso": engine_report.get("end_timestamp_iso"),
        "initial_capital": INITIAL_CAPITAL,
        "skipped_symbols": engine_report.get("skipped_symbols", []),
        "dataset_hash": engine_report.get("dataset_hash"),
        **summary,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest of the live basket strategy for the dashboard")
    parser.add_argument("--variant", choices=sorted(BASKET_BACKTESTS), default=MAIN_BASKET_VARIANT_ID)
    parser.add_argument("--ranked-universe-cache", default="")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args(argv)

    report = run(args.variant, ranked_universe_cache=args.ranked_universe_cache or None, top_n=args.top_n)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(out)
    m = report["metrics"]
    print(
        f"{report['label']}: {m['n_periods']} perioda, win {m['win_rate_pct']}%, avg net {m['avg_net_pct']}%, "
        f"ukupno {m['compounded_net_pct']}%, max DD {m['max_drawdown_pct']}% -> {out}"
    )
    for a in report.get("account_simulation") or []:
        print(
            f"   ${a['start_usd']:.0f} / ${a['leg_usd']:.0f} po poziciji, {a['name']:16}: kraj ${a['end_usd']:.0f} "
            f"({a['return_pct']:+.1f}%), sedmica u plusu {a['weeks_positive']}/{a['n_weeks']}, "
            f"najgora sedmica ${a['worst_week_usd']:.0f}, max pad ${a['max_drawdown_usd']:.0f}"
        )


if __name__ == "__main__":
    main()
