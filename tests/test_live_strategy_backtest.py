"""Unit tests for scanner.backtesting.live_strategy_backtest (no network)."""

from __future__ import annotations

import pytest

from scanner.backtesting import live_strategy_backtest as lsb

pytestmark = pytest.mark.unit


def _period(i, net, basket, btc=1.0):
    return {
        "rebalance_time": f"2026-01-{i + 1:02d}T00:00:00+00:00",
        "exit_time": f"2026-01-{i + 2:02d}T00:00:00+00:00",
        "basket": basket,
        "gross_return_pct": net + 0.2,
        "cost_pct": 0.15,
        "funding_cost_pct": 0.05,
        "net_return_pct": net,
        "btc_return_pct": btc,
        "turnover": 0.4,
    }


def test_summarize_equity_drawdown_and_picks():
    b1 = {"AUSDT": "long", "BUSDT": "short"}
    b2 = {"AUSDT": "long", "CUSDT": "short"}
    report = {
        "rebalance_periods": [_period(0, 10.0, b1), _period(1, -20.0, b2), _period(2, 5.0, b1)],
        "win_rate_pct": 66.67,
        "avg_turnover": 0.4,
    }
    s = lsb.summarize(report)
    m = s["metrics"]
    assert m["n_periods"] == 3
    assert m["compounded_net_pct"] == pytest.approx((1.1 * 0.8 * 1.05 - 1) * 100, abs=1e-3)
    assert m["max_drawdown_pct"] == pytest.approx(-20.0, abs=1e-6)
    assert m["best_period_pct"] == 10.0 and m["worst_period_pct"] == -20.0
    assert s["equity_curve"][0]["equity"] == lsb.INITIAL_CAPITAL
    assert len(s["equity_curve"]) == 4
    assert s["top_picks"]["long"][0] == {"symbol": "AUSDT", "count": 3, "pct_of_periods": 100.0}
    assert s["periods"][0]["net_return_pct"] == 5.0  # newest first
    assert s["periods"][0]["longs"] == ["AUSDT"]


def test_summarize_empty_report():
    s = lsb.summarize({"rebalance_periods": []})
    assert s["metrics"]["n_periods"] == 0
    assert s["metrics"]["compounded_net_pct"] is None
    assert s["equity_curve"] == []


def test_variants_match_live_strategy_specs():
    assert lsb.BASKET_BACKTESTS[lsb.MAIN_BASKET_VARIANT_ID].tercile == "all"
    assert lsb.BASKET_BACKTESTS["momentum_30d_top"].tercile == "top"


def test_run_without_ranked_cache_uses_current_top_and_flags_bias(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tradingagents.dataflows.vendors.binance.client.get_top_symbols_by_quote_volume",
        lambda top_n: [f"S{i}USDT" for i in range(top_n)],
    )

    def fake_run_backtest(config, symbols, **kwargs):
        captured["symbols"] = symbols
        captured["config"] = config
        return {"rebalance_periods": [], "symbols_used": list(symbols)}

    monkeypatch.setattr(lsb.cse, "run_backtest", fake_run_backtest)
    report = lsb.run("momentum_30d_top", top_n=100, end_timestamp_ms=1_790_000_000_000)
    assert len(captured["symbols"]) == 57
    assert captured["config"].lookback_hours == 336 and captured["config"].rebalance_hours == 168
    assert report["universe"]["survivorship_bias"] is True
    assert report["kind"] == "basket"


# --- stop-loss sweep -------------------------------------------------------

from datetime import UTC, datetime, timedelta  # noqa: E402

import pandas as pd  # noqa: E402

from scanner import live_strategy as ls  # noqa: E402
from scanner.backtesting import cross_sectional_engine as cse  # noqa: E402
from scanner.backtesting.funding_simulator import FundingEvent  # noqa: E402

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
SPEC1 = ls.LiveStrategySpec(
    key="t", template="momentum_30d", tercile="top", variant_label="t",
    lookback_hours=24, rebalance_hours=48, basket_size=1, tercile_rank_range=(1, 57),
)


def _hist(closes, lows=None, highs=None, opens=None):
    n = len(closes)
    return pd.DataFrame(
        {
            "open_time": [pd.Timestamp(_T0 + timedelta(hours=i)) for i in range(n)],
            "open": opens or closes,
            "high": highs or closes,
            "low": lows or closes,
            "close": closes,
        }
    )


def _engine_periods(histories, funding, baskets_at):
    periods, prev = [], None
    for t_h, basket in baskets_at:
        t, t_exit = _T0 + timedelta(hours=t_h), _T0 + timedelta(hours=t_h + 48)
        p = cse.build_rebalance_period(t, t_exit, 2, basket, prev, histories, funding, 0.10, 1)
        periods.append({
            "rebalance_time": p.rebalance_time.isoformat(), "exit_time": p.exit_time.isoformat(),
            "basket": p.basket, "gross_return_pct": p.gross_return_pct, "cost_pct": p.cost_pct,
            "funding_cost_pct": p.funding_cost_pct, "net_return_pct": p.net_return_pct,
        })
        prev = basket
    return periods


def test_no_stop_reproduces_engine_period_returns():
    up = [100 + i for i in range(100)]
    down = [200 - i for i in range(100)]
    histories = {"AUSDT": _hist(up), "BUSDT": _hist(down)}
    funding = {"AUSDT": [FundingEvent(_T0 + timedelta(hours=8 * k), 0.0001) for k in range(12)], "BUSDT": []}
    periods = _engine_periods(histories, funding, [(0, {"AUSDT": "long", "BUSDT": "short"}),
                                                   (48, {"AUSDT": "long", "BUSDT": "short"})])
    sweep = lsb.stop_loss_sweep(periods, histories, funding, SPEC1, levels=(None,))
    assert sweep[0]["period_nets"] == pytest.approx([p["net_return_pct"] for p in periods], abs=1e-6)
    assert sweep[0]["legs_stopped_pct"] == 0.0


def test_stop_hits_and_gap_fills_at_open():
    # Long leg: entry close 100 at h0; at h5 the candle opens at 80 (gap below
    # a 10% stop at 90) -> fill at 80, not 90.
    closes = [100.0] * 60
    opens = list(closes)
    lows = list(closes)
    opens[5], lows[5], closes[5] = 80.0, 79.0, 81.0
    histories = {"AUSDT": _hist(closes, lows=lows, opens=opens), "BUSDT": _hist([50.0] * 60)}
    periods = _engine_periods(histories, {}, [(0, {"AUSDT": "long", "BUSDT": "short"})])
    no_sl, sl10 = lsb.stop_loss_sweep(periods, histories, {}, SPEC1, levels=(None, 10.0))
    assert sl10["legs_stopped_pct"] == 50.0
    # long leg realised -20% (gap fill), weight 0.5 -> -10% basket gross
    assert sl10["period_nets"][0] == pytest.approx(-10.0 - periods[0]["cost_pct"], abs=1e-6)
    assert no_sl["period_nets"][0] == pytest.approx(periods[0]["net_return_pct"], abs=1e-6)


def test_stopped_leg_kept_next_period_pays_reentry():
    closes = [100.0] * 120
    lows = list(closes)
    lows[10] = 85.0  # touches a 10% stop intrabar, closes back at 100
    histories = {"AUSDT": _hist(closes, lows=lows), "BUSDT": _hist([50.0] * 120)}
    basket = {"AUSDT": "long", "BUSDT": "short"}
    periods = _engine_periods(histories, {}, [(0, basket), (48, basket)])
    sl10 = lsb.stop_loss_sweep(periods, histories, {}, SPEC1, levels=(10.0,))[0]
    from scanner.calibration import round_trip_cost_pct_for

    assert sl10["period_nets"][1] == pytest.approx(-0.5 * round_trip_cost_pct_for("AUSDT"), abs=1e-6)


def test_account_simulation_caps_loss_at_leg_size_and_counts_tp():
    # Short leg on a coin that 8x's (like the AKE squeeze) -> liquidated, loses
    # exactly its $200, not 700%. Long leg rises +12% at h3 -> TP 10% hit.
    squeeze = [10.0] * 4 + [80.0] * 56
    rally = [100.0] * 3 + [112.0] * 57
    # h3 opens at 100 and trades up to 112 intrabar -> TP fills at exactly +10%
    histories = {"AUSDT": _hist(rally, opens=[100.0] * 4 + [112.0] * 56), "BUSDT": _hist(squeeze)}
    periods = _engine_periods(histories, {}, [(0, {"AUSDT": "long", "BUSDT": "short"})])
    no_rules, tp10 = lsb.account_simulation(
        periods, histories, {}, SPEC1, scenarios=(("none", None, None), ("tp", 10.0, None))
    )
    from scanner.calibration import round_trip_cost_pct_for

    cost = 200 * (round_trip_cost_pct_for("AUSDT") + round_trip_cost_pct_for("BUSDT")) / 100
    assert no_rules["liquidated_legs"] == 1
    assert no_rules["end_usd"] == pytest.approx(2000 + 200 * 0.12 - 200 - cost, abs=0.01)
    assert tp10["end_usd"] == pytest.approx(2000 + 200 * 0.10 - 200 - cost, abs=0.01)
    assert tp10["tp_hit_pct"] == 50.0


def test_all_terciles_registered():
    assert {v.tercile for v in lsb.BASKET_BACKTESTS.values()} == {"all", "top", "mid", "bottom"}
