"""Low-level REST client for Binance USDT-M Futures public market data.

Every function here calls one public ``fapi.binance.com`` endpoint, raises
the shared :mod:`tradingagents.dataflows.errors` taxonomy on failure, and
returns plain Python data (list/dict) — no pandas, no indicator logic. That
lives in :mod:`tradingagents.dataflows.vendors.binance.indicators` so this
module stays a thin, easily-testable HTTP boundary.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

import requests

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"

# Binance kline interval strings the scanner is expected to request.
# https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
VALID_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5
_TIMEOUT_SECONDS = 10

_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET a public futures endpoint, retrying transient failures.

    Raises ``VendorUnavailableError`` after exhausting retries so the caller
    (router/scanner) can skip this symbol/cycle instead of crashing the loop.
    """
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
            if response.status_code == 429 or response.status_code == 418:
                # Rate-limited / IP banned: back off and retry.
                last_error = VendorUnavailableError(
                    f"Binance rate limit on {path} (status {response.status_code})"
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            if 400 <= response.status_code < 500:
                # Permanent client error (e.g. unknown/invalid symbol): retrying
                # with the same request won't help, so fail fast instead of
                # burning _MAX_RETRIES * backoff seconds on a doomed call.
                symbol = (params or {}).get("symbol", "?")
                raise NoMarketDataError(
                    symbol,
                    detail=f"Binance rejected {path} (status {response.status_code}): {response.text[:200]}",
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Binance request failed (attempt %d/%d) %s: %s", attempt, _MAX_RETRIES, path, exc)
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise VendorUnavailableError(f"Binance futures request to {path} failed: {last_error}") from last_error


def get_exchange_info() -> dict[str, Any]:
    """Return the full futures exchange info (symbols, filters, status)."""
    return _get("/fapi/v1/exchangeInfo")


def get_usdt_perpetual_symbols() -> list[str]:
    """Return every symbol trading as a USDT-margined PERPETUAL contract."""
    info = get_exchange_info()
    symbols = [
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    ]
    if not symbols:
        raise NoMarketDataError("USDT-PERPETUAL", detail="exchangeInfo returned no tradable symbols")
    return symbols


def get_klines(
    symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"],
    interval: Annotated[str, "kline interval, e.g. 15m, 1h, 4h"],
    limit: Annotated[int, "number of candles, max 1500"] = 300,
) -> list[dict[str, Any]]:
    """Return OHLCV candles as a list of dicts (oldest first)."""
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}; expected one of {VALID_INTERVALS}")
    raw = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not raw:
        raise NoMarketDataError(symbol, detail=f"no klines returned for interval {interval}")
    return [dict(zip(_KLINE_COLUMNS, row, strict=False)) for row in raw]


def get_mark_price_and_funding(symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"]) -> dict[str, Any]:
    """Current mark price, index price, and the *current* (unsettled) funding rate."""
    data = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
    if not data:
        raise NoMarketDataError(symbol, detail="no premiumIndex data")
    return data


def get_funding_rate_history(
    symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"],
    limit: Annotated[int, "number of funding settlements, max 1000"] = 8,
) -> list[dict[str, Any]]:
    """Historical (settled, every 8h) funding rate entries, oldest first."""
    data = _get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
    return data or []


def get_open_interest(symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"]) -> dict[str, Any]:
    """Current open interest, in contracts (base asset units)."""
    data = _get("/fapi/v1/openInterest", {"symbol": symbol})
    if not data:
        raise NoMarketDataError(symbol, detail="no openInterest data")
    return data


def get_open_interest_history(
    symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"],
    period: Annotated[str, "5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d"] = "5m",
    limit: Annotated[int, "number of points, max 500"] = 30,
) -> list[dict[str, Any]]:
    """Open-interest history snapshots. Binance only retains ~30 days of this."""
    data = _get(
        "/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit},
    )
    return data or []


def get_24hr_ticker(symbol: Annotated[str, "futures symbol, e.g. BTCUSDT"] | None = None) -> Any:
    """24h rolling ticker stats. Omit ``symbol`` to get every symbol at once."""
    params = {"symbol": symbol} if symbol else None
    return _get("/fapi/v1/ticker/24hr", params)


def get_top_symbols_by_quote_volume(
    top_n: Annotated[int, "how many symbols to return"] = 50,
    quote_asset: Annotated[str, "quote asset filter"] = "USDT",
) -> list[str]:
    """Rank USDT perpetuals by 24h quote volume, highest first.

    This is the default universe for the scanner: liquid symbols only, so
    indicators and funding/OI reads are meaningful and tradable.
    """
    perpetuals = set(get_usdt_perpetual_symbols())
    tickers = get_24hr_ticker()
    ranked = sorted(
        (t for t in tickers if t.get("symbol") in perpetuals),
        key=lambda t: float(t.get("quoteVolume", 0.0)),
        reverse=True,
    )
    return [t["symbol"] for t in ranked[:top_n]]
