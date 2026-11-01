"""Unit tests for the deterministic StockTwits sentiment-count helper and
the crypto sentiment snapshot built on top of it. No network calls --
``urlopen`` is mocked."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agents_crypto.sentiment import get_crypto_sentiment_snapshot
from tradingagents.dataflows.vendors import stocktwits


def _stream(*sentiments):
    payload = {
        "messages": [
            {"body": "x", "created_at": "2026-01-09T15:00:00Z", "user": {"username": "u"}, "entities": {"sentiment": {"basic": s} if s else {}}}
            for s in sentiments
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    return _Resp()


pytestmark = pytest.mark.unit


class TestFetchStocktwitsSentimentCounts:
    def test_counts_bullish_bearish_and_unlabeled(self):
        with patch.object(stocktwits, "urlopen", return_value=_stream("Bullish", "Bullish", "Bearish", None)):
            counts = stocktwits.fetch_stocktwits_sentiment_counts("BTCUSDT")
        assert counts == {
            "bullish": 2,
            "bearish": 1,
            "unlabeled": 1,
            "total": 4,
            "bullish_pct": 50.0,
            "bearish_pct": 25.0,
            "error": None,
        }

    def test_transport_error_returns_zeroed_counts_with_error_set(self):
        class _Raising:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                raise TimeoutError("slow")

        with patch.object(stocktwits, "urlopen", return_value=_Raising()):
            counts = stocktwits.fetch_stocktwits_sentiment_counts("BTCUSDT")
        assert counts["total"] == 0
        assert counts["error"] is not None


class TestGetCryptoSentimentSnapshot:
    def test_labels_bullish_when_bullish_clearly_dominates(self):
        with patch.object(stocktwits, "urlopen", return_value=_stream(*(["Bullish"] * 15 + ["Bearish"] * 2))):
            snap = get_crypto_sentiment_snapshot("BTCUSDT")
        assert snap["label"] == "bullish"
        assert snap["insufficient_reason"] is None

    def test_labels_neutral_when_split_is_close(self):
        with patch.object(stocktwits, "urlopen", return_value=_stream(*(["Bullish"] * 10 + ["Bearish"] * 9))):
            snap = get_crypto_sentiment_snapshot("BTCUSDT")
        assert snap["label"] == "neutral"

    def test_insufficient_data_below_minimum_labeled_messages(self):
        with patch.object(stocktwits, "urlopen", return_value=_stream("Bullish", "Bearish")):
            snap = get_crypto_sentiment_snapshot("OBSCUREUSDT")
        assert snap["label"] is None
        assert "Only" in snap["insufficient_reason"]

    def test_error_surfaces_as_insufficient_reason_not_a_crash(self):
        class _Raising:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                raise TimeoutError("slow")

        with patch.object(stocktwits, "urlopen", return_value=_Raising()):
            snap = get_crypto_sentiment_snapshot("BTCUSDT")
        assert snap["label"] is None
        assert "unavailable" in snap["insufficient_reason"]
