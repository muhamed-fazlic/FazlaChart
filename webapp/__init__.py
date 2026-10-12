"""Local web dashboard over the Market Scanner's SQLite output.

This is intentionally a read-only view of ``scanner_data/scanner.db`` — it
never calls Binance or an LLM itself. See :mod:`webapp.main`.
"""
