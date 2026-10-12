"""Crypto futures interpretation agents.

Deliberately separate from ``tradingagents/agents`` (the original stock-
oriented multi-agent pipeline with a portfolio manager, researcher debates,
etc.): this package is small and scanner-native. It takes facts the Market
Scanner has *already* objectively measured (see ``scanner.*`` and
``tradingagents.dataflows.vendors.binance.snapshot``) plus the calibration
backtest's historical win-rates (see ``scanner.calibration``), and asks an
LLM to interpret — never to invent a probability or refetch data itself.
"""
