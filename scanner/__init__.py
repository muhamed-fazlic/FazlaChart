"""Market Scanner: a continuously-running service that objectively measures
the futures market (price, volume, indicators, funding, open interest) for
many symbols at once, filtered through BTC context and flagged for anomalies.

It deliberately does **not** call an LLM or output BUY/SELL. Its job is to
produce the structured fact sheet described in the project brief; agents,
rule engines, or backtesting consume its output afterwards.
"""
