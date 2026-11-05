# FazlaChart — pregled sistema (šta ovo danas stvarno radi)

Ovaj dokument je jedno mjesto koje odgovara na pitanje "šta ovaj sistem
danas umije" — organizovano oko tabova live dashboard-a, kako ih korisnik
vidi. Nije arhitekturna knjiga ni line-by-line walkthrough; za statističku
metodologiju (train/holdout, pre-registracija, multiple-testing, DSR,
block-bootstrap) vidi `PHASE8_PROFITABILITY_RESEARCH_TODO.md`,
`PREREGISTRATION_*.md` i `STRATEGY_LAB_DESIGN.md` — ovaj dokument ih ne
ponavlja, samo referiše.

Sistem je privatni fork [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
prilagođen kripto futures-ima, i danas ima **dvije odvojene cjeline** koje
dijele kod ali se koriste nezavisno:

1. **Market Scanner + webapp dashboard** (`scanner/` + `webapp/`) — 24/7
   proces koji mjeri tržište i istraživačka infrastruktura iznad njega
   (Phase 8 + Strategy Lab). Nema LLM uključen, osim na eksplicitan klik.
2. **TradingAgents CLI** (`cli/` + `tradingagents/agents/`) — originalni,
   odvojeni multi-agent LLM pipeline (tim analitičara/istraživača/trejdera/
   risk-managera koji debatuju) za on-demand analizu jednog tickera na
   jedan datum. Ne dijeli runtime sa dashboard-om.

---

## 📡 Scanner (live tab)

`scanner/market_scanner.py` nad Binance Futures public REST API-jem (bez
API ključa za tržišne podatke), ciklus svakih **300 sekundi (5 min)**
(`poll_interval_seconds`, `scanner/config.py`).

- **Univerzum**: top-N po 24h volumenu (default 120) ili eksplicitna lista
  simbola, ograničen na max 250 simbola.
- **Timeframe-ovi po simbolu**: `5m, 15m, 1h, 4h`. BTC se posebno prati na
  `15m, 1h, 4h` kao referentni trend.
- **Indikatori po svijeći** (`tradingagents/dataflows/vendors/binance/indicators.py`):
  klasifikacija trenda, EMA20/EMA50 cross (+ cross-event), RSI, relativni
  volumen i promjena volumena, support/resistance (lookback 50), Fibonacci
  nivoi, momentum-fade provjera, volatility regime (ATR%-bazirano),
  drawdown%.
- **Derivativi**: funding rate i promjena open interest-a na 5m/15m/60m
  prozorima (`binance/snapshot.py`).
- **Anomalije** (`binance/anomaly.py` + `scanner/anomaly.py`): cijena raste
  bez potvrde volumenom, OI raste uz ravnu cijenu i skok funding-a, failed
  breakout, volatility squeeze-then-expansion — plus cross-symbol provjera
  **"altcoin se razilazi s BTC trendom"** (`btc_divergence_min_steps`,
  default 2 koraka razlike na istom timeframe-u) kao BTC-context filter.
- Svaki scan (snapshot + anomalije) se upisuje u `scanner_data/scanner.db`
  (SQLite) za kasniji replay/kalibraciju; retencija 60 dana.
- "🏆 Best signal" panel na dashboard-u rangira trenutne kandidate po
  keširanoj istorijskoj kalibraciji (win-rate CI), nikad ne pokreće
  backtest uživo po kliku — pozadinski "warmup loop" na serveru puni taj
  keš u pozadini.

## 📊 Analitika trejdova

Tab prikazuje svaki praćeni signal (otvoren i zatvoren) u punom detalju —
tačan entry/SL/TP/exit i obećano-naspram-ostvareno, ne samo agregatne
brojeve. Podržava raščlanu po bucket-u (kvaliteta signala), pattern-u,
tržišnom režimu i quality-score-u, s vremenskim rasponom (`range`/`since`/
`until`) i drift-status napomenom kad je neki bucket privremeno "mutovan"
zbog degradacije performansi (`scanner/drift_detector.py`).

## 🧪 Backtest

Tri odvojena, zamrznuta backtest engine-a, svaki testira jednu
pre-registrovanu hipotezu porodicu (sve tri su **zatvorene bez dokaza
edge-a** nakon troškova, vidi Korak 7–9 u Phase 8 TODO-u):

- **`portfolio_backtest.py`** — trend/anomaly direkcioni signal (`original`
  i `inverted` varijanta), portfolio-level counterfactual sa SL/TP
  egzekucijom, troškovima i walk-forward/OOS/independence validacijom.
  Ovo je i glavni "🧪 Backtest" tab dashboard-a (dropdown po varijanti,
  dugme "🔄 Osvježi" koje pokreće pun refresh kao odvojen, throttle-ovan
  background proces, nikad inline u web request-u).
- **`funding_extreme_backtest.py`** — funding-rate percentil-ekstrem
  reversion hipoteza, sa cooldown-om i dedup-om koreliranih signala,
  sopstvenim funding-cost simulatorom.
- **`cross_sectional_momentum_backtest.py`** — market-neutral long-top/
  short-bottom kvartil 24h momentum košare (dollar-neutral, turnover-
  bazirani troškovi, beta-provjera protiv BTC-a).

Sve tri dijele isti **jedan** zamrznuti statistički primitivni blok —
`scanner/validation/statistical_validation.py`'s
`block_bootstrap_confidence_interval()` — pozivan direktno iz sva tri
modula (nema tri paralelne re-implementacije CI logike).

## 🔬 Istraživanje

`GET /api/research/registry` čita `scanner_data/research_registry.json` —
ručno kurirani, human-maintained zapis svake pre-registrovane hipoteza-
porodice (potvrđena/odbačena), djeljeni zaključani holdout datum, i
**multiple-testing brojač** (koliko je porodica/varijanti ikad testirano —
računovodstvo protiv p-hacking-a). Ovo je izvor istine za istoriju "šta je
probano i zatvoreno", odvojeno od Strategy Lab-ove žive baze.

## ⚒️ Strategy Lab

Novi, kontinuirani dvoslojni pipeline (`scanner/backtesting/strategy_miner.py`,
`strategy_lab_runner.py`, `rolling_holdout.py`, `power.py`,
`scanner/strategy_lab_storage.py`, planiran kao odvojen systemd/scheduler
proces `scanner/strategy_lab_scheduler.py`):

- **Layer 1 (miner)** — neograničeno po broju pokretanja (default svakih
  6h), ali ograničeno po pre-deklarisanoj mreži varijanti (8 template-
  varijanti × do 3 likvidnosna tercila). Nikad ne računa CI niti izriče
  pass/fail — samo opisni rang ("trag za buduću potvrdu, ne nalaz").
- **Layer 2 (confirmation gate)** — rijedak po dizajnu: pokreće se samo
  kad je statistička snaga (power, z-test sample-size formula) stvarno
  dostignuta na **garantovano svježem**, nikad ranije korištenom podataka
  prozoru (odvojen rolling holdout od zaključanog 6-sedmičnog holdout-a
  koji koriste tri gornje porodice). Jedino mjesto u cijelom Strategy
  Lab-u sa pravim pass/fail jezikom (`edge_survives_lower_bound`).
- Sedmični snapshot + Telegram izvještaj, dashboard tab prikazuje trenutno
  najbolju potvrđenu strategiju (ili iskreno prazno stanje), Layer 1
  rang-listu, istoriju snapshotova/potvrda, i rolling-holdout status.

## Telegram alarmi

`scanner/telegram_alerts.py` — jedno mjesto za sve poruke: novi signal
alert (filtriran na `telegram_min_win_rate_ci_low_pct`/`telegram_min_n`/
opciono BTC-alignment), drift mute/unmute, Strategy Lab sedmični snapshot i
Layer 2 potvrda. Health watchdog (webapp startup loop, provjera na svakih
120s) šalje poseban alert (cooldown 1h) ako scanner prestane pisati svježe
podatke.

## "Analyze with AI" dugme

`POST /api/analyze/{symbol}` (`agents_crypto/derivatives_risk_agent.py`) —
**jedino mjesto** gdje dashboard napušta "read-only, bez LLM-a" svijet, i
samo na eksplicitan klik (nikad iz auto-refresh petlje). Uzima live
Binance snapshot + keširane kalibracione statistike + poziva jedan LLM.

- Provider/model se biraju preko `DASHBOARD_LLM_PROVIDER`/
  `DASHBOARD_LLM_MODEL` env varijabli (fallback default: **Ollama**,
  lokalni `llama3.1:8b`).
- LLM klijent stack (`tradingagents/llm_clients/factory.py`) je
  multi-provider (OpenAI, Anthropic, Google, Azure, Bedrock, Ollama,
  Groq i bilo koji OpenAI-kompatibilan endpoint) — isti registry koji
  koristi i TradingAgents CLI. Webapp na startu "zagrijava" LLM stack
  importom (ne mrežnim pozivom) protiv Groq providera, da prvi klik na
  dugme ne stoji zaglavljen iza sporog `langchain`/`openai` importa.
- TradingAgents CLI (odvojeno, ne ovaj dashboard) koristi `gpt-6-sol`/
  `gpt-6-luna` (OpenAI) kao default deep/quick-think model.

## CLI alati (`cli/`) — odvojeno od dashboard-a

`cli/main.py` (Typer app, `TradingAgents` originalni multi-agent LLM
finansijski trading framework): puna agent-debate analiza jednog
tickera/datuma (market/news/social/fundamentals analitičari → bull/bear
istraživači → trader → risk manager → portfolio manager), sa
checkpoint-resume, portfolio-aware run-om, i grid backtestom preko
tickera/datuma (`tradingagents/backtest.py`). Ovo je potpuno nezavisna
funkcionalnost od scanner/webapp dashboard-a — radi i na dionicama i na
kripto tickerima, ne čita `scanner_data/`.

## Kako radi 24/7 na VPS-u

Tri nezavisna, uvijek-živa procesa (svaki log-and-continue na grešku po
ciklusu, bez tight `while True` bez strukture):

1. `python -m scanner` (`market_scanner.py` `run_forever`) — glavni scan
   svakih 300s.
2. `uvicorn webapp.main:app` — read-only dashboard, nikad sam ne zove
   Binance/LLM osim na eksplicitan klik ("Analyze", "🔄 Osvježi backtest").
3. `python -m scanner.strategy_lab_scheduler` — Strategy Lab Layer 1/2,
   odvojen proces/systemd unit od (1).

Pogonski skriptovi za lokalni dev/VPS start-stop žive u `tool/`
(`run_scanner.ps1/.sh`, `run_dashboard.ps1/.sh`, `start_all.ps1`,
`stop_all.ps1`); runtime PID/log fajlovi iz tih skriptova idu u
`tool/.run/` (gitignored).
