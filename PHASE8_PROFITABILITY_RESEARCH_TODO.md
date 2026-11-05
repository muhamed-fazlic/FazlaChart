# Phase 8 — Profitability Research Engine (TODO)

Cilj ove faze: ne dodavati jos indikatora/agenata, nego **dokazati (ili oboriti)**
da Top Signal Now selekcija signala stvara pozitivnu neto ocekivanu vrijednost
na potpuno nezavisnim podacima, nakon realnih troskova. Postojeci live scanner
(`Top Signal Now`, tezine, ranking) se NE dira dok se ovo ne zavrsi — ovo je
odvojen, paralelan research/validation sloj.

Status legenda: `[ ]` nije zapoceto, `[~]` u toku, `[x]` zavrseno.

---

## Korak 0 — Code review postojeceg koda (jeftino, radi se prvo) — ZAVRSENO

Rezultati code review-a (background agent, cijela analiza u istoriji chata):

- [x] `scanner/calibration.py` — **Nema problema.** Potvrdjeno point-in-time:
      `backtest_symbol()` koristi `window = df.iloc[:i+1]`, svi indikatori
      (ATR/EMA/RSI preko stockstats) su kauzalni (nema `center=True`).
      `_walk_forward_exit()` ispravno gleda samo unaprijed za ISHOD, i
      konzervativno pretpostavlja SL prvi kad su oba u istoj svijeci.
- [x] `agents_crypto/best_signal.py` — **Nema problema (trenutno).** Live
      ranking ne curi buduce informacije. UPOZORENJE za Korak 2: ako se
      ponovo iskoristi `calibration_lookup.py`-ev TTL kes (6h, racuna se
      relativno na "sada") za portfolio backtest replay, MORA se "zamrznuti"
      kalibracija po historijskom timestamp-u, inace replay tiho curi
      buducnost. Rijesiti PRIJE Koraka 2, ne sada.
- [x] `scanner/storage.py` — **Potvrdjen problem (djelimican).** `scans`
      tabela cuva sve skenirane simbole, ali `prune_old_scans()` brise sve
      starije od 60 dana (default) + VACUUM. Trajno se cuva (u
      `signal_outcomes`) samo Top-5-po-timeframe-u (`DEFAULT_TRACKING_LIMIT
      = 5`). Bez decision-context arhive (rank, razlog odbijanja, da li
      izabran) za SVE kandidate -> blokira counterfactual test. Ovo je
      tacno ono sto Korak 1 (candidate_archive.py) rjesava.
- [x] `scanner/drift_detector.py` — **Potvrdjen problem + djelimicna praznina.**
      `compute_bucket_drift()` koristi zadnjih 60 resolved signala po
      (timeframe, direction) bucketu, pooled preko SVIH simbola — nista ne
      deduplicira korelisane/preklapajuce signale (npr. 5 altcoina koji
      prate isti BTC pomak u isto vrijeme). Prati SAMO win rate, ne
      expectancy (55% win sa +0.3% dobitkom i 45% loss sa -3% gubitkom bi
      izgledalo "zdravo" iako je neto negativno).
- [x] `agents_crypto/risk_calc.py` — **Potvrdjen problem.** `risk_reward_ratio`
      se racuna BEZ fee/slippage. Fee/slippage ulazi samo preko
      `calibration.py`-evog FIKSNOG `DEFAULT_ROUND_TRIP_COST_PCT = 0.10`
      (isto za 5m scalp i 4h swing). Funding payments se NIGDJE ne
      modeliraju (ni ovdje ni u calibration.py — potvrdjeno odsutno).
- [x] `scanner/market_scanner.py` — **Potvrdjen problem (survivorship bias).**
      `resolve_universe()` uvijek zove `get_top_symbols_by_quote_volume()`
      za DANASNJI top-N, union sa `fetch_known_symbols()` (svi ikad
      skenirani simboli, BEZ timestamp-a kada su usli/izasli iz top-N).
      Nema perzistentne historijske membership tabele — ovo je tacno ono
      sto Korak 4 (historical_universe.py) rjesava.
- [x] `webapp/` — **Uglavnom OK, jedna UX nijansa.** Postoji jasan
      "Praćenje uživo" panel koji razdvaja "Obećano" (backtest) od "Stvarni
      win rate" (realized). ALI glavna tabela skenera ima kolonu "Win rate"
      BEZ vidljive oznake "(backtest)" u headeru — razlika postoji samo u
      hover tooltip-u. Nije data-bug, ali UX poboljsanje vrijedno razmatranja.

**Prioritet za sljedece korake (po ozbiljnosti nalaza):**
1. Najozbiljnije: #6 survivorship bias (Korak 4) i #3 samo-Top-5 trajno
   cuvanje (Korak 1) — oba direktno blokiraju Korak 2 (portfolio backtest).
2. Takodje ozbiljno: #5 flat fee + nema funding troska (Korak 5) —
   vjerovatno precjenjuje edge na kracim timeframe-ovima.
3. Treba ljudsku odluku: #4 drift detector (definicija "jednog trzisnog
   dogadjaja" je sudska odluka, ne cisti bug) i #7 UI oznaka (UX poziv).

---

## Usputno uradjeno (prije Koraka 1) — bogatiji podaci po trejdu — ZAVRSENO

Van redoslijeda koraka, na korisnikov zahtjev: `signal_outcomes` sada
pamti znatno vise po svakom trejdu, direktno korisno za Korak 1/2/6 kasnije
(commit `892dc4a`):

- **MFE/MAE** (`mfe_pct`/`mfe_price`, `mae_pct`/`mae_price`) — najbolja i
  najgora cijena koju je trejd dotakao u svom "zivotu", prati se SVAKI ciklus
  dok je otvoren (ne samo uslovno na koju je stranu na kraju otisao).
  Isti sampling limit kao i resolucija (close po skener-ciklusu, ~5 min).
- **entry_snapshot_json** — zamrznuta slika indikatora (RSI, EMA9/20/50/200,
  MACD, ATR, Bollinger, volatility regime, relative volume, support/
  resistance), derivatives (funding rate, OI i OI promjene), Quality Score
  breakdown-a, BTC-korelacionog taga (`diverges_from_btc`, `correlation`) i
  BTC trenda/cijene — sve u trenutku logovanja, jer se `scans` tabela brise
  nakon 60 dana (vidi Korak 1 ispod — ovo je djelimicna zamjena dok
  `candidate_archive.py` ne postoji).
- **tp1_hit_at / tp2_hit_at / tp3_hit_at** — tacan timestamp kad je svaki
  partial-close target bankovan (multi-target 1h/4h trejdovi).
- **timeout_candles** — live-tracker timeout je odvojen od backtest/
  kalibracionog horizonta (`BEST_SIGNAL_HORIZON_CANDLES=12`, nepromijenjen):
  72 svijece za multi-target 1h/4h (vec zasticeno breakeven stopom nakon
  TP1), 24 svijece za single-target 5m/15m.

---

## Korak 1 — Candidate archive (preduslov za sve ostalo) — ZAVRSENO (osnova)

- [x] Novi modul: `scanner/candidate_archive.py` (`record_candidates()`,
      plus `scanner/storage.py`'s nova `candidate_archive` tabela,
      `fetch_candidate_archive()`, `prune_old_candidate_archive()`).
- [x] Trajno cuvati SVE eligible kandidate (ne samo Top 5), sa:
  - feature vrijednosti u trenutku odluke (pun kandidat dict kao
    `candidate_json`, plus denormalizovane kolone za upite: win rate/CI/n,
    regime, quality score, entry/SL/TP, itd.)
  - razlog odbijanja (ako je odbijen): `rank_exceeds_limit` ili
    `muted_bucket`
  - ranking poziciju (`rank`, 1-based, po timeframe-u)
  - da li je odabran (`selected`)
- [ ] kasniji ishod za one koji NISU izabrani — **namjerno odgodjeno na
      Korak 2.** Umjesto da se gradi posebno live forward-tracking i za
      odbijene kandidate (sto bi dupliciralo `signal_outcomes`-ovu MFE/MAE
      masinu za svaki odbijeni kandidat, svaki ciklus), Korak 1 samo
      arhivira dovoljno konteksta (simbol, timeframe, smjer, entry/SL/TP,
      scanned_at) da Korak 2-ov point-in-time replay (isti princip kao
      `scanner.calibration.backtest_symbol`) moze KASNIJE izracunati ishod
      za bilo kojeg kandidata iz historijskih klines-a, selected ili ne —
      ne treba live pracenje unaprijed.
- [x] `agents_crypto/best_signal.py`'s `compute_candidates()` dobio opcioni
      `include_rejected=True` parametar koji vraca
      `all_eligible_by_timeframe` (svi eligible kandidati po timeframe-u,
      tagovani `rank`/`selected`/`muted`/`rejection_reason`) — default
      `False` ne mijenja nista u postojecem payload-u/ponasanju (dashboard
      ovo nikad ne prosljedjuje).
- [x] `scanner/market_scanner.py` racuna kandidate JEDNOM po ciklusu
      (`include_rejected=True`) i prosljedjuje isti rezultat i
      `log_new_signals` (zivi tracker) i `record_candidates` (arhiva) —
      nema duplog calibration-lookup posla po ciklusu.
- [x] Testovi: `tests/test_best_signal.py::TestIncludeRejected` (6 novih),
      `tests/test_candidate_archive.py` (7 novih). Puna unit/integration
      suita i dalje pokazuje samo iste 8 postojecih, nepovezanih failova.
- [ ] Ovo omoguceva pravi counterfactual test kasnije (Korak 2).

---

## Korak 2 — Portfolio-level counterfactual backtest (najveci prioritet) — ZAVRSENO (v1)

- [x] Novi modul: `scanner/backtesting/portfolio_backtest.py`
- [x] Replay svih historijskih signala hronoloski, koristeci SAMO informacije
      dostupne u tom trenutku (no look-ahead). Implementirano preko
      resolution-time-gated accumulatora: case-ov *ishod* ulazi u statistiku
      tek kad `resolution_time <= trenutni_entry_time`, ne samo kad case
      *postoji* (entry_time) — sprijecava suptilniji look-ahead bug nego
      obicno sortiranje po entry_time. Pokriveno testom
      (`TestNoLookAhead.test_candidate_only_sees_already_resolved_cases`) koji
      eksplicitno provjerava da kandidat na odredjenom satu koristi tacno N
      prethodno-rezolvanih slucajeva, nikad svoj vlastiti (jos nerazrijeseni)
      ishod.
- [x] Primijeniti isti ranking koji koristi live sistem. Isti gate kao
      `agents_crypto.best_signal.compute_candidates`: min sample size po
      timeframe-u (`BEST_SIGNAL_MIN_SAMPLE_SIZE`) + Wilson 95% CI-low > 50%,
      isti regime-vs-blended fallback kao
      `agents_crypto.calibration_lookup.find_cached_regime_matched_stats`
      (regime bucket se koristi cim ima ijedan rezolvan slucaj, cak i ispod
      min_n — gate se primjenjuje tek poslije, na koji god bucket je
      izabran). **Vazna arhitektonska napomena**: kalibracija je per
      (symbol, timeframe), NIKAD pool-ovana preko vise simbola (tacno kao u
      produkciji) — svaki (symbol, timeframe) par ima svoj nezavisni
      accumulator.
- [x] Simulirati ogranicen broj istovremenih pozicija (portfolio exposure
      limit = `max_concurrent_positions`), kapital, naknade i slippage
      (`round_trip_cost_pct` + `extra_slippage_pct`, oduzeto iz net_return
      prije R-multiple konverzije). Margina nije eksplicitno modelirana
      (fixed-fractional position sizing po R-multiple ne zahtijeva apsolutnu
      velicinu pozicije niti margin-nivo — vidi napomenu o R-multiple niz).
- [x] Primijeniti korelaciju izmedju istovremenih signala — **POJEDNOSTAVLJENO
      za v1**: `max_concurrent_per_direction` (koliko long/short pozicija
      smije biti otvoreno istovremeno) kao proxy za pravu BTC-divergence
      korelaciju koju live sistem koristi
      (`agents_crypto.best_signal._tag_correlated_exposure`). Prava
      implementacija bi zahtijevala paralelnu point-in-time BTC trend seriju
      po svakom historijskom timestampu — odlozeno, eksplicitno dokumentovano
      kao poznato pojednostavljenje u docstringu modula.
- [x] Sprijeciti otvaranje vise pozicija nego sto kapital dozvoljava —
      `max_concurrent_positions` + fixed-fractional sizing (svaki trejd
      rizikuje `risk_per_trade_pct`% equity-a *u trenutku otvaranja*).
- [x] Pratiti equity curve kroz vrijeme — azurira se na svakom zatvaranju
      pozicije (resolution_time), ne kontinuirano mark-to-market.
- [x] Izracunati i izvjestiti sve trazene metrike (net profit, CAGR,
      Sharpe/Sortino, profit factor, max drawdown, Calmar, expectancy,
      exposure-adjusted return, turnover/troskovi, broj trejdova i broj
      nezavisnih trzisnih epizoda) — vidi `_compute_metrics` i
      `_count_independent_episodes`.
  - **Dokumentovana pojednostavljenja** (navedena i u docstringu modula, ne
    kriju se): Sharpe/Sortino su anualizovani preko `trades_per_year`
    (broj trejdova / raspon u godinama), ne preko resample-ovane equity
    krive na fiksne periode — standardna retail-backtest aproksimacija, ne
    rigorozan time-series Sharpe. Turnover/troskovi su `n_trades *
    (round_trip_cost_pct + extra_slippage_pct)` — sumarna metrika, dok je
    stvarni trosak po trejdu vec ukalkulisan u equity curve prije ovoga.
    Nezavisne epizode se klasteruju samo po (smjer + preklapajuci
    vremenski prozor) — ne hvata korelaciju izmedju razlicitih smjerova
    koji dijele istu BTC izlozenost.
- [x] Dodan `sl_distance_pct` field na `scanner.calibration.LabeledCase`
      (populisan u `backtest_symbol()`-u iz `entry_atr *
      DEFAULT_SL_ATR_MULTIPLIER / entry_close * 100`) — omogucava
      R-multiple konverziju (`net_return_pct / sl_distance_pct`) za
      risk-normalizovano position sizing, bez potrebe za apsolutnim
      velicinama pozicija.
- [x] Testovi: `tests/test_portfolio_backtest.py` (15 testova) — direction
      correction, no-look-ahead (2 scenarija), regime fallback (2
      scenarija), portfolio admission (concurrency + per-direction cap +
      vendor error handling), metrike (poznati rucno-izracunati primjeri),
      nezavisne epizode. Sve sintetske, bez mrezastih poziva.
- [x] CLI entry point: `python -m scanner.backtesting.portfolio_backtest
      --symbols ... --timeframes ...` — stampa metrike kao JSON, opciono
      dump-uje pun trade log preko `--trades-json`. **Nije ukljucen u
      `fazlachart-scanner` servis** (resource constraint na 2 vCPU/4GB VPS-u
      — fetch-uje ~1500 klines-a po (symbol, timeframe) paru preko mreze,
      namjenski on-demand alat, pokrece se rucno kad treba analiza).
- [x] **Jos nije pokrenuto nad stvarnim istorijskim podacima** — sljedeci
      korak je zapravo izvrsiti backtest nad trenutnim univerzumom simbola i
      protumaciti rezultate (CAGR/Sharpe/drawdown itd.) pre nego sto se
      bilo sta proglasi "dokazanom prednoscu".
  - **Pokrenuto na VPS-u** (20 likvidnih simbola x 4 timeframe-a, puni
    realni scanner.db univerzum): rezultat znacajno negativan (net profit
    -65.45%, win rate 39.8%, profit factor 0.77, expectancy -0.145R/trade,
    max DD 67%, Sharpe -4.36) — **ALI otkriven ozbiljan metodoloski
    problem prije donosenja zakljucka**: v1 je gradio stats accumulator
    "od nule" na pocetku istog 1500-candle prozora koji se replay-uje, dok
    produkcija uvijek vec ima ~1500 prethodnih candle-a kalibracije prije
    nego sto se ijedna odluka ikad prikaze korisniku ("cold-start bias"
    koji vjestacki pogorsava rani/srednji dio simulacije).
- [x] **Cold-start fix implementiran** (korisnik eksplicitno odabrao "prvo
      popravi, pa ponovi test" prije donosenja zakljucaka):
  - `scanner/calibration.py::load_extended_history()` — nova funkcija,
    pagira unazad preko Binance-ovog 1500-candle limita po requestu
    (koristeci novi `end_time` parametar na
    `tradingagents.dataflows.vendors.binance.client.get_klines()`),
    spaja/dedup-uje/sortira sve batch-eve, pa racuna indikatore JEDNOM nad
    cijelom spojenom serijom (EMA/ATR zahtijevaju neprekinutu seriju — racunanje
    po batch-u pa spajanje bi pokvarilo indikatore oko granica batch-eva).
  - `scanner.calibration.backtest_symbol()` sad prima opcioni `history:
    pd.DataFrame | None` parametar — kad je zadan, preskace interni
    `_load_history()` fetch i koristi dati DataFrame direktno.
  - `PortfolioBacktestConfig` dobija `warmup_candles` (default 1500,
    0 = reprodukuje stari v1 cold-start bug za poredjenje).
    `run_portfolio_backtest()` sad fetch-uje `warmup_candles + kline_limit`
    candle-a preko `load_extended_history()`, prosljedjuje ih u
    `backtest_symbol(..., history=df)`, izracunava `decision_start_time`
    kao vremensku granicu izmedju warm-up i "tradable" dijela, i
    prosljedjuje je u `_build_candidate_events(...,
    decision_start_time=...)` — candle-i prije granice i dalje pune stats
    accumulator (vidljivi za kalibraciju), ali se nikad ne emituju kao
    tradable CandidateEvent.
  - Testovi: `tests/test_portfolio_backtest.py::TestWarmup` (3 nova
    scenarija: suppression tacno na granici, dozvola poslije granice,
    `None` ponasa se identicno starom kodu) + integracioni test da
    `run_portfolio_backtest` zaista zove `load_extended_history` sa
    ispravnim `total_candles` i prosljedjuje `history=` u
    `backtest_symbol`. `tests/test_calibration.py::TestLoadExtendedHistory`
    (4 scenarija: single-batch, multi-batch pagination, rano zaustavljanje
    kad vendor nema vise istorije, indikatori racunati jednom nad punom
    serijom) — sve sintetsko, bez mreznih poziva.
  - **Ponovo pokrenuto na VPS-u sa fix-om** (isti 20 simbola: AAVE, ADA,
    AVAX, BCH, BNB, BTC, DOGE, ETH, INJ, LINK, NEAR, ONDO, OP, SOL, SUI,
    TAO, TRX, UNI, WLD, XRP; 4 timeframe-a; warmup=1500): net profit
    -41.35% (prvobitni, prije dodavanja `--report-json`), kasnije ponovo
    regenerisano sa istim simbolima/konfiguracijom radi dashboard
    izvjestaja -> -45.15%, win rate 38.1%, profit factor 0.67, expectancy
    -0.188R, max DD 46.98%, Sharpe -3.92. Manja razlika izmedju dva runa
    istom konfiguracijom je ocekivana varijacija (nove svijece su se
    pojavile izmedju runova, Binance podaci se mijenjaju). I dalje
    znacajno negativno nakon cold-start popravke — potvrdjuje da problem
    NIJE (samo) metodoloski artefakt. Breakdown po timeframe-u pokazuje da
    su 15m (29 trejdova, avg R -0.664) i 5m (2 trejdova) najgori, dok 1h
    (103 trejdova, avg R -0.081) gubi najmanje po trejdu — podupire ideju
    da se 5m/15m iskljuce iz live signala.
  - **Novi dashboard tab "🧪 Backtest"** (`webapp/static/index.html` +
    `app.js` + `style.css`, endpoint `GET /api/backtest/report`,
    `scanner/backtesting/portfolio_backtest.py::build_dashboard_report()`)
    — vizuelizuje kesiran, rucno-generisan JSON izvjestaj (metrike, equity
    kriva, breakdown po timeframe-u/smjeru/label-u) sa jasnim upozorenjem
    da je rijec o retrospektivnoj simulaciji, ne live rezultatu. Izvjestaj
    se NE regenerise automatski — pokrece se rucno preko `--report-json`
    CLI flag-a kad je potrebna svjezija analiza. Live na VPS-u, prikazuje
    gorenavedeni 20-simbola rezultat.
  - **Sljedece**: pun run nad svih ~274 simbola (sa resource-cap preko
    `systemd-run --scope -p CPUQuota=...` da ne ometa live scanner/
    dashboard na 2 vCPU VPS-u) jos nije pokrenut — odlozeno dok ne zavrsi
    Korak 3 (statisticka validacija).

---

## Korak 3 — Statisticka validacija (overfitting / multiple testing) — ZAVRSENO (v1)

- [x] Novi modul: `scanner/validation/statistical_validation.py`
- [x] **Deflated Sharpe Ratio (DSR)** — `deflated_sharpe_ratio()` +
      `expected_max_sharpe()`, Bailey & Lopez de Prado (2014). Sharpe
      korigovan za broj testiranih strategija/kombinacija (preko
      `n_trials` + varijanse Sharpe-a kroz te pokusaje) i non-normalnost
      prinosa (skew/kurtosis korekcija). Vraca `DeflatedSharpeResult` sa
      `probabilistic_sharpe_ratio` (vjerovatnoca da je STVARNI Sharpe > 0
      nakon korekcije) i `likely_skill` property (DSR > 0.95 prag).
      Namjerno bez scipy/numpy zavisnosti — koristi stdlib
      `statistics.NormalDist` (projekat nema numpy/scipy nigdje drugo).
- [x] **Probability of Backtest Overfitting (PBO)** —
      `probability_of_backtest_overfitting()`, Combinatorially Symmetric
      Cross-Validation (CSCV), Bailey/Borwein/Lopez de Prado/Zhu (2014).
      Dijeli matricu (periodi x pokusaji) na N blokova, testira svih
      C(N, N/2) train/test kombinacija, racuna frakciju split-ova gdje
      in-sample pobjednik zavrsi ispod OOS medijane.
- [x] **Multiple-testing correction** — `benjamini_hochberg()`, standardna
      BH step-up FDR procedura. (White's Reality Check / Hansen's SPA
      ODLOZENI — DSR + PBO + BH pokrivaju iste rizike preko jednostavnijeg,
      dependency-free pristupa; mogu se dodati kasnije ako se pokaze
      potreba za direktnim poredjenjem strategija jedne naspram druge.)
- [x] **Experiment registry** — `scanner/validation/experiment_registry.py`:
      samostalna SQLite baza (`scanner_data/experiment_registry.db`, NE
      dira `scanner.db`), `record_experiment()`/`fetch_experiments()`/
      `count_prior_trials()`. Svaki eksperiment (strategy_family/version,
      timeframe, label, horizon, sl_tp_config, train/validation/test
      period, n_trades, sharpe, net_profit, pun JSON rezultat, notes) se
      trajno evidentira — `count_prior_trials()` daje posten broj
      pokusaja za DSR korekciju (broj se NE smije rekonstruisati naknadno
      jer bi to omogucilo "cherry-picking" nakon vidjenih rezultata).
- [x] **Ziva integracija na "🧪 Backtest" dashboard tab** (odluka: dodano
      u POSTOJECI tab, ne novi, da sve ostane na "jednom vizuelnom
      mjestu" po eksplicitnom zahtjevu korisnika):
  - `portfolio_backtest.py::build_dashboard_report()` prima
    `n_trials`/`trial_sharpe_ratios` i ugradjuje
    `report["statistical_validation"]` (DSR racunat iz trejd-nivo
    `net_return_pct` liste).
  - CLI flagovi `--experiment-family`/`--experiment-registry-path`:
    kad su dati, automatski upisuju trial u registry I povlace SVE
    istorijske Sharpe vrijednosti te familije kao `trial_sharpe_ratios`
    (bez ovoga bi DSR benchmark ostao inertan/nula cak i sa n_trials>1).
  - Novi endpoint `GET /api/backtest/experiments` (webapp/main.py) +
    dvije nove sekcije na Backtest tabu: statisticka validacija (DSR
    kartice + obojeni badge + objasnjenje na bosanskom) i historija
    eksperimenata (tabela svih pokusaja, ne samo najboljeg).
  - **Ziva verifikacija na VPS-u (2026-11-04/05)**: rerun istog
    20-simbola/4-TF/1500-warmup configa dao je trial #2 (Sharpe -3.28,
    net profit -40.34%, 303 trejdova) — blago bolje od trial #1
    (Sharpe -3.92, -45.15%, 302 trejdova), ali i dalje jasno negativno.
    Sa n_trials=2 dashboard sada prikazuje REALAN (ne-inertan) DSR:
    observed_sharpe -0.127, deflated_sharpe_benchmark 0.166, PSR 0.0%,
    badge "VJEROVATNO SLUCAJNOST/OVERFITTING" (likely_skill: false).
    Potvrdjeno vizuelno u browseru na http://95.217.162.207:8000/ —
    oba tab-a (statisticka validacija + historija eksperimenata)
    renderuju se ispravno sa pravim brojevima.
  - **Zakljucak**: strategija u trenutnom obliku (20 simbola, 5m/15m/1h/4h,
    bez utezivanja) NEMA dokazanu statisticku prednost — PSR ~0% znaci
    da je i dalje vrlo moguce da je negativan rezultat cisto slucajan
    (iako je negativan, sto samo po sebi vec govori da strategija nije
    profitabilna; DSR ovdje prvenstveno sluzi kao okvir za BUDUCE
    pozitivne rezultate — da se ne prevarimo da je neki pozitivan Sharpe
    "pravi" kad je zapravo artefakt multiple-testing-a).
      pokusaja za DSR/BH korekcije kasnije, umjesto naknadnog prebrojavanja.
- [x] Testovi: `tests/test_statistical_validation.py` (18 testova: DSR
      edge-cases + monotonost sa brojem pokusaja, PBO na sintetski
      skillful/noise matricama, BH na klasicnim primjerima), `tests/
      test_experiment_registry.py` (8 testova: round-trip, prior-trial
      brojanje, cutoff filtriranje). Sve sintetsko/SQLite, bez mreznih
      poziva. Puna suita: 1548 passed, iste 8 postojecih nepovezanih
      failova, nula regresija.
- [x] **Prvi stvarni eksperiment upisan u registry na VPS-u**: 20-simbola/
      302-trejda run (vidi Korak 2) upisan kao trial #1 porodice
      `portfolio_backtest_20symbol_warmup1500` — od sada svaki sledeci
      backtest/validation run te porodice treba zvati `record_experiment()`
      PRIJE gledanja rezultata, da `count_prior_trials()` ostane posten.
- [x] **Sada primijenjeno na stvarni portfolio backtest rezultat** (azurirano
      nakon Phase 8 Korak 7 Test 2/3 runa) — `deflated_sharpe_ratio` se
      racuna za svaki `--report-json` izvjestaj kad je prosledjen
      `--experiment-family` (potvrdjeno na VPS-u, 5.11.2026.: n_trials=3,
      DSR benchmark 0.198 > observed Sharpe -0.1044, PSR 0.01% —
      nedvosmisleno "nije vjestina"). `probability_of_backtest_overfitting`
      i dalje NIJE ozicen (vidi Korak 7 Test 3 ispod) — CSCV trazi
      (periodi x pokusaji) matricu koju trenutni pipeline ne generise.

---

## Korak 4 — Point-in-time universe (survivorship bias fix) — ZAVRSENO (v1, backend+dashboard)

- [x] Novi modul: `scanner/backtesting/historical_universe.py`
- [x] Cuva historijski membership kroz rebalance prozore
      (`UniverseWindow(start, end, symbols)`) — koji su simboli bili u
      top-N na pocetku svakog prozora, rangirano po TRAILING prosjecnom
      dnevnom quote volumenu (koristi samo svijece koje su vec zatvorene
      prije tog datuma, nema look-ahead-a). `save_universe_cache()`/
      `load_universe_cache()` perzistiraju rezultat u JSON (ne mora se
      ponovo racunati pri svakom runu — gradnja je skupa, O(n_simbola x
      n_prozora) klines poziva, namjerno odvojena, ne pokrece se automatski).
- [x] Standalone CLI
      (`python -m scanner.backtesting.historical_universe --start ... --end
      ... --top-n ... --cache-path ...`) za offline gradnju kesa.
- [x] NE koristi danasnji volumen za izbor historijskog univerzuma —
      `_avg_daily_quote_volume()` eksplicitno trazi klines sa
      `end_time < as_of` za svaki rebalance prozor.
- [x] NE izbacuje lose performere retroaktivno — rangiranje je iskljucivo
      po volumenu/likvidnosti u tom trenutku, ne po naknadnom P&L-u.
- [x] Uvazava vrijeme kada je ugovor postao dostupan za trgovanje —
      `min_history_days` prag iskljucuje simbole koji jos nemaju dovoljno
      historije u datom prozoru (simbol koji je tek listovan prije 2 dana
      se ne rangira kao da postoji 30 dana).
- [x] Delistovani ugovori — DJELOMICNO, dokumentovano kao poznato
      ogranicenje: Binance `exchangeInfo` prikazuje samo trenutno aktivne
      ugovore, pa se potpuno delistovani simboli ne mogu automatski
      otkriti. `PointInTimeUniverseConfig.include_delisted` omogucava
      rucno dodavanje poznatih delistovanih simbola (best-effort, ne
      automatsko rjesenje). Ovo pokriva vazniji smjer problema za nas
      altcoin-tezak univerzum (simboli koji su POSTALI popularni kasnije),
      ne i suprotan (simboli koji su NESTALI).
- [x] **Integracija u `portfolio_backtest.py`**: novi `--point-in-time-
      universe-cache PATH` CLI flag. Kad je zadan, svaki kandidat-trejd
      cija simbol NIJE bio u point-in-time top-N univerzumu u trenutku
      ulaska (`entry_time`) se izbacuje PRIJE portfolio simulacije — trejd
      se broji samo ako bi taj simbol stvarno bio biran tada, ne samo zato
      sto je u danasnjoj `--symbols` listi. Rezultat/report sada nosi
      `point_in_time_filtered_count` i `report["point_in_time_universe"]`
      blok (`enabled`, `cache_path`, `trades_excluded`).
- [x] **Dashboard**: meta-linija na "🧪 Backtest" tabu sada pokazuje da li je
      point-in-time filter bio aktivan i koliko je trejdova iskljucio (ili
      jasnu napomenu da NIJE koristen, pa postojeci rezultati i dalje imaju
      survivorship bias).
- [x] 18 novih testova (`tests/test_historical_universe.py`) + 6
      novih/prosirenih testova (`tests/test_portfolio_backtest.py`) —
      ukupno 1567 prolazi (isti 8 pre-postojecih network-zavisnih
      neuspjeha, nula regresija). Ruff cist.
- **Sljedece (odlozeno)**: stvarno pokretanje CLI alata na VPS-u da se
  izgradi pravi kes za 2024-2025 period i rerun 20-simbola backtesta SA
  point-in-time filtrom ukljucenim, da se vidi koliko se trenutni
  (survivorship-biased) rezultat mijenja kad se simboli koji jos nisu bili
  "top 20" u pojedinim mjesecima izbace. Gradnja kesa je mrezno skupa
  (desetine/stotine simbola x nekoliko rebalance prozora) — razmisliti da
  li ograniciti `--candidate-symbols` na trenutnih ~274 praceni simbola
  umjesto punog exchangeInfo discovery-ja da se smanji broj poziva.

---

## Korak 5 — Execution realism (SL/TP i troskovi)

- [x] Rijesiti "koji nivo je prvi pogodjen" problem kad jedna svijeca ima i
      High iznad TP i Low ispod SL u OHLC podacima:
  - [x] Conservative model (SL prvi) — ostaje default, nepromijenjeno
        ponasanje za sve postojece pozivaoce.
  - [x] Optimistic model (TP prvi) — novi `tie_break="optimistic"`
        parametar u `_walk_forward_exit()`, eksplicitno opt-in preko
        `backtest_symbol(..., sl_tp_tie_break="optimistic")`.
  - [ ] Higher-resolution model (provjera niz(eg) timeframe-a) — **NIJE
        implementirano**, svjesno odlozeno: zahtijevalo bi dodatni (mnogo
        veci) skup kline poziva za svaku dvosmislenu svijecu. Dokumentovano
        u docstringu `_walk_forward_exit()`-a kao poznato ogranicenje, a ne
        preceno.
- [ ] Novi modul: `scanner/backtesting/execution_simulator.py` — gapovi,
      slippage na stop-market nalozima, partial fills, spread pri
      ulasku/izlasku, stop izvrsenje pri slaboj likvidnosti. **NIJE
      implementirano** ovaj put — najspekulativniji dio (najtezi za
      validaciju bez stvarnih fill podataka), namjerno odlozen u korist
      funding troska i cost-tier modela, koji TODO sam oznacava kao
      ozbiljnije.
- [x] Novi modul: `scanner/backtesting/funding_simulator.py` — funding
      payments kod futures pozicija drzanih preko funding perioda.
  - `FundingEvent` (timestamp, rate), `fetch_funding_history(symbol,
    limit=1000)` (wraps `client.get_funding_rate_history`; vraca `[]` na
    bilo koji `VendorError` ili neispravan red, nikad ne baca izuzetak —
    nedostatak funding podataka za jedan simbol ne smije prekinuti cijeli
    backtest), `funding_cost_pct(direction, entry_time, exit_time,
    events)` (sabira sve settlemente strogo unutar `(entry_time,
    exit_time]`, vec predznak-korigovano po smjeru — pozivalac uvijek samo
    oduzima rezultat).
  - Poznato ogranicenje (dokumentovano u modulu): Binance-ov
    `/fapi/v1/fundingRate` vraca samo zadnjih `limit` (max 1000, ~333 dana
    na 8h) settlementa kad nije zadan eksplicitan vremenski opseg — za
    period stariji od toga, `funding_cost_pct` tiho pada nazad na 0.0
    (tretirano kao "nepoznato, pretpostavi da nema troska", ne kao greska).
  - Ozicen u `scanner/calibration.py`: `LabeledCase` dobija novo polje
    `funding_cost_pct: float = 0.0` (default ne mijenja postojece
    pozivaoce); `backtest_symbol()` prima opcionalni `funding_events:
    list[FundingEvent] | None = None` i, kad je zadan, racuna stvarni
    exit-timestamp svakog slucaja (iz `df["open_time"]`, ne iz
    timeframe-minuta aritmetike — `calibration.py` ne smije importovati iz
    `signal_tracking.py` zbog kruzne zavisnosti) i puni novo polje;
    `summarize()` oduzima `funding_cost_pct` u oba (long/short) racuna.
  - Ozicen u `scanner/backtesting/portfolio_backtest.py`: novi
    `PortfolioBacktestConfig.include_funding_cost: bool = False` (default
    iskljuceno — zahtijeva mrezni poziv po simbolu; postojeci
    monkeypatch-ovani test lambda-i za `backtest_symbol` koji ne
    primaju `funding_events` kwarg ostaju neizmijenjeni kad je flag
    iskljucen, jer se `funding_events` prosljedjuje samo uslovno).
    `run_portfolio_backtest()` fetch-uje funding historiju **jednom po
    simbolu** (keširano/dijeljeno preko svih timeframe-ova tog simbola, ne
    po (simbol, timeframe) paru). `_net_return_pct()` sada takodje oduzima
    `case.funding_cost_pct`. Novi CLI flag `--include-funding-cost` i
    report/config transparentnost (`config.include_funding_cost` u
    `build_dashboard_report()`-u).
- [x] Zamijeniti fiksni round-trip trosak (0.10%) sa troskom po
      coin/timeframe kombinaciji umjesto jedne univerzalne vrijednosti
      (5m strategija moze izgubiti cijelu prednost na troskovima koju 4h
      strategija jos uvijek zadrzava).
  - Implementirano **po coinu** (ne po timeframe-u — flat fee se ne mijenja
    sa timeframe-om; manja prosjecna prednost kratkih timeframe-ova vec
    sama od sebe cini isti flat trosak relativno vecim udarcem, ne treba
    duplo kodirati): `scanner/calibration.py` dodaje
    `TIER1_MAJORS` (BTC/ETH/BNB, +0.0%), `TIER2_LARGE_CAP` (SOL/XRP/ADA/
    DOGE/AVAX/LINK/TRX/SUI, +0.02%), sve ostalo Tier3 (+0.05%), izlozeno
    kroz `round_trip_cost_pct_for(symbol, base_cost_pct=
    DEFAULT_ROUND_TRIP_COST_PCT)`. Eksplicitno dokumentovano kao
    "judgment call, ne mjerenje" — treba revidirati sa stvarnim order-book/
    fill podacima prije nego se ovome vise vjeruje od grube procjene smjera.
  - Ozicen u `portfolio_backtest.py` preko novog
    `PortfolioBacktestConfig.use_symbol_cost_tiers: bool = True` — za
    razliku od funding troska, ovo je **ukljuceno po defaultu** jer svi
    postojeci test simboli (BTCUSDT/ETHUSDT) spadaju u Tier1 (+0.0%
    surcharge), pa je uvodjenje bezbjedno za cijeli postojeci test suite,
    dok je u produkciji odmah aktivno za sve ostale simbole. Novi CLI flag
    `--no-symbol-cost-tiers` vraca staro (flat) ponasanje ako je ikad
    potrebno za poredjenje.
- **Testovi**: 26 novih (12 `tests/test_funding_simulator.py` — fetch/parse/
  VendorError/malformed-row otpornost, direction-aware cost racunanje,
  settlement-na-granici edge slucajevi; 9 u `tests/test_calibration.py` —
  `round_trip_cost_pct_for()` po tieru, `tie_break` parametar prosljedjen
  kroz `backtest_symbol()`, stvaran walk-forward test sa sintetickim OHLCV
  podacima kroz pravi indicator/trend pipeline koji provjerava da
  `funding_events=None` znaci nula trosak a stvarni eventi naplacuju
  direktne slucajeve; 5 u `tests/test_portfolio_backtest.py` — tier1 vs
  tier3 surcharge razlika, `use_symbol_cost_tiers=False` flat fallback,
  `include_funding_cost=False` ne poziva fetch niti prosljedjuje kwarg
  (kompatibilnost sa starim monkeypatch lambda-ima), `include_funding_cost=
  True` fetch-uje jednom po simbolu i dijeli rezultat preko timeframe-ova).
  Puni test suite nakon izmjena: 1593 passed (26 novih + isti baseline od
  8 unrelated pre-existing failures), ruff cist na svim dirnutim fajlovima.
- [x] **Dovrseno ozicenje `sl_tp_tie_break`-a** (prvobitno implementiran samo
      u `calibration.py`, nikad dostupan iz `portfolio_backtest.py`): novo
      `PortfolioBacktestConfig.sl_tp_tie_break: str = "conservative"` polje,
      uslovno prosljedjeno u oba poziva `backtest_symbol()` u glavnoj petlji
      (uslovno da ne pokvari stare monkeypatch test lambda-e bez tog kwarg-a),
      novi CLI flag `--sl-tp-tie-break {conservative,optimistic}`, dodan u
      `build_dashboard_report()`-ov config i u experiment-registry
      `sl_tp_config`. 4 nova testa, puni suite i dalje 1595 passed / isti 8
      baseline.

### Audit — "izgradjeno, ali nikad koristeno" provjera (nakon Koraka 5)

Nakon sto je gornja `sl_tp_tie_break` rupa otkrivena (implementirano u jednom
modulu, nikad stvarno dostupno iz portfolio backtesta/CLI/dashboarda),
zatrazena je sistematska provjera cijele kodne baze za slicne slucajeve.
Pronadjeno 5 nalaza, sva adresirana:

1. [x] **OOS (out-of-sample) validacija nedostupna** —
   `scanner.calibration.validate_out_of_sample()` / `train_test_split_cases()`
   / `OutOfSampleResult` su postojali i bili testirani izolovano, ali ih je
   jedino pozivao `agents_crypto/calibration_lookup.py` sa fiksnim
   defaultima — nikad dostizno iz portfolio backtesta. Ozicano:
   - Nova `PortfolioBacktestConfig` polja: `include_oos_validation: bool =
     True` (ukljuceno po defaultu — cista re-kalkulacija nad vec dohvacenim
     slucajevima, bez dodatnih mreznih poziva, za razliku od
     `include_funding_cost`), `oos_test_fraction`, `oos_min_test_n`.
   - Novo `PortfolioBacktestResult.oos_validation: list[OutOfSampleResult]`
     polje; `all_cases` se akumulira kroz sve (simbol, timeframe) parove u
     glavnoj petlji, pa se na kraju runa racuna jedan pooled OOS rezultat
     grupisan po `(label, horizon_candles)` — namjerno pooled preko svih
     simbola (pravi per-simbol breakdown bi trazio drugaciju strukturu
     podataka; ovaj pogled odgovara na "da li se label generalizuje kroz
     citav backtest univerzum", sto je razuman prvi korak).
   - OOS summary (`oos_labels_holding_up`, `oos_labels_total`) ubacen
     direktno u postojeci `metrics` dict — isti dict koji se vec salje u
     `record_experiment(result=...)`, pa se automatski perzistira u
     postojecu `result_json` kolonu experiment-registrija bez ikakve
     izmjene seme baze.
   - Novi CLI flagovi `--no-oos-validation`, `--oos-test-fraction`,
     `--oos-min-test-n`; dodano u `report.config` i experiment-registry
     `sl_tp_config`.
   - Novi `"oos_validation"` kljuc u dashboard reportu (lista dict-ova, po
     label/horizon), plus `report.config.point_in_time_universe_cache_path`
     (nalaz #3 ispod, trivijalan fix u istom prolazu).
   - 7 novih testova (`TestOosValidationWiring` + prosirenje
     `TestDashboardReport`), plus strukturni fix (`TestMetricsComputation`
     izdvojen kao posebna klasa nakon greske pri edit-u). 36 testova u
     `tests/test_portfolio_backtest.py` ukupno.
2. [x] **Frontend nikad nije prikazivao 9+ config kljuceva koje backend
   racuna** (cost tiers, funding cost, tie_break, OOS podesavanja, point-in-
   time universe putanja, …) — korisnik je mogao vidjeti REZULTATE ali ne i
   SA KOJIM TACNO podesavanjima su proizvedeni. Novo `<section id=
   "backtest-full-config">` + `renderBacktestFullConfig()` u
   `webapp/static/{index.html,app.js}` — puna transparentnost tabela svih
   config vrijednosti koje uticu na simulaciju, sa bojenim oznakama
   (ukljuceno/iskljuceno, conservative/optimistic, …).
3. [x] **`point_in_time_universe_cache_path` nedostajao iz `report.config`**
   — trivijalan jedan-red fix, dodan uz nalaz #1.
4. [x] **`scanner/calibrate.py` standalone CLI nije podrzavao funding trosak
   ni tie-break** (samo `--cost-pct` je postojao; `backtest_symbol()` vec
   podrzava `funding_events`/`sl_tp_tie_break` kwargs preko
   `portfolio_backtest.py`, ali ne i preko ove jednostavnije komandne
   linije). Dodani `--include-funding-cost` (fetch-uje
   `fetch_funding_history()` jednom po simbolu) i `--sl-tp-tie-break
   {conservative,optimistic}` flagovi, uslovno prosljedjeni u
   `backtest_symbol()` poziv. 3 nova testa
   (`tests/test_calibrate_cli.py`).
5. [x] **`scanner/backtesting/historical_universe.py` CLI nije pomenut ni u
   jednom runbook-u/automatizaciji** — pregledano i zakljuceno da je ovo
   NAMJERNO (dokumentovano u modulovom docstring-u: gradnja kesa je mrezno
   skupa, pa se "deliberately NOT run automatically"), ne previdjen
   nedostatak. Ostaje rucni/offline korak (vidi "Sljedece (odlozeno)" pod
   Korakom 4 iznad za konkretan plan rerun-a na VPS-u); ne treba cron/timer
   jer bi tiha periodicna re-gradnja bez ljudske provjere rezultata bila
   losija nego eksplicitan, namjeran korak kad god se zaista zeli novi kes.

Puni test suite nakon cijelog audita: ruff cist, `tests/test_portfolio_backtest.py`
36/36 i `tests/test_calibrate_cli.py` 3/3 lokalno; puni suite zakazan prije
commit-a.

### Post-audit nalaz: funding cost + point-in-time NIKAD stvarno kombinovani u jednom runu — ZAVRSENO

Audit iznad je provjerio da li je svaki pojedinacni dio ozicen i dostiziv.
Nije provjerio da li su svi dijelovi ikad **stvarno pokrenuti zajedno** u
jednom runu koji je i vidljiv na dashboardu i upisan u experiment registry.
Nisu bili: `--include-funding-cost` i `--point-in-time-universe-cache` su
dosad uvijek pokretani odvojeno (i to u pozadinske fajlove koje dashboard
`/api/backtest/report` endpoint uopste ne cita, jer cita samo jednu fiksnu
putanju), tako da nikad nije postojao jedan rezultat koji racuna sve
korekcije zajedno — klasican "izgradjeno, ali nikad iskoristeno zajedno"
slucaj.

- [x] Novi **kanonski** skript `tool/run_full_portfolio_backtest.sh` —
  jedina komanda koja kombinuje point-in-time universe filter + funding
  trosak + cost-tier model (vec default) + OOS validaciju (vec default) +
  conservative tie-break, pod NOVOM experiment-family (
  `portfolio_backtest_20symbol_warmup1500_full`), namjerno odvojenom od
  stare `portfolio_backtest_20symbol_warmup1500` familije da Deflated
  Sharpe Ratio-ova korekcija za multiple-testing ne meisa Sharpe-ove iz
  dvije nekompatibilne konfiguracije.
- [x] Pokrenut na VPS-u (throttled na 30% CPU-a), zavrsen i zapisan u
  dashboard + experiment registry. Stvaran (ne vise orphaned) rezultat
  kombinovane, najkorigovanije konfiguracije do sad:

  | | Stari uncorrected run | Kanonski kombinovani run |
  |---|---|---|
  | Net profit | -45.15% | **-23.34%** |
  | Win rate | 38.1% | 42.2% |
  | Sharpe | -3.92 | -2.34 |
  | Trejdova | 302 | 173 (point-in-time filter izbacio ostatak) |
  | OOS labela koje drze | nije racunato | **0 / 10** |

  I dalje negativan rezultat, ali realniji. **0/10 OOS labela koje drze**
  je najvazniji broj iz ovog runa — jak signal da trenutna pravila setup-a
  precjenjuju (overfituju na) in-sample period, ne nuzno da sama
  korekcija troskova/survivorship-a nije vrijedjela.
- [x] **Manuelni refresh dugme** na dashboardu ("🔄 Osvježi cijeli test" na
  🧪 Backtest tabu) — `POST /api/backtest/refresh` pokrece gornji kanonski
  skript kao detached, nice+cpulimit throttlovan (30%) background proces
  direktno sa servera; `GET /api/backtest/refresh/status` se pollu-je sa
  frontenda dok ne zavrsi. Prije ovoga je JEDINI nacin da se kanonski run
  ponovo pokrene bio rucni SSH na VPS.

---

## Korak 6 — Overlapping trades / zavisnost uzoraka (Wilson interval fix)

**ZAVRSENO (backtest/OOS strana)** — implementirano u
`scanner/validation/independence.py`, oznaceno `cluster_independent_events()`
(sweep-line spajanje preklapajucih [entry, entry + horizon*timeframe_min]
prozora u konektovane komponente, namjerno cross-symbol -- jedan BTC pump
koji upali isti label na vise altcoin-a istovremeno racuna se kao JEDAN
dogadjaj) i `summarize_with_independence()` (grupise po (label,
horizon_candles) kao i `calibration.summarize()`, klasterizuje unutar svake
grupe, racuna predstavnicki ishod klastera kao prosjecan net return njegovih
clanova, pa ponovo racuna Wilson interval nad brojem nezavisnih dogadjaja
umjesto sirovog broja slucajeva). Ozicano u
`scanner/backtesting/portfolio_backtest.py`:
`PortfolioBacktestConfig.include_independence_validation` (default True,
cist re-compute, bez mrezhih poziva -- ista kategorija kao
`include_oos_validation`), `PortfolioBacktestResult.independence_validation`,
`build_dashboard_report()`-ov `"independence_validation"` kljuc, CLI flag
`--no-independence-validation` / `--independence-min-n`. Prikazano na
dashboard-u (nova sekcija ispod OOS provjere, `renderIndependenceValidation()`
u `app.js`). Testovi: `tests/test_independence.py` (klasterizacija,
agregacija, Wilson korekcija) + `TestIndependenceValidationWiring` u
`tests/test_portfolio_backtest.py`.

- [x] Grupisati signale koji dijele isti market event (npr. isti BTC pomak,
      preklapajuci vremenski prozor) u jedan "independent event" --
      implementirano za backtest-side slucajeve (`LabeledCase`), ne za
      live `signal_outcomes` tabelu (vidi nize).
- [x] Racunati DVIJE statistike odvojeno:
  - Raw signal statistics (svaki signal se broji kao sada) --
    `win_rate_raw_pct`/`win_rate_raw_ci_low_pct`.
  - Independent event statistics (grupisani signali -> jedan brojac) --
    `win_rate_independent_pct`/`win_rate_independent_ci_low_pct`.
- [x] Wilson interval nad "independent event" statistikama je ono sto
      stvarno treba koristiti za kalibraciju, ne raw signal count --
      `holds_up_independent` koristi upravo tu donju granicu.

**Namjerno odlozeno (dokumentovano, ne skriveno):**
- [ ] U `signal_outcomes` (live tracking tabela, `scanner/storage.py`) dodati
      `signal_cluster_id`, `overlap_group_id`, `event_id`,
      `entry_timestamp`, `resolution_timestamp` -- online klasterizacija je
      teza (ne znas da ce se buduci signal preklopiti u trenutku kad se
      trenutni otvori) i live podaci se gomilaju mnogo sporije od vec
      dostupnih hiljada backtest slucajeva. Hitno pitanje (da li je vec
      izracunati OOS rezultat uopste pouzdan) rjesava se samo offline
      stranom, sto je upravo ono sto je ovaj korak uradio.
- [ ] Puni cross-label/cross-timeframe globalni event-clustering model
      (npr. "ovaj BTC crash je izazvao jedan klaster kroz 6 labela i 3
      timeframe-a istovremeno") -- veci, rizicniji refaktor; eksplicitni
      "stretch goal", ne nesto preskoceno u tisini. Trenutna implementacija
      klasterizuje SAMO unutar (label, horizon_candles) grupe.
- [ ] Reprezentativni ishod klastera je prosjecan net return njegovih
      clanova (simetrican, nepristrasan izbor) -- nije jedina braniva opcija
      (alternativa npr. "bilo koji win racuna se kao klaster-win"); ako se u
      praksi pokaze da ovo pravilo mijenja zakljucke na granici, vrijedi
      preispitati.

---

## Korak 7 — Profitability Validation Protocol (finalna provjera)

Redoslijed testova, svaki sljedeci ovisi o prethodnom:

1. **Test 1 — Historical validation** (obavezno): point-in-time universe,
   bez look-ahead, realni troskovi, SL/TP putanja, portfolio ogranicenja.
2. **Test 2 — Walk-forward validation** (obavezno): rolling
   train/validation/test, bez ponovnog koristenja finalnog test perioda za
   podesavanje, vise trzisnih rezima. **ZAVRSENO** (Phase 8 Korak 7 Test 2,
   `scanner/validation/walk_forward.py`): historija se dijeli hronoloski u
   N uzastopnih expanding-window foldova (podrazumijevano 4 folda, tj. 5
   jednakih chunk-ova po broju slucajeva — fold *k* trenira na svemu prije
   njega i testira na sljedecem chunk-u, nikad obrnuto). Label "drzi se"
   (`holds_up_walk_forward=True`) samo ako SVAKI ocijenjeni fold pojedinacno
   potvrdi Wilson 95% CI donju granicu iznad 50% — najstroziji moguci
   kriterij (ne vecina, nego svi), ista "nikad ne precjenjuj povjerenje"
   pristrasnost kao kod Koraka 6. Fold se racuna u verdikt samo ako ima
   barem `walk_forward_min_test_n` (podrazumijevano 20) test slucajeva;
   potreban je minimum `walk_forward_min_folds_evaluated` (podrazumijevano
   2) ocijenjenih foldova prije nego se bilo kakav verdikt uopste daje.
   Ozicen kroz cijeli isti sloj kao Korak 6 (config/result polja u
   `portfolio_backtest.py`, `build_dashboard_report()`, CLI flagovi
   `--no-walk-forward-validation`/`--walk-forward-n-folds`/
   `--walk-forward-min-test-n`/`--walk-forward-min-folds-evaluated`,
   `experiment_registry`'s `sl_tp_config`), nova dashboard sekcija
   "📆 Walk-forward provjera" (po foldu: N, win%, status). 14 novih testova
   u `tests/test_walk_forward.py` + 3 wiring testa u
   `tests/test_portfolio_backtest.py`. Namjerno ogranicen opseg (dokumentovano
   u modulu, ne skriveno): (a) koristi SIROVO N po foldu, isto kao Korak 6-ov
   OOS single-split, ne independence-korigovano N — ponovno pokretanje
   independence korekcije unutar vec manjeg foldovskog uzorka bi dodatno
   izgladnilo uzorak; ovo je komplementarna, ne zamjenska provjera za Korak
   6; (b) foldovi se grade na POOLED (cross-symbol, cross-label) hronoloskom
   redoslijedu, isto kao postojeci OOS split, tako da fold *k* znaci isti
   kalendarski period za svaki label u tabeli. **Potvrdjeno stvarnim VPS
   pokretanjem (5.11.2026., 20-simbola full univerzum run):**
   `walk_forward_labels_holding_up: 0 / walk_forward_labels_total: 10` —
   isti ishod kao vec poznati 0/10 OOS i 0/10 independence-corrected
   rezultati iz istog runa (Sharpe -1.94, net profit -18.69%, n_trades=160).
   Nijedan label se ne drzi konzistentno kroz sve hronoloske foldove, sto je
   ocekivano s obzirom na vec poznatu sliku — strategija trenutno nema
   dokazanu edge ni po jednoj od tri nezavisne provjere.
3. **Test 3 — Statistical validation** (obavezno): PBO, Deflated Sharpe,
   block bootstrap, multiple-testing correction. **DJELIMICNO ZAVRSENO.**
   - [x] **Deflated Sharpe Ratio** — vec od Koraka 3, sada i primijenjen na
     stvarni portfolio backtest rezultat preko `--experiment-family`
     (potvrdjeno na VPS-u: `observed_sharpe: -0.1044`,
     `deflated_sharpe_benchmark: 0.198`, `probabilistic_sharpe_ratio: 0.0001`,
     `likely_skill: false`, `n_trials: 3`) — DSR sam po sebi vec jasno kaze
     da ovaj rezultat NIJE dokaz vjestine, sto je konzistentno sa sve tri
     ostale provjere iznad.
   - [x] **Block bootstrap CI za prosjecan povrat po trejdu** (ovo je novi
     dio zavrsen ovog puta) — nova funkcija
     `scanner.validation.statistical_validation.block_bootstrap_confidence_interval`:
     umjesto obicnog bootstrap-a (preuzorkuje pojedinacne trejdove, presutno
     pretpostavlja nezavisnost), preuzorkuje UZASTOPNE BLOKOVE trejdova
     (podrazumijevana velicina bloka: kubni korijen broja trejdova,
     Politis-White heuristika) da sacuva kratkorocnu serijsku korelaciju
     izmedju susjednih trejdova — isti duh kao Korak 6-ova independence
     korekcija, ali primijenjen na kontinuirani povrat umjesto na win-rate
     brojanje dogadjaja. Ozicen kroz isti sloj kao ostale provjere: nova
     config polja `include_block_bootstrap`/`block_bootstrap_n_resamples`/
     `block_bootstrap_confidence_pct`, nova stavka
     `block_bootstrap_mean_return` unutar postojeceg
     `statistical_validation` izvjestaja (ne novi top-level kljuc — direktno
     je vezano za isti Sharpe/povrat racun), CLI flagovi
     `--no-block-bootstrap`/`--block-bootstrap-n-resamples`/
     `--block-bootstrap-confidence-pct`, nova dashboard sekcija "🎲 Block
     bootstrap CI" unutar postojece statisticke validacije. 9 novih testova

---

## Korak 7 — "Inverted" varijanta: cisti post-fix rezultati i zatvaranje porodice signala

Kontekst: nakon sto je otkriven i ispravljen bug u `--trades-json` CLI
argumentu (nije bio variant-suffixovan, pa je refresh "inverted" varijante
tiho prepisivao "original" varijantin trades fajl i obrnuto — commit
`a379d5f`) i nakon sto je otkriveno i dokumentovano da identicne konfiguracije
daju razlicite rezultate zavisno od trenutka pokretanja (live scanner 24/7
dodaje nove svijece ispod svakog runa — izmjereno ~9pp suma na "original"
varijanti kroz 5 identicnih runova prije ove ispravke), uradjene su DVIJE
stvari: (1) frozen-snapshot cache sloj (`scanner/backtesting/data_snapshot.py`,
`--end-timestamp`/`--kline-cache-dir` CLI flagovi, commit `43213cf`) i (2)
dva nova, cista post-fix runa obje varijante na VPS-u, 5.11.2026.

### Cisti post-fix rezultati (live fetch, bez end-timestamp zakljucavanja)

| Varijanta | Exp ID | N | Net % | Win % | Gross % (net+trosak) | OOS | Independence | Walk-forward |
|---|---|---|---|---|---|---|---|---|
| original | #13 | 38  | -13.47% | 28.9% | ~-9.7% (trosak 3.8pp)  | 0/10 | 0/10 | 0/10 |
| inverted | #12 | 225 | -22.11% | 53.3% | ~+0.39% (trosak 22.5pp) | 4/10 | 3/10 | 1/10 |
| inverted | #14 | 220 | -21.10% | 53.6% | ~+0.90% (trosak 22.0pp) | 4/10 | 3/10 | 1/10 |

**Zakljucak (potvrdjen od strane tri nezavisna vanjska misljenja + ova tri
runa):**

- **Original varijanta je mrtva** — 0/10 na sva tri nezavisna testa (OOS,
  independence, walk-forward), konzistentno kroz vise runova. Nema daljnje
  rasprave.
- **Inverted varijanta nema stvarnu edge.** Bruto (prije troskova) profit po
  cijelom runu je ~+0.4-0.9% preko ~8 mjeseci i ~220 trejdova — to je
  prakticki nula, ne "mala ali pozitivna" edge. Win rate od 53% ne dokazuje
  nista jer je SL/TP obrnut u odnosu na original (break-even win rate je
  vjerovatno iznad 50%, ne 50%).
- **OOS 4/10 nije nezavisna replikacija.** Dva runa (#12, #14) dijele ~95%
  istih svijeca (razlika je samo par svjezih dana) — isti osnovni podaci,
  ista arhitektura testa, pa isti rezultat nije iznenadjenje niti dodatni
  dokaz.
- **Walk-forward (1/10) je stroziji i vjerovatno mjerodavniji test od OOS-a
  (4/10)** — razlika izmedju njih je upravo ono sto se ocekuje od suma na
  10 nezavisnih pokusaja (labela) kad je stvarna edge ~0.

**Odluka: ova porodica signala (trend/anomaly labele na 5m-4h timeframe-ovima,
originalna i invertovana varijanta) se zatvara kao "nema dokazane edge nakon
troskova".** Sljedeci koraci istrazivanja idu na drugu porodicu hipoteza
(funding/OI ekstremi), ne na dalje "utezanje" ove.

### Pre-registracija: frozen-snapshot baseline run (zakljucano PRIJE pokretanja)

Cilj ovog runa nije da promijeni gornji zakljucak (vec je donesen), nego da
formalno, reproducibilno (sa `dataset_hash`) zabiljezi rezultat trenutne
specifikacije na podacima zakljucanim prije zadnjih 6 sedmica — bez ikakvog
tuninga/izmjene parametara u odnosu na konfiguraciju iznad.

- **End-timestamp (zakljucavanje):** `2026-09-24T12:43:40Z` (= trenutak
  pisanja ovog zapisa minus 6 sedmica). Sve svijece POSLIJE ovog trenutka
  ostaju NEDIRNUTE — ne gledaju se, ne koriste se za bilo kakvo podesavanje
  dok se formalno ne otvore tacno jednom, kao posebna provjera.
- **Specifikacija:** identicna trenutnoj produkcijskoj konfiguraciji
  (20 simbola, warmup 1500 svijeca, `round_trip_cost_pct=0.1`,
  `use_symbol_cost_tiers=True`, `include_funding_cost=True`,
  `sl_tp_tie_break=conservative`) — i original i inverted varijanta, bez
  izmjena.
- **Kriterij (vec zatvoren, odluka iznad ne zavisi od ovog runa):** ovaj run
  sluzi iskljucivo kao dokumentacioni/reproducibilni "pecat" na vec donesenu
  odluku. Ne koristi se kao test koji bi mogao preokrenuti zakljucak —
  taj test bi bio buduci jednokratni pogled na zakljucane posljednje
  sedmice, uz unaprijed definisan kriterij: neto expectancy po trejdu > 0 I
  donja granica block-bootstrap CI za prosjecan povrat po trejdu > 0.
     u `tests/test_statistical_validation.py` + 4 wiring testa u
     `tests/test_portfolio_backtest.py`. Pure re-computation na vec
     izracunatim trejdovima (bez mrezhnih poziva), pa je ukey po
     defaultu, kao i OOS/independence/walk-forward provjere. Jos NIJE
     pokrenuto na VPS-u sa stvarnim podacima (ceka sljedeci "Osvjezi cijeli
     test").
   - [ ] **PBO (Probability of Backtest Overfitting, CSCV)** —
     `probability_of_backtest_overfitting()` vec postoji i testiran je
     (Korak 3), ALI **nije i ne moze biti jednostavno ozicen u trenutni
     pipeline** bez dodatnog rada: CSCV trazi matricu (periodi x pokusaji)
     — tj. POVRATE VISE RAZLICITIH VARIJANTI STRATEGIJE preko ISTIH
     vremenskih perioda, da bi provjerio da li pobjednik in-sample ostaje
     pobjednik out-of-sample. Trenutni pipeline pokrece JEDAN backtest
     odjednom i belezi samo skalarni Sharpe po pokusaju u
     `experiment_registry` (ne per-period povratke po pokusaju), pa nema
     odakle sagraditi tu matricu bez novog mehanizma koji bi snimao
     per-period povratke za svaki zapisani eksperiment. Ostaje otvoren TODO
     — potreban je prvo dizajn te matrice (najvjerovatnije: per-period
     povrataka za svaki od N proslih zapisanih pokusaja iz iste familije,
     ako njihovi trejd logovi jos postoje) prije nego se moze stvarno
     ozicati, ne samo pozvati sa izmisljenim ulazom.
   - [ ] **Benjamini-Hochberg FDR korekcija** — `benjamini_hochberg()` vec
     postoji i testiran je, ali jos nije pozvana nad stvarnim skupom
     p-vrijednosti (jedna po (label, horizon) kombinaciji) iz backtest
     rezultata — odlozen dok se ne odluci sta tacno predstavlja "test" za
     svaki label (najvjerovatnije: p-vrijednost da je win rate > 50%,
     izvedena iz vec postojeceg Wilson CI racuna).
4. **Test 4 — Live paper trading** (obavezno): stvarni live signali, stvarni
   trzisni podaci, simulirani fillovi i troskovi, BEZ promjene strategije
   tokom evaluacionog perioda.
5. **Test 5 — Controlled live validation** (tek nakon 1-4): minimalna
   izlozenost, nezavisni risk engine, kontrola stvarnih fillova, kill switch.

Vazno: finalni OOS (out-of-sample) period mora biti zaista zakljucan — ako se
koristi za podesavanje strategije, prestaje biti OOS.

---

## Napomene / podsjetnici

- 1464 prolazna testa = dokaz kvaliteta implementacije, NIJE dokaz statisticke
  profitabilnosti. Ovo su dvije razlicite vrste validacije — ne mijesati ih.
- Cilj ovog rada nije da FazlaChart "izgleda pametnije", nego da pronadjemo
  sve nacine na koje sistem moze POGRESNO vjerovati da ima prednost — i
  uklonimo ih prije rizikovanja stvarnog kapitala.
- Redoslijed rada: Korak 0 (code review) -> Korak 1 (candidate archive) ->
  Korak 2 (portfolio backtest, najveci prioritet) -> ostalo paralelno/po
  potrebi.
