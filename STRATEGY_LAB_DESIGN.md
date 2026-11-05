# Strategy Lab — Design Doc

Cilj: kontinuirani, dvoslojni sistem koji **generiše nove kandidate** opisno,
bez trošenja statističkog "budžeta" (Layer 1), i **rijetko, odgovorno ih
potvrđuje** samo kada ima dovoljno snage (power) iz GARANTOVANO svježih
podataka (Layer 2). Ovo NE zamjenjuje Phase 8 disciplinu
(`PHASE8_PROFITABILITY_RESEARCH_TODO.md`) — nadograđuje je istom
rigoroznošću (pre-registracija, block-bootstrap CI, frozen holdout,
multiple-testing računovodstvo, neutralan ton za "nema edge-a").

Status: implementirano, lokalno verifikovano end-to-end (prazno i
popunjeno stanje), testirano (45 novih testova), deployovano.

---

## Zašto dva sloja

Jedan plošan "probaj sve kombinacije i vidi šta prolazi CI" bi:

1. eksplodirao broj statističkih testova (multiple-testing problem) bez
   ikakvog računovodstva,
2. trošio frozen/rolling holdout podatke na kombinacije koje nikad nisu
   bile obećavajuće,
3. brisao liniju između "gdje da tražimo" i "šta zaista radi".

Zato:

- **Layer 1 (miner)** je neograničen po broju pokretanja, ali **ograničen po
  broju varijanti** (pre-deklarisana mreža, tačan broj se loguje/printa pri
  svakom pokretanju) i **nikad ne računa CI niti izriče pass/fail** — izlaz
  je čisto opisni rang, "trag za buduću potvrdu, ne nalaz".
- **Layer 2 (confirmation gate)** je rijedak po dizajnu: pokreće se samo kad
  je statistička snaga (power) stvarno dostignuta na GARANTOVANO svježim
  (nikad ranije viđenim) podacima, i tada troši TAČNO jedan prozor podataka
  po potvrdi — jedino mjesto u cijelom Strategy Lab-u gdje postoji pravi
  pass/fail jezik (`edge_survives_lower_bound`).

---

## Layer 1 — kontinuirani miner

Modul: `scanner/backtesting/strategy_miner.py`.

### Pre-deklarisana mreža (tačan broj kombinacija)

3 zamrznuta template signala (ista 3 porodice iz Phase 8, frozen specs):

| Template | Izvor | Varijante (pre-deklarisano, malo) |
|---|---|---|
| `trend_anomaly` | `portfolio_backtest.py` + `strategy_variants.py` | 2 (`original`, `inverted`) |
| `funding_extreme_reversion` | `funding_extreme_backtest.py` | 3 (male varijacije percentil-praga/hold-hours oko zamrznutih defaulta) |
| `cross_sectional_momentum` | `cross_sectional_momentum_backtest.py` | 3 (male varijacije lookback-a/basket-size oko zamrznutih defaulta) |

Ukupno **8 template-varijanti** × broj tercila likvidnosti (3 u punom
170-simbol Korak-10 univerzumu, ili 1 "unsliced" fallback tercil ako
ranked-universe keš još ne postoji za dati `as_of_ts`):

```
total_combinations = len(template_variants) * len(tercile_symbol_sets)
```

Ova vrijednost se **loguje i printa pri svakom pokretanju** `run_mining_cycle()`
— ista disciplina kao Korak 10 ("tačan broj se mora znati unaprijed"). U punom
3-tercilnom slučaju to je **8 × 3 = 24 kombinacije** po ciklusu; lokalna
end-to-end verifikacija je (dokumentovano, namjerno) koristila fallback
1-tercilni slučaj = 8 kombinacija, pošto 170-simbol ranked-universe keš još
nije popunjen ni od strane ovog ni od strane paralelnog Korak-10 agenta u
trenutku ove implementacije.

Pre-deklarisane varijacije parametara se primjenjuju na zamrznute module
BEZ izmjene tih fajlova, kroz `_patched_module_constants()` kontekst-menadžer
koji privremeno (i uvijek sigurno, `finally`-restore) mijenja modul-nivo
konstante koje zamrznuti moduli čitaju kao globalne u trenutku poziva.

### Šta se čuva po kombinaciji

`MiningCandidate`: `template`, `tercile`, `variant_label`, `params`,
`n_independent`, `gross_observed_pct`, `net_observed_pct`,
`stability_score` (gruba heuristika: dijeli hronološki niz događaja na 3
dijela, provjerava slaganje predznaka sa cjelinom; `None` ako je <6
događaja), `composite_score` (`net × stability`, rang heuristika — NIJE
statistička tvrdnja), i sirovi `report` dict.

**Nikad se ne čuva CI niti pass/fail verdikt u Layer 1** — to je
arhitekturna odluka, ne propust.

### Skladište

`scanner/strategy_lab_storage.py` — SQLite `scanner_data/strategy_lab.db`,
ista WAL/`init_db`/`connect_for_reads` konvencija kao `scanner/storage.py`.
Tabele: `mining_candidates`, `weekly_snapshots`, `confirmations`,
`rolling_holdout_state`.

### Keširanje podataka (bez nepotrebnih API poziva)

Layer 1 koristi iste zamrznute backtest module koji već imaju svoj
on-disk candle/funding keš (vidi `data_snapshot.py` i
`historical_universe.py`'s ranked-universe keš) — mining ciklusi NE
re-fetch-uju podatke koji su već na disku.

---

## Sedmični snapshot (opisno, bezbjedno, bez trošenja holdout-a)

Funkcija: `run_weekly_snapshot()` u `scanner/backtesting/strategy_lab_runner.py`.

Uzima trenutno #1-rangiranog Layer-1 kandidata i ponovo ga pušta kroz PUNI
zamrznuti backtest engine (`strategy_miner.evaluate_variant()` — nema
reimplementacije backtest/CI logike) na train-only podacima, da dobije pune
statistike uključujući block-bootstrap CI. Eksplicitno obilježeno kao
eksploratorno — kandidat je izabran gledanjem u Layer 1 rang-listu
(selection bias), pa čak i puna CI ovdje NIJE validan konfirmatorni test.

Telegram izvještaj (`format_strategy_lab_weekly_report()` u
`scanner/telegram_alerts.py`, reuses `send_telegram_message`/
`TelegramAlertConfig`) uvijek sadrži "⚠️ Eksploratorno, nije potvrđeno"
napomenu.

---

## Layer 2 — potvrda ograničena snagom (power), ne kalendarom

Funkcija: `run_layer2_check()` u `scanner/backtesting/strategy_lab_runner.py`.

### Rolling holdout (odvojen od zaključanog 6-sedmičnog holdout-a)

Modul: `scanner/backtesting/rolling_holdout.py`.

- `initial_boundary_ms(shared_holdout_end_ms, now_ms)` = `max(shared_holdout_end_ms, now_ms)`
  — nikad prije kraja postojećeg zaključanog holdout-a (taj je rezervisan,
  ne "svjež"), nikad u budućnosti relativno na "sada".
- `fresh_window_available_ms(state, now_ms)` = `max(0, now_ms - boundary)`.
- `advance_boundary(state, consumed_until_ms)` — baca `ValueError` na bilo
  koji ne-forward pomak. Ovo je JEDINA garancija da se nijedan prozor
  podataka nikad ne koristi dva puta u potvrdi. Testirano direktno
  (`tests/test_rolling_holdout.py`).

**Odluka (ne otvoreno pitanje):** kad se potvrda okine, `run_layer2_check()`
poziva `advance_boundary(state, consumed_until_ms=now_ms)` — tj. potroši
CIJELI do tad akumulirani svježi prozor, ne samo statistički minimum
(`n_required`) potreban za taj konkretan test. Alternativa ("banker" samo
minimum, ostatak sačuvaj za sljedeću, bržu potvrdu) je razmatrana i
odbačena: granica između "minimalno potrebnih" i "viška" dana bi zavisila
od toga koliko dana walk-forward/warmup logika backtest enginea stvarno
dotakne za taj uzorak, što nije čisto odvojivo od samog broja nezavisnih
proba (`n_independent`) bez rizika da se neki dan tretira kao nikad
"pogledan" iako ga je neki raniji poziv već djelimično koristio za
warmup/kontekst. Trošenje cijelog prozora je konzervativnije i garantuje
"nijedan bajt podataka nikad dva puta" bez te dvosmislenosti, po cijenu
rjeđih potvrda nego što bi teoretski bilo moguće. Ovo je svjestan
trade-off (jednostavnost i sigurnost ispred brzine), ne nedovršen dio.

### Power-gated odluka (ne fiksni kalendar)

Modul: `scanner/backtesting/power.py` — `required_sample_size(sigma_pct, ...)`
ekstrahovana formula (standardni two-sided z-test sample-size za srednju
vrijednost):

```
n_required = ceil( (z_alpha/2 + z_beta)^2 * sigma^2 / d^2 )
```

sa defaultima `z_alpha=1.96` (α=0.05), `z_beta=0.84` (power=0.80),
`d=0.3%` — identični konstantama već zamrznutim (odvojeno, inline) u
`funding_extreme_backtest.py` i `cross_sectional_momentum_backtest.py`.
Ova formula je ranije postojala samo kao inline kod u dva mjesta (prosa u
markdown dokumentaciji dodatno objašnjava isto) — sada postoji JEDNO
testirano, uvozivo mjesto (`tests/test_power.py`, 13 testova, direktno
provjereno naspram brojeva iz obje zatvorene porodice: σ=2.0614 → n=371,
σ=1.1838 → n=123).

Svake sedmice, `run_layer2_check()`:

1. Projektuje koliko je nezavisnih događaja kandidat vjerovatno akumulirao u
   svježem prozoru, ekstrapolirajući linearno njegovu VLASTITU
   trening-periodnu stopu događaja (`events_per_day`).
2. Ako je procijenjeni broj < `n_required`: NE troši ništa, samo loguje i
   (sedmično) izvještava "čekam, treba još ~N dana" (`power.estimate_days_needed()`).
3. Ako je dostignuto: pokreće TAČNO JEDAN pravi confirmatory backtest kroz
   isti zamrznuti engine na prozoru `[boundary_ms, now_ms]`, računa isti
   block-bootstrap CI, bilježi `edge_survives_lower_bound` (net CI donja
   granica > 0), pomjera `boundary_ms` na `now_ms` (trošeći prozor zauvijek),
   i šalje Telegram izvještaj
   (`format_strategy_lab_confirmation_report()`) — **jedino mjesto u
   cijelom Strategy Lab-u sa pravim pass/fail jezikom.**

### "Trenutno najbolja potvrđena strategija"

`strategy_lab_storage.insert_confirmation()`: ako nova potvrda prolazi
(`edge_survives_lower_bound=True`), prvo briše `is_current_best` zastavicu
sa svih prethodnih redova, pa označava novi red kao trenutno najbolji.
Prošli prolasci OSTAJU u istoriji (nikad se ne brišu) — dashboard prikazuje
i istoriju i trenutno najbolje.

Ako potvrda NE prolazi: zabilježena kao normalno, neutralno "nema edge-a"
zatvaranje, isto kao prethodne 3 porodice — nikad se ne tretira kao greška.

---

## Multiple-testing računovodstvo

`scanner_data/research_registry.json`'s `multiple_testing_counter` dobija
DVA nova, aditivna polja (ne diraju postojeća 3-porodice + Korak-10 polja):

- `strategy_lab_layer1_combinations_explored` — periodični snapshot broja
  Layer-1 kombinacija ikad evaluiranih (živi, detaljni brojevi su u
  `scanner_data/strategy_lab.db`, izloženi kroz
  `GET /api/strategy-lab/overview`).
- `strategy_lab_layer2_confirmations_run` — isto, za Layer 2 pokušaje.

Jedan konzistentan izvor istine: `research_registry.json` nosi periodični
sažetak za "Istraživanje" tab; `strategy_lab.db` nosi pune, žive detalje za
"Strategy Lab" tab — oba su generička i aditivna, nijedno ne duplira
drugo.

---

## Scheduler (ne tight `while True` petlja bez strukture)

Modul: `scanner/strategy_lab_scheduler.py`, pokreće se kao
`python -m scanner.strategy_lab_scheduler` (odvojen systemd unit od
glavnog Market Scanner-a). Mirrors `market_scanner.py`'s `run_once`/
`run_forever` konvenciju: jedan uvijek-živi proces, log-and-continue na
grešku po ciklusu.

- Layer 1 mining: svakih nekoliko sati (default 6h, `--mining-interval`).
- Sedmični snapshot + Layer 2 provjera: jednom sedmično (default 7 dana,
  `--weekly-interval`), unutar iste petlje (provjerava da li je prošla
  sedmica od zadnjeg sedmičnog pokretanja).

`--once` flag za jedan ciklus + jedan sedmični check (testiranje/CI).

---

## Dashboard tab: "⚒️ Strategy Lab"

`GET /api/strategy-lab/overview` (webapp/main.py) — read-only, graceful
`{"available": false}` prije nego što je ijedan mining ciklus ikad
pokrenut (`strategy_lab_storage.database_exists()`), isti obrazac kao
postojeći `GET /api/research/registry`.

Sekcije (top-down, svaka sa jednolinijskim objašnjenjem šta JESTE i šta
NIJE):

1. **Trenutno najbolja potvrđena strategija** — ili iskreno prazno stanje
   ("još nijedna strategija nije prošla Layer 2 potvrdu") ako ništa nije
   prošlo.
2. **Layer 1 rang-lista** (top 10) — obilježeno "eksploratorno — trag za
   buduću potvrdu, ne nalaz".
3. **Istorija sedmičnih snapshotova** — tabela, pokazuje stabilnost/
   nestabilnost #1 kandidata kroz sedmice.
4. **Istorija Layer 2 potvrda** — svaki pokušaj: datum, kandidat, svježi
   prozor, CI, verdikt, Telegram indikator.
5. **Rolling holdout / multiple-testing status** — koliko je potrošeno,
   koliko pokušaja ukupno.

Vizuelno reuse-uje postojeće CSS klase (`.research-badge-confirmed`,
`.research-badge-noedge`, `.bucket-breakdown`, `.backtest-disclaimer`,
`.backtest-meta`) — izgleda nativno uz postojeći "Istraživanje" tab, bez
paralelnog vizuelnog jezika.

---

## Eksplicitno van opsega

**Nijedna potvrđena strategija se NIKAD automatski ne pušta na stvarni
novac**, bez obzira na verdikt Layer 2 potvrde. Prolazak kroz Layer 2 je
kandidat za DALJI LJUDSKI PREGLED (execution layer, forward/paper test) —
ništa više. Ovo je tvrd, namjeran ustupak sigurnosti nauštrb automatizacije,
i mora ostati tačno ovako dok god neko eksplicitno, svjesno ne odluči
drugačije kroz odvojen, posebno pregledan execution-layer projekat.

---

## Testovi

- `tests/test_power.py` (13) — formula, cross-checked naspram zamrznutih
  brojeva obje zatvorene porodice.
- `tests/test_rolling_holdout.py` (8) — forward-only garancija.
- `tests/test_strategy_miner.py` (17) — broj kombinacija, rangiranje,
  stability score, izolacija grešaka po kombinaciji.
- `tests/test_strategy_lab_runner.py` (7) — sedmični snapshot (prazno i
  popunjeno stanje, Telegram slanje), Layer 2 čekanje vs. potvrda,
  garancija da se potrošen prozor ne koristi dva puta.

Ukupno 45 novih testova, svi network-free (monkeypatch na seam-ovima, ne
na HTTP nivou).
