# Pre-registracija: "Cross-sectional momentum" hipoteza (market-neutral)

**Status:** PRE-REGISTROVANO, test jos NIJE pokrenut.
**Datum zapisa:** 2026-11-05 (prije bilo kakvog koda/runa za ovu hipotezu).
**Prethodnici:**
- Korak 7 (`PHASE8_PROFITABILITY_RESEARCH_TODO.md`) — trend/anomaly porodica
  (original + inverted, 5m-4h) zatvorena kao "nema dokazane edge nakon
  troskova".
- Korak 8 (isti fajl) — "funding-extreme reversion" porodica zatvorena kao
  "nema edge-a, ni bruto ni neto" (`n_independent=448 > n_required=371`,
  dakle odlucno, ne neodlucivo).

Ovo je **trial/porodica #3**, namjerno STRUKTURALNO drugacija od prve dvije:
obje prethodne su bile direkcione, po-simbolu opklade (izlozene BTC-beti).
Ova hipoteza je cross-sectional i market-neutral (long najjacih, short
najslabijih ISTOVREMENO) — beta prema trzistu bi trebala biti priblizno
nula, sto je nezavisna provjera da li problem lezi u konkretnom signalu
(prve dvije) ili u klasi pristupa (direkcione opklade na ovom univerzumu).

Svrha ovog fajla: zamrznuti hipotezu, specifikaciju, kriterij uspjeha i
"look once" pravilo PRIJE nego iko vidi ijedan rezultat backtesta za ovu
hipotezu. Bilo kakva izmjena ovog fajla NAKON prvog runa je sama po sebi
signal da je nesto krenulo po zlu (data snooping) i mora biti eksplicitno
obrazlozena, ne tiho uradjena.

## 1. Hipoteza

Simboli koji su u posljednjih 24h imali relativno najjaci povrat (u odnosu
na ostale simbole iz istog univerzuma, ISTOVREMENO) nastavljaju biti
relativno jaci u narednih 24h, a simboli sa relativno najslabijim povratom
nastavljaju biti relativno slabiji — tj. **momentum** (ne reversion).
Long najjacih / short najslabijih, ISTOVREMENO i u jednakim iznosima, ciji
je ocekivani neto povrat (nakon troskova i stvarnog turnover-a) u narednih
24h pozitivan.

**Eksplicitno zamrznut smjer: MOMENTUM.** Reversion (obrnuto — short
najjacih, long najslabijih) je ZASEBNA hipoteza koja bi se morala
pre-registrovati odvojeno, sa NOVIM holdoutom. Ako ovaj run pokaze
negativan povrat, to se NE smije citati kao "dakle treba okrenuti smjer" —
vidi sekciju 8 (poznata ogranicenja).

## 2. Specifikacija (zamrznuta — ne tunirati nakon prvog runa)

- **Rangiranje:** za svaki tacan trenutak rebalansa t, izracunaj
  24h-unazad % promjenu cijene (close[t] / close[t-24h] - 1) za SVAKI
  simbol koji je u tom trenutku dio point-in-time univerzuma (isti
  mehanizam kao `scanner/backtesting/historical_universe.py`, nema
  promjene). Bez look-ahead-a: rang u trenutku t koristi samo cijene do
  i ukljucujuci t.
- **Univerzum:** isti skup od 20 USDT-perp simbola kao prethodne dvije
  porodice. **Ne siriti univerzum nakon ovog zapisa, cak ni ako bi to
  popravilo rezultat** — uzak univerzum (20 simbola, visoko korelisanih
  preko BTC-bete) je poznato ogranicenje ove hipoteze (sekcija 8), ne
  parametar koji se podesava.
- **Minimalni broj simbola za rebalans:** ako je u trenutku t manje od
  `2*N=10` simbola uopste eligible (point-in-time univerzum), taj rebalans
  se PRESKACE (ne moze se formirati obje noge pune velicine) — ne
  smanjivati N za taj period.
- **Kosare:** long **top-5** (najveci 24h povrat), short **bottom-5**
  (najmanji 24h povrat) — N=5 je kvartil od 20, izabran UNAPRIJED kao
  prirodna, ne-tunirana granica (N=3 bi dao previse sumovite noge; bilo
  koje drugo N je NOVA hipoteza).
- **Tezinjenje:** equal-weight unutar svake noge (1/N = 20% kapitala te
  noge po simbolu). Obje noge jednako velike (dollar-neutral: 50%/50% ili
  100%/100% ako se koristi leveridz — izbor jedinice kapitala ne mijenja
  % povrat dok je simetrican).
- **Rebalans / izlaz:** dnevno (24h), vremenski fiksno — cijela kosara se
  zatvara i ponovo formira na sljedecem rebalansu. Nema SL/TP (isti duh
  kao funding hipoteza: izlaz je cisto vremenski, da se izbjegne dodatni
  stepen slobode prije nego se zna da li sirovi signal ima ikakvu edge).
  **Lookback (24h) i rebalans-frekvencija (24h) su zamrznuti.** 8h rebalans
  je razmatran i ODBACEN unaprijed: na 24h lookbacku bi dijelio ~2/3
  informacije sa prethodnim rangom (preklapajuci prozori), pa broj STVARNO
  nezavisnih opservacija ne bi rastao proporcionalno broju rebalansa.
- **Troskovi (turnover-bazirano, zamrznuto):** pozicija placa svoj
  round-trip trosak (`round_trip_cost_pct_for`, isti cost model kao
  prethodne dvije porodice) SAMO kad se stvarno MIJENJA izmedju dva
  uzastopna rebalansa:
  - Simbol koji OSTAJE u istoj nozi (long→long ili short→short) izmedju
    t-1 i t: **ne placa trosak tog dana** (pozicija se ne zatvara/otvara).
  - Simbol koji NAPUSTA nogu (bio u njoj na t-1, nije na t) ili koji TEK
    ULAZI u nogu (nije bio na t-1, jeste na t) ili koji FLIPUJE direktno iz
    long u short ili obrnuto: placa **puni round-trip trosak** za taj
    simbol, skaliran njegovom tezinom u nozi (1/N).
  - Izvjestava se i **turnover** (udio pozicija koje su se promijenile po
    rebalansu, 0-1) kao zaseban broj, ne samo agregatni trosak.
- **Funding trosak:** pozicije se drze ~24h (2-3 funding ciklusa), pa se
  stvarni funding (placen/primljen) racuna isto kao u funding hipotezi,
  preko `scanner/backtesting/funding_simulator.py` — ne graditi novi
  model.

## 3. Primarna metrika i dijagnostike

- **Primarna metrika:** neto povrat PO REBALANSU cijele kosare (jedan broj
  po danu, ne po simbolu), block-bootstrap CI preko
  `block_bootstrap_confidence_interval` sa `block_size > 1` (autokorelacija
  izmedju uzastopnih dana je ocekivana i mora biti ukljucena, ne
  pretpostavljena na nulu — koristiti isti default heuristiku kao ranije,
  ne rucno birati block_size nakon sto se vide rezultati).
- **Obavezno izvijestiti i:** bruto (prije troskova) povrat po rebalansu
  (ista block-bootstrap metodologija), prosjecan dnevni turnover (sekcija
  2), i **beta-provjeru**: Pearson korelacija izmedju dnevne serije povrata
  kosare i dnevne serije BTC povrata za isti period. Ako ta korelacija
  nije priblizno 0 (npr. |r| > 0.3), to se mora eksplicitno zapisati kao
  nalaz — "market-neutral" oznaka tada NE stoji empirijski, bez obzira na
  dizajn.
- **n_independent:** broj REBALANS-PERIODA (ne broj pojedinacnih
  simbol-pozicija) u train prozoru za koje je kosara stvarno formirana
  (iskljucujuci preskocene periode iz sekcije 2). Ovo je fundamentalno
  manji broj nego n_independent u prethodne dvije porodice (koje su brojale
  pojedinacne trejdove) — to je ocekivano i tacno, ne greska.

## 4. Zakljucani period / holdout

- **Train/validation period:** sve dostupne svijece do
  `end_timestamp_ms=1790253820000` (`2026-09-24T12:43:40Z`) — ISTA granica
  kao prethodne dvije porodice.
- **Holdout:** period NAKON te granice (zadnjih ~6 sedmica) ostaje
  NEDIRNUT — isti holdout kao prethodne dvije porodice, josh nije
  iskoristen ni za jednu od njih. Ova hipoteza NE otvara holdout osim ako
  prodje kriterij na train periodu (vidi Korak 8 zatvaranje: holdout se
  otvara samo za hipotezu koja je vec prosla trening-fazu).
- **Vazna napomena o dubini podataka:** za razliku od funding hipoteze
  (gdje je funding-istorija bila dublja od candle-istorije), ovdje je
  OGRANICAVAJUCI FAKTOR tacno onoliko candle-istorije koliko ima
  NAJKASNIJE LISTOVANI simbol u univerzumu od 20 (jer rangiranje zahtjeva
  SVIH eligible simbola ISTOVREMENO) — point-in-time univerzum filter ovo
  vec ispravno rjesava (noviji simboli jednostavno nisu eligible prije
  svog listinga, umjesto da se rupe u podacima tiho preskacu). Realno
  ocekivanje (zapisano UNAPRIJED): ovo daje priblizno **~150 nezavisnih
  dnevnih rebalans-perioda** na train periodu (manje od "punih" ~240 koje
  bi 8 mjeseci dnevnih rebalansa teoretski dala, zbog kombinacije
  point-in-time univerzum ogranicenja i granice od `end_timestamp_ms`).
  Ovo je PROCJENA, ne gornja granica — stvarni broj ce biti izmjeren i
  izvjesten iz samog runa, ne pretpostavljen.

## 5. Minimalni N (power racun, PRIJE prvog runa)

Formula: `n = (z_(a/2) + z_beta)^2 * sigma^2 / d^2`, dvostrani alpha=0.05
(`z=1.96`), power=80% (`z=0.84`) — isti standard kao prethodne dvije
porodice.

Nemamo jos stvarnu varijansu dnevnog povrata KOSARE (razlicito od
varijanse pojedinacnog trejda iz prethodnih porodica — diversifikacija
kroz 10 pozicija i beta-cancel bi trebali SMANJITI varijansu, ali
nepoznato je koliko), pa je donja tabela plan preko pretpostavljenog
raspona `sigma = 1.5% / 2.0% / 2.5%` po danu:

| Ciljani edge po rebalansu (nakon troskova) | N potrebno, sigma=1.5% | sigma=2.0% | sigma=2.5% |
|---|---|---|---|
| 0.10% | ~1 764 | ~3 136 | ~4 900 |
| 0.20% | ~441 | ~784 | ~1 225 |
| 0.30% | ~196 | ~349 | ~545 |
| 0.50% | ~71 | ~126 | ~196 |
| 0.75% | ~32 | ~56 | ~88 |
| 1.00% | ~18 | ~32 | ~49 |

**Prag "neodlucivo" (isto pravilo kao funding hipoteza, za dosljednost):**

> **neodlucivo** = `n_independent < n_required(sigma_izmjerena, d=0.3%)`,
> gdje se `sigma_izmjerena` racuna IZ OVOG runa (block-bootstrap/stdev na
> stvarnim dnevnim povratima kosare), ne iz tabele iznad. Ciljani edge je
> zamrznut na **d = 0.3%** po rebalansu (ista konvencija kao funding
> hipoteza — bira se konzervativnija/teza granica).

**Realno ocekivanje (zapisano UNAPRIJED, prije nego vidimo rezultat):** uz
procijenjenih ~150 nezavisnih rebalans-perioda (sekcija 4) i sigmu u
pretpostavljenom rasponu 1.5-2.5%, potrebno N za d=0.3% je 196-545 — u SVIM
slucajevima vise od ocekivanih ~150. **Najvjerovatniji ishod prvog runa je
zato "neodlucivo"**, isto kao kod funding hipoteze. Ovo NIJE razlog da se
N (velicina noge), lookback ili rebalans-frekvencija mijenjaju da bi se
"nabralo" vise opservacija — takva izmjena bi bila NOVA, zasebna hipoteza.
Realan put ka odlucivom rezultatu je ili (a) veci stvarni efekat nego
pretpostavljeno, (b) manja stvarna sigma kosare nego pretpostavljeno
(zbog diversifikacije/beta-cancel), ili (c) duzi period u buducnosti
(sljedeca sedmica-dvije uzivo skupljenih podataka, NE retroaktivno siriti
isti train prozor). Ako je ishod neodlucivo: hipoteza se NE vraca na istu
specifikaciju s "malo vise podataka" bez novog, posebno pre-registrovanog
perioda — pauza znaci cekanje da prirodno prodje dovoljno kalendarskog
vremena za nove podatke, ne ponovno gledanje istog train prozora.

## 6. Multiple-testing racunovodstvo

Ovo je **trial/porodica #3** u ukupnom projektu:
1. trend/anomaly (original + inverted) — zatvorena, 0/10 i 4/10→1/10
   (walk-forward), frozen baseline CI ne prolazi.
2. funding-extreme reversion — zatvorena, n_independent=448 > n_required=371,
   neto i bruto CI ne prolaze.
3. cross-sectional momentum (ova hipoteza).

Broj TESTIRANIH PORODICA (ne pojedinacnih runova unutar jedne porodice)
raste na 3 nakon ovog runa. Ako se ikad racuna formalna multiple-testing
korekcija (Bonferroni/Holm) preko svih dosad testiranih porodica, N=3 je
broj koji se koristi, ne broj pojedinacnih varijanti/labela unutar svake
porodice (te su vec racunate odvojeno unutar svoje porodice).

**Napomena o smjeru kao dodatnom stepenu slobode:** momentum je
pre-registrovan kao JEDINI smjer koji se testira ovim trial-om (sekcija
1). Ako bi se ikad testirao i reversion (suprotan smjer), to bi bio
cetvrti trial (#4), ne "dio istog #3", cak i da koristi identican
kod/parametre osim smjera.

## 7. Rok

Implementacija + jedan run u narednih nekoliko dana (bez fiksnog
kalendarskog roka), u skladu sa dogovorenim budzetom za ovaj treci i
(po trenutnoj odluci) posljednji pokusaj prije pauze/preusmjeravanja ako
ni ovaj ne prodje.

## 8. Poznata ogranicenja (zapisano UNAPRIJED)

- **Uzak univerzum.** 20 najlikvidnijih USDT-perp simbola su medjusobno
  visoko korelisani (dominira BTC-beta), pa razlika izmedju top i bottom
  kvartila moze biti mala u odnosu na sum — ovo je poznato ogranicenje
  ove hipoteze na OVOM univerzumu, ne razlog da se univerzum siri nakon
  sto se vide rezultati.
- **Kripto 24h momentum je cesto empirijski reversal, ne momentum**, prema
  siroj trzisnoj literaturi i iskustvu — ako ovaj run pokaze NEGATIVAN
  povrat za momentum smjer, to je LEGITIMAN I OCEKIVAN nalaz, ne razlog da
  se smjer naknadno okrene (vidi sekciju 1 i 6 — to bi bio nov trial).
- **Manji n_independent nego prethodne dvije porodice** (sekcija 3-5) je
  strukturalna karakteristika cross-sectional pristupa (jedna opservacija
  po danu za cijelu kosaru, ne po simbolu), ne greska u metodologiji.

## 9. Implementation notes (appended AFTER coding, BEFORE the one run --
design decisions not pinned above)

This section is appended by the implementer at the time
`scanner/backtesting/cross_sectional_momentum_backtest.py` was written, as
instructed by this doc's own rule (section 1-8 content above is frozen and
unedited; only this new section is new). Every item below is a decision
the pre-registered spec deliberately left open and asked to be recorded,
not silently made.

1. **Price data / candle timeframe.** 1h candles
   (`cached_load_extended_history(..., "1h", 8000, end_timestamp_ms, ...)`),
   same choice as the funding-extreme-reversion module, for consistency.
   Entry/24h-lookback/exit price = close of the most recent candle at or
   before the target timestamp (never a candle opening after it, to
   preserve no-look-ahead).
2. **Rebalance-timestamp grid.** Anchored to the EARLIEST candle
   timestamp across all requested symbols' fetched histories, then
   stepped forward by exactly 24h until the latest candle timestamp
   across all histories is reached. This does not hand-pick a particular
   hour-of-day; it simply follows whatever hour the earliest available
   candle happens to fall on, so no rebalance timestamp is invented ahead
   of real data. The last grid point cannot form a basket (no further
   grid point 24h later to price the exit against) and is therefore
   always excluded from `n_independent`.
3. **Basket-to-single-number capital convention.** Section 2's "izbor
   jedinice kapitala ne mijenja % povrat dok je simetrican" is read as:
   whichever capital unit you use, the WEIGHT of a symbol within its own
   leg (1/N) is unaffected. It does not, on its own, pin how the two legs
   combine into one basket-wide % return. The simplest, least-levered
   reading is used here: each leg represents HALF of total capital
   (50%/50%, not 100%/100% leveraged), so a symbol's overall weight is
   `0.5 / N` (10% of total capital for N=5). This keeps the reported
   basket return directly comparable in magnitude to a single-leg
   directional bet, rather than doubled by leverage.
4. **Turnover-cost accounting for pure exits.** A symbol that leaves a
   leg entirely (held at `t-1`, absent from the basket at `t`, not simply
   flipped into the other leg) still pays its full round-trip cost -- it
   is included in the cost-charged set even though it does not appear in
   the CURRENT basket. The `turnover` diagnostic itself, however, is
   measured only over TODAY's basket slots (`2*N`): the fraction of
   today's positions that are not a same-leg continuation from the prior
   rebalance. This is the simplest direct reading of "udio pozicija koje
   su se promijenile po rebalansu" (today's basket IS "the positions" for
   that rebalance) -- not otherwise pinned by section 2.
5. **First rebalance has no prior basket.** Every symbol in the very
   first formed basket is treated as a fresh entrant (full cost charged,
   turnover = 1.0) -- there is no rebalance before it to compare against.
6. **Funding-data-optional participation.** A symbol with usable candle
   data but no funding history is still included in ranking/pricing,
   contributing 0 funding cost for any period it is held, rather than
   being excluded from the universe entirely. The pre-registration doc
   does not say whether missing funding data should gate participation;
   this choice maximizes `n_independent` without ever inventing a funding
   number, since `funding_cost_pct` on an empty events list is defined as
   0 by that function's own existing contract.
7. **Measured sigma definition.** Sample standard deviation
   (`statistics.stdev`, i.e. `ddof=1`), not population standard deviation
   -- same convention as the funding hypothesis's implementation, for
   consistency, and the standard inferential choice when estimating a
   population sigma from a single finite observed sample.
8. **Block-bootstrap seed.** No explicit `seed` is passed to
   `block_bootstrap_confidence_interval` (mirrors both prior hypotheses'
   own call sites) -- the CI is therefore not bit-for-bit reproducible run
   over run, though with the library's default `n_resamples` the
   variation between runs is expected to be small. This affects only the
   CI's own estimation noise, not which rebalance periods/baskets were
   formed (that part is fully deterministic given a frozen
   `end_timestamp_ms`).
9. **Point-in-time universe filter.** Supported via an optional
   `--point-in-time-universe-cache` CLI flag, reusing
   `historical_universe.load_universe_cache`/`is_symbol_in_universe_at`
   verbatim (same mechanism as both prior hypotheses). As with the
   funding-extreme-reversion run, no such cache existed on disk at the
   time of this run (`scanner_data/` is gitignored and building one from
   scratch was out of scope for this task), so the filter ran disabled;
   the fallback eligibility check is simply "does the symbol have a
   priced candle at both `t-24h` and `t`", which already correctly
   excludes not-yet-listed symbols from ranking (point-in-time by
   construction via each symbol's own candle-history start), just without
   a formal top-N-by-volume cache.
10. **Cost-model base rate.** `DEFAULT_ROUND_TRIP_COST_PCT` plus
    `round_trip_cost_pct_for`'s existing per-symbol liquidity tiers, used
    as-is with no override -- same base rate both prior hypotheses use by
    default.

## 10. Rezultat prvog (i jedinog) runa i FINALNO zatvaranje

**Datum:** 2026-11-05. **Run:** jedan, train/validation period, na VPS-u
(`/opt/FazlaChart`), `end_timestamp_ms=1790253820000` (isti holdout-granica
kao obje prethodne zatvorene porodice). Holdout ostaje netaknut.
`dataset_hash: 1cdef092f543ed96` (identican hash-u iz funding-extreme
porodice -- ocekivano, isti 20 simbola/isti candle+funding dataset na
istom `end_timestamp_ms`, ne greska). 20/20 simbola koristeno, 0
preskoceno.

| Mjera | Vrijednost |
|---|---|
| n_rebalance_grid_points | 334 |
| n preskocenih (min-eligible/pricing) | 1 |
| **n_independent** (stvarno formiranih kosara) | **332** |
| Net (nakon troskova i funding-a) observed_pct | **-0.0793%** |
| Net 95% CI | **[-0.1778%, +0.0282%]** |
| Net edge_survives_lower_bound | **false** |
| Gross (prije troskova/funding-a) observed_pct | **+0.0760%** |
| Gross 95% CI | **[-0.0246%, +0.1872%]** |
| Gross edge_survives_lower_bound | **false** |
| Prosjecan dnevni turnover | **0.7271** (72.71%) |
| Beta-provjera (Pearson r, kosara vs BTC) | **-0.0012** (|r| < 0.3) |
| Izmjerena sigma (sample stdev, net) | **1.1838%** |
| Win rate | 42.47% |
| n_required (d=0.3%, izmjerena sigma) | 123 |
| n_independent vs n_required | 332 > 123 -> **NIJE neodlucivo** |

**Interpretacija:** Ovo NIJE "neodlucivo" -- 332 nezavisnih dnevnih
rebalans-perioda je znacajno vise nego sto je pre-registrovani power-racun
(sa IZMJERENOM sigmom od ovog runa) trazio (123), pa je test imao dovoljno
snage da donese odluku. Neto expectancy je blago negativan i CI prelazi
nulu s obje strane (gornja granica je pozitivna), ali DONJA granica je
ispod nule, pa `edge_survives_lower_bound=false` -- nema dokazane edge
nakon troskova i funding-a. Bruto (prije troskova/funding-a) povrat je
blago POZITIVAN, ali njegov CI takodjer prelazi nulu (donja granica je
ispod nule) -- dakle ni sirovi signal sam po sebi nema cvrst dokaz edge-a,
cak i prije ijednog troska. Prosjecan dnevni turnover od 72.71% je visok
(ocekivano za dnevni cross-sectional re-rank sa uskim univerzumom od 20
simbola) i objasnjava zasto je neto povrat niziji od bruto povrata --
troskovi (turnover-bazirani + funding) jedu priblizno
`0.076% - (-0.0793%) = ~0.155%` po rebalansu u prosjeku. Beta-provjera
POTVRDJUJE da je dizajn "market-neutral" ispao ispravan i empirijski: r =
-0.0012 je prakticno nula, duboko unutar |r| <= 0.3 praga -- ova hipoteza
NIJE izlozena BTC-beti (za razliku od prve dvije, direkcione porodice), sto
je bila i namjera dizajna iz uvoda ovog dokumenta.

**Odluka:** Hipoteza "cross-sectional momentum" (24h lookback, dnevni
rebalans, top-5/bottom-5 od 20 simbola, dollar-neutral) se FORMALNO
ZATVARA kao "nema edge-a, ni bruto ni neto, nakon sto se uzme u obzir
turnover-trosak i funding". Beta-provjera potvrdjuje da je metodoloski
cilj (market-neutral izlozenost) postignut -- problem dakle NIJE u
beta-izlozenosti nego u samom signalu (24h unazadni cross-sectional
momentum na ovom univerzumu nema mjerljivu edge). Po pravilu iz sekcije
6: ovaj ishod se NE smatra povodom da se smjer okrene na reversion (to bi
bio zaseban, cetvrti trial, #4, sa novim holdoutom) niti da se N, lookback
ili rebalans-frekvencija retuniraju na istim podacima. Holdout
(`end_timestamp_ms > 1790253820000`) ostaje netaknut. Trial/porodica #3
zatvorena.
