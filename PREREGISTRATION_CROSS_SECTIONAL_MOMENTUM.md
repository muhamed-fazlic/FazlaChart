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
