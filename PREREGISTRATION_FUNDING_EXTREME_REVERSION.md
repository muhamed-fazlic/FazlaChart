# Pre-registracija: "Funding-extreme reversion" hipoteza

**Status:** PRE-REGISTROVANO, test jos NIJE pokrenut.
**Datum zapisa:** 2026-11-05 (prije bilo kakvog koda/runa za ovu hipotezu).
**Prethodnik:** Korak 7 u `PHASE8_PROFITABILITY_RESEARCH_TODO.md` — trend/
anomaly signal family (original + inverted, 5m-4h) formalno zatvorena kao
"nema dokazane edge nakon troskova" (frozen baseline CI `[-0.73%, +0.34%]`,
`edge_survives_lower_bound: false`). Ovaj dokument je spec za SLJEDECU,
nezavisnu hipotezu — odvojen trial-brojac, odvojen holdout.

Svrha ovog fajla: zamrznuti hipotezu, specifikaciju, kriterij uspjeha i
"look once" pravilo PRIJE nego iko (covjek ili AI) vidi ijedan rezultat
backtesta za ovu hipotezu. Bilo kakva izmjena ovog fajla NAKON prvog runa
je sama po sebi signal da je nesto krenulo po zlu (data snooping) i mora
biti eksplicitno obrazlozena, ne tiho uradjena.

## 1. Hipoteza

Kad je funding rate na USDT-perpetualu u ekstremu (gornji ili donji 5% svoje
HISTORIJSKE (prosle, rolling) distribucije za taj konkretni simbol), trziste
je prenatrpano na jednoj strani (previse long ili previse short pozicija
placa funding), pa je ocekivani povrat u SUPROTNOM smjeru u narednih 8-24h
pozitivan nakon troskova.

## 2. Specifikacija (zamrznuta — ne tunirati nakon prvog runa)

- **Ulazni signal:** rolling percentil funding rate-a za (symbol, trenutak)
  racunat ISKLJUCIVO iz proslih podataka (bez look-ahead-a — percentil u
  trenutku t koristi samo funding istoriju striktno prije t).
  - percentil > 95 → otvori **short**
  - percentil < 5 → otvori **long**
- **Izlaz:** fiksno vrijeme drzanja = **8h** (jedan funding ciklus na
  Binance Futures). BEZ SL/TP tuniranja u ovoj, prvoj verziji — izlaz je
  cisto vremenski, da se izbjegne dodatni stepen slobode (SL/TP prag) prije
  nego se uopste zna da li sirovi smjer signala ima ikakvu edge.
- **Univerzum:** isti skup od 20 simbola kao zatvorena porodica, sa
  point-in-time universe filterom (isti mehanizam kao
  `scanner/backtesting/historical_universe.py`, nema promjene).
- **Dedup:** isti cluster/correlated-event pristup kao zatvorena porodica
  (npr. isti BTC-wide funding-ekstrem dogadjaj na vise simbola = jedan
  predstavnik trejda, ne N odvojenih).
- **Troskovi:** isti cost model (fee + spread/slippage preko
  `round_trip_cost_pct`/`use_symbol_cost_tiers`) + STVARNI funding koji se
  placa ili prima tokom perioda drzanja (vec postoji
  `scanner/backtesting/funding_simulator.py` — koristiti, ne graditi
  ispocetka).
- **Parametri koji se NE tuniraju nakon prvog runa:** prag percentila (95/5)
  i vrijeme drzanja (8h). Svaka alternativa (npr. percentil 90/10, drzanje
  4h ili 24h) je ZASEBNA hipoteza i broji se kao zaseban pokusaj (trial) u
  multiple-testing racunovodstvu — ne "jos jedan pokusaj iste hipoteze".

## 3. Zakljucani period / holdout

- **Train/validation period:** sve dostupne svijece do
  `end_timestamp_ms = 1790253820000` (`2026-09-24T12:43:40Z`) — ISTI
  zakljucani end-timestamp kao vec zatvorena trend/anomaly porodica, radi
  konzistentnosti i da se iskoristi vec postojeci kline cache
  (`scanner_data/kline_cache/`).
- **Holdout (zadnjih ~6 sedmica, zakljucan):** sve posle gornjeg
  end-timestampa. Ne gleda se, ne koristi se za podesavanje bilo cega, dok
  se formalno ne otvori TACNO JEDNOM, nakon sto specifikacija iznad vec
  prodje kriterij na train/validation periodu.
- **Pravilo "gledaj jednom":** holdout se otvara najvise jednom za ovu
  hipotezu. Ako se otvori i rezultat ne prodje kriterij, hipoteza se
  zatvara — nema drugog pokusaja na istom holdoutu niti "poravke" parametara
  pa ponovnog gledanja.

## 4. Kriterij uspjeha (definisan PRIJE prvog runa)

Na train/validation periodu:
1. **Neto expectancy po trejdu > 0** (sirovi, block-bootstrap
   `observed_pct`, ne equity-curve kompaundovana mjera — vidi upozorenje iz
   Koraka 7 o tome da se ove dvije mjere mogu razilaziti po predznaku kad je
   efekat mali).
2. **Donja granica 95% block-bootstrap CI za prosjecan povrat po trejdu > 0**
   (`edge_survives_lower_bound: true`).

Tek ako OBA uslova prodju na train/validation periodu, otvara se holdout
(tacno jednom, isti kriterij se primjenjuje na njega). Prolaz na holdoutu
vodi u **forward paper test** (live papir-trading na zamrznutoj
specifikaciji), NE direktno u trgovanje pravim novcem.

## 5. Minimalni N (power racun, PRIJE prvog runa)

Nemamo jos funding-specificnu varijansu povrata po trejdu (nijedan trejd ove
hipoteze jos ne postoji), pa je donja tabela provizorna procjena koja
pozajmljuje empirijsku standardnu devijaciju iz UPRAVO ZATVORENE trend/
anomaly porodice (nazad izracunato iz njenog block-bootstrap CI-a na
frozen-baseline runu: `n_independent=103`, CI pola-sirine `0.5358%` →
implicirano `sigma ~= 2.77%` po trejdu). Ovo ce se ZAMIJENITI stvarnom
izmjerenom varijansom cim se prvi funding-backtest run zavrsi — ovaj broj je
samo plan koliko podataka treba prikupiti, ne konacan test.

Formula: `n = (z_(a/2) + z_beta)^2 * sigma^2 / d^2`, dvostrani alpha=0.05
(`z=1.96`), power=80% (`z=0.84`).

| Ciljani edge po trejdu (nakon troskova) | Potreban N (nezavisnih dogadjaja) |
|---|---|
| 0.10% | ~6 033 |
| 0.20% | ~1 509 |
| 0.30% | ~671 |
| 0.50% | ~242 |
| 0.75% | ~108 |
| 1.00% | ~61 |

**Prakticna posljedica:** ako prvi run (na train/validation periodu) vrati
manje od otprilike **100-250 nezavisnih dogadjaja** (zavisno od toga koliko
je ciljani edge realno "ekonomski znacajan" za ovu strategiju — vjerovatno
negdje izmedju 0.3% i 0.5% po trejdu s obzirom na niz troskova + funding),
hipoteza se na ovom skupu podataka proglasava **"neodlucivom"** (ne "nema
edge", nego "nemamo dovoljno podataka da bilo sta tvrdimo") — to se zapisuje
kao takvo, ne tumaci se kao negativan rezultat niti kao povod za labaviji
prag percentila da bi se dobilo vise dogadjaja (to bi bilo tuniranje nakon
rezultata, zabranjeno pravilom iz sekcije 2).

## 6. Rok

4-6 sedmica od prvog runa do odluke: nastaviti (holdout prolazi →
forward paper test), pauzirati (neodlucivo, treba vise podataka ili vremena)
ili zatvoriti (train/validation ili holdout ne prolazi kriterij).

## 7. Poznato ogranicenje specifikacije (zapisano unaprijed, ne naknadno opravdanje)

Funding ekstrem se cesto poklapa sa jakim trendom (funding postaje ekstreman
upravo kad cijena dugo ide u jednom smjeru) — kontra-signal moze izgubiti
visestruko prije nego se trend obrne. Fiksno vrijeme drzanja bez SL u ovoj,
prvoj verziji je namjerno: da se izlaz ne tunira, cak i ako bi dodavanje
SL-a "popravilo" rezultat nakon sto se vidi gdje su najgori gubici. Ako
train/validation period pokaze da su gubici dominantno vodjeni ovim
efektom, to je legitiman nalaz (hipoteza u ovom obliku ne radi), ne razlog
da se odmah doda SL i re-testira kao da je "ista" hipoteza.
