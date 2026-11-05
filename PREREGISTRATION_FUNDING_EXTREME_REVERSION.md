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
placa funding), pa je ocekivani povrat u SUPROTNOM smjeru u narednih **8h**
pozitivan nakon troskova.

## 2. Specifikacija (zamrznuta — ne tunirati nakon prvog runa)

- **Ulazni signal:** rolling percentil funding rate-a za (symbol, trenutak)
  racunat ISKLJUCIVO iz proslih podataka (bez look-ahead-a — percentil u
  trenutku t koristi samo funding istoriju striktno prije t).
  - percentil > 95 → otvori **short**
  - percentil < 5 → otvori **long**
  - **Rolling prozor (zamrznuto, ne tunirati):** 90 dana unazad = **270**
    proslih funding-opservacija (funding na Binance Futures se obracunava
    svakih 8h → 3 opservacije/dan). Prije nego za dati simbol postoji
    minimalno **90 proslih opservacija** (= 30 dana istorije), taj simbol
    NE emituje signal (percentil nad manje od 90 tacaka se smatra
    nepouzdanim/nedefinisanim, ne "priblizno OK").
- **Izlaz:** fiksno vrijeme drzanja = **8h** (jedan funding ciklus na
  Binance Futures). BEZ SL/TP tuniranja u ovoj, prvoj verziji — izlaz je
  cisto vremenski, da se izbjegne dodatni stepen slobode (SL/TP prag) prije
  nego se uopste zna da li sirovi smjer signala ima ikakvu edge.
- **Cooldown / ponovljeni ekstremi (zamrznuto, ranije nedefinisan stepen
  slobode):** funding ekstremi su jako autokorelisani — isti simbol cesto
  ostaje u percentil>95 (ili <5) vise uzastopnih 8h ciklusa. Pravilo (jedna
  recenica): **novi ulaz na istom (symbol, smjer) je dozvoljen najranije
  16h nakon prethodnog ulaza** na tom (symbol, smjer) — bez obzira da li je
  prethodna pozicija vec zatvorena (16h = drzanje od 8h + jos jedan pun
  funding ciklus od 8h). Ovo je izabrano unaprijed (ne tunirano na
  rezultatima) jer odgovara logici "jedan dogadjaj = jedan trejd", isti duh
  kao postojeci
  correlated-event dedup za cross-symbol klastere ispod.
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

### 2a. Ispravka cinjenica o dubini funding podataka (dodano 2026-11-05, prije prvog runa)

Ovo NIJE izmjena specifikacije (prag, prozor i izlaz ostaju isti kao gore),
nego ispravka pogresne tehnicke pretpostavke otkrivene provjerom PRIJE
pisanja signal-koda:

- Binance `/fapi/v1/fundingRate` je DOKUMENTOVAN kao "max 1000" po upitu,
  ali je EMPIRIJSKI izmjereno (direktan API poziv, 2026-11-05) da stvarno
  vraca najvise **500** zapisa po upitu bez obzira na trazeni `limit` —
  tj. jedan upit pokriva ~166.7 dana (500 * 8h), ne ~333 dana. Ovo je
  ispravljeno u kodu (`fetch_full_funding_history`, paginacija preko
  `endTime` istim obrascem kao za svijece).
- Sa paginacijom, STVARNA funding istorija ide mnogo dalje unazad nego
  nasa postojeca svijeca-istorija (~8 mjeseci): izmjereno 2026-11-05,
  anchored na vec zamrznutu holdout granicu
  (`end_timestamp_ms=1790253820000`), svih 20 simbola ima izmedju **805 i
  2481 dana** upotrebljivog perioda (nakon 90-dnevnog warm-upa) funding
  podataka — daleko vise od candle-podataka.
- **Zakljucak:** funding istorija NIJE ogranicavajuci faktor. Stvarna
  granica perioda signala ostaje dubina POSTOJECE candle-istorije
  (~8 mjeseci, isti prozor koji je vec koristen za zatvorenu trend/anomaly
  porodicu), jer je candle-podatak potreban za ulaz/izlaz po tacnom
  vremenu i za troskove. Period signala = presjek (funding istorija) ∩
  (candle istorija), do istog zamrznutog `end_timestamp_ms` za train period.
  Realno ocekivanje ostaje ~5 mjeseci upotrebljivog train perioda nakon
  90-dnevnog warm-upa (vidi sekciju 5) — i dalje malo, pa "neodlucivo"
  ostaje najvjerovatniji ishod.
- **ONDOUSDT i TAOUSDT:** cak i sa punom paginacijom imaju manje ukupne
  istorije (977 i 895 dana ukupno od listinga) nego ostali simboli, ali i
  dalje vise nego dovoljno za 90-dnevni warm-up + ~5-mjesecni train period
  — ostaju u univerzumu bez izmjene (pravilo minimuma opservacija je vec
  zamrznuto u sekciji 2 i prirodno ih propusta tek kad stignu do minimuma).
- **Holdout:** paginacija sama po sebi moze povuci zapise i nakon
  `end_timestamp_ms` ako se ne prosledi eksplicitno. Kod
  (`fetch_full_funding_history(symbol, end_time_ms=...)`) STROGO filtrira
  sve zapise nakon `end_time_ms` prije vracanja — holdout ne moze procuriti
  kroz ovaj (funding) data-izvor.
- **Dataset hash:** funding zapisi se sad kesiraju i hesiraju
  (`cached_fetch_full_funding_history` / `funding_dataset_fingerprint` u
  `scanner/backtesting/data_snapshot.py`), kombinovano sa postojecim
  candle-hashom preko `combined_dataset_fingerprint`, tako da jedan
  `dataset_hash` po trial-u pokriva OBA izvora podataka.

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
- **Holdout je resurs koji se trosi (bitno, zapisano unaprijed):** ovaj isti
  zakljucani period (zadnjih ~6 sedmica, posle `2026-09-24T12:43:40Z`) je
  JEDINI trenutno rezervisani holdout u projektu. Ako se otvori za ovu
  (funding) hipotezu, time je POTROSEN — vise nije "netaknut" ni za jednu
  buducu hipotezu. Svaka sljedeca hipoteza (nakon funding-ekstrema) ce
  trebati NOVI, svjez zakljucani period (npr. prirodno pomjeren naprijed u
  vremenu, posto scanner i dalje radi 24/7 i akumulira nove podatke). Ovo
  se eksplicitno zapisuje OVDJE da se izbjegne tiha pretpostavka da je isti
  holdout "besplatno" dostupan za svaku narednu porodicu.

## 4. Kriterij uspjeha (definisan PRIJE prvog runa)

Na train/validation periodu:
1. **Neto expectancy po trejdu > 0** (sirovi, block-bootstrap
   `observed_pct` — ovo je PRIMARNA mjera za ovu hipotezu, ne equity-curve
   kompaundovana mjera — vidi upozorenje iz Koraka 7 o tome da se ove dvije
   mjere mogu razilaziti po predznaku kad je efekat mali).
2. **Donja granica 95% block-bootstrap CI za prosjecan povrat po trejdu > 0**
   (`edge_survives_lower_bound: true`).

Uz ova dva (obavezna) kriterija, run MORA izvjestiti i **bruto rezultat
prije troskova** (gross profit/expectancy bez fee/spread/slippage/funding)
kao treci, dijagnosticki broj — ne kao uslov prolaza, nego da se odmah vidi
da li eventualni neuspjeh dolazi od (a) nepostojece sirove edge, ili (b)
postojece sirove edge koju troskovi pojedu (isto razlikovanje koje je
razjasnilo zatvaranje trend/anomaly porodice u Koraku 7).

Tek ako OBA obavezna uslova prodju na train/validation periodu, otvara se
holdout (tacno jednom, isti kriterij se primjenjuje na njega, ukljucujuci
bruto izvjestavanje). Prolaz na holdoutu vodi u **forward paper test** (live
papir-trading na zamrznutoj specifikaciji), NE direktno u trgovanje pravim
novcem.

## 5. Minimalni N (power racun, PRIJE prvog runa)

Nemamo jos funding-specificnu varijansu povrata po trejdu (nijedan trejd ove
hipoteze jos ne postoji), pa je donja tabela provizorna procjena koja
pozajmljuje empirijsku standardnu devijaciju iz UPRAVO ZATVORENE trend/
anomaly porodice (nazad izracunato iz njenog block-bootstrap CI-a na
frozen-baseline runu: `n_independent=103`, CI pola-sirine `0.5358%` →
implicirano `sigma ~= 2.77%` po trejdu). Ovo ce se ZAMIJENITI stvarnom
izmjerenom varijansom cim se prvi funding-backtest run zavrsi — ovaj broj je
samo plan koliko podataka treba prikupiti, ne konacan test.

**VAZNO — ova tabela je samo plan, NIJE konacan test (eksplicitno pravilo):**
Prag "neodlucivo" NIJE fiksiran na brojevima iz ove tabele (koji koriste
pozajmljenu sigmu). Stvarno pravilo je:

> **neodlucivo** = `n_independent < n_required(sigma_izmjerena, d=0.3%)`,
> gdje se `sigma_izmjerena` racuna IZ OVOG runa (block-bootstrap na
> funding-trejdovima), ne iz tabele ispod. Ciljani edge za ovu odluku je
> zamrznut na **d = 0.3%** po trejdu (srednja vrijednost iz "ekonomski
> znacajnog" raspona 0.3-0.5% navedenog ranije — bira se gornja/konzervativnija
> granica tog raspona da prag bude teze proci slucajno).

**Definicija `n_independent` (zapisano unaprijed, da se ne raspravlja
naknadno):** broj preostalih KLASTERA nakon primjene cooldown-a (16h
pravilo, sekcija 2) i cross-symbol correlated-event dedup-a (sekcija 2) —
NE sirovi broj pojedinacnih trejdova prije tog filtriranja. Isto znacenje
kao `n_independent_episodes` koji vec postoji u postojecem backtest
izvjestaju za trend/anomaly porodicu.

Tabela ispod postoji samo da se UNAPRIJED vidi red velicine (koliko
dogadjaja bi bilo "dovoljno" za razlicite pretpostavljene sigme/edge-ove), ne
da propisuje tacan broj.

Formula: `n = (z_(a/2) + z_beta)^2 * sigma^2 / d^2`, dvostrani alpha=0.05
(`z=1.96`), power=80% (`z=0.84`).

| Ciljani edge po trejdu (nakon troskova) | Potreban N (nezavisnih dogadjaja), sigma=2.77% (pozajmljeno) |
|---|---|
| 0.10% | ~6 033 |
| 0.20% | ~1 509 |
| 0.30% | ~671 |
| 0.50% | ~242 |
| 0.75% | ~108 |
| 1.00% | ~61 |

**Realno ocekivanje (zapisano UNAPRIJED, prije nego vidimo rezultat):**
funding ekstremi na 8h ciklusu su jako autokorelisani (isti simbol cesto
ostaje u percentil>95/<5 vise uzastopnih ciklusa) i cesto se poklapaju
preko vise simbola odjednom (cross-symbol klasteri, isto kao sto je
viddjeno u trend/anomaly porodici). Nakon cooldown-a (sekcija 2) i
cross-symbol dedup-a na ~8 mjeseci podataka, **realno ocekivan broj
nezavisnih dogadjaja je vjerovatno NEKOLIKO DESETINA, ne stotina** — dakle
znatno ispod bilo kojeg broja u tabeli iznad. Najvjerovatniji ishod prvog
runa je zato **"neodlucivo"**, ne jasan prolaz ili pad. Ovo se ne smatra
razocaravajucim niti povodom da se prag percentila ublazi (npr. na 90/10)
da bi se "nabrao" vise dogadjaja — takva izmjena bi bila NOVA, zasebna
hipoteza (vidi sekciju 2), ne popravka ove.

**Sta tacno znaci "neodlucivo" (zapisano unaprijed, da se izbjegne tiho
beskonacno ponavljanje):** ako je ishod neodlucivo, hipoteza se NE vraca na
istu specifikaciju "s malo vise podataka" bez novog, posebno
pre-registrovanog perioda/datuma. "Pauzirati" (sekcija 7) znaci cekati da
prirodno prodje dovoljno kalendarskog vremena da se nakupi NOVI, svjez
train/validation period — ne ponovno gledati isti vec iskorisceni period
nadajuci se da ce ovaj put biti dovoljno dogadjaja.

## 6. Multiple-testing racunovodstvo

Ovo je **trial/porodica #2** u ukupnom projektu (porodica #1 = trend/anomaly,
original+inverted, zatvorena u Koraku 7). Ukupan broj TESTIRANIH porodica
(ne pojedinacnih runova unutar jedne porodice) raste na 2 nakon ovog runa —
ovaj broj se mora koristiti ako/kad se kasnije racuna bilo kakva
multiple-testing korekcija (npr. Bonferroni/Holm) preko svih dosad
pokusanih hipoteza, ne samo unutar jedne porodice. Zapisuje se ovdje da se
ne izgubi kad porodica #3 bude pre-registrovana.

## 7. Rok

4-6 sedmica od prvog runa do odluke: nastaviti (holdout prolazi →
forward paper test), pauzirati (neodlucivo, treba vise podataka ili vremena)
ili zatvoriti (train/validation ili holdout ne prolazi kriterij).


## 8. Poznato ogranicenje specifikacije (zapisano unaprijed, ne naknadno opravdanje)

Funding ekstrem se cesto poklapa sa jakim trendom (funding postaje ekstreman
upravo kad cijena dugo ide u jednom smjeru) — kontra-signal moze izgubiti
visestruko prije nego se trend obrne. Fiksno vrijeme drzanja bez SL u ovoj,
prvoj verziji je namjerno: da se izlaz ne tunira, cak i ako bi dodavanje
SL-a "popravilo" rezultat nakon sto se vidi gdje su najgori gubici. Ako
train/validation period pokaze da su gubici dominantno vodjeni ovim
efektom, to je legitiman nalaz (hipoteza u ovom obliku ne radi), ne razlog
da se odmah doda SL i re-testira kao da je "ista" hipoteza.
