# Digital twin: twin-adb-base

Portfolio `adb-base`. Every answer is certified at 99% confidence (IQAE with Chernoff-Hoeffding bounds, emulated at the exact amplitude of the verified circuit) and every quantum query is recorded. Exact values are shown for checking.

## Certified answers

| question | estimate | certified interval | exact |
|---|---:|---:|---:|
| P(both pay) | 4.4162% | 4.3013% to 4.5311% | 4.4138% |
| P(L>T) | 1.2004% | 1.1751% to 1.2257% | 1.2047% |
| stop-loss | USD 0.282m | USD 0.272m to USD 0.292m | USD 0.282m |
| P(both exhausted) | 0.0032% | 0.0031% to 0.0032% | 0.0032% |

Quantum queries: 847,100.

## Dependence sweep: shared-earthquake multiplier m from 0.5 to 1.5 (21 versions)

| target | twin queries | recompute queries | fresh estimates | coverage |
|---|---:|---:|---:|---:|
| P(both pay) | 91,400 | 106,700 | 9 of 21 | 100% |
| P(both exhausted) | 2,823,800 | 4,645,700 | 10 of 21 | 100% |

## Structure comparison by stop-loss (earthquake / rain split, USD m)

| A | B | certified | quantum queries | exact A | exact B | classical paired MC scenarios |
|---|---|---|---:|---:|---:|---:|
| 55/25 | 60/20 | A < B | 75,800 | USD 0.137m | USD 0.171m | 2,323 |
| 60/20 | 65/15 | A < B | 39,000 | USD 0.171m | USD 0.212m | 1,578 |
| 65/15 | 70/10 | A < B | 59,700 | USD 0.212m | USD 0.282m | 795 |

## What drives 'P(both exhausted)' (shares of conditional scenarios; categories can overlap)

| driver | share now (library) |
|---|---:|
| shared quakes alone exhaust both quake covers | 33% |
| each country's own quakes alone exhaust its cover | 56% |
| a shared storm triggers both rain covers | 84% |

At m = 1.2: kept 177 cached scenarios; shared quakes alone exhaust both quake covers: 42% (exact 42%); each country's own quakes alone exhaust its cover: 50% (exact 47%); a shared storm triggers both rain covers: 86% (exact 89%)

At m = 1.5: kept 175 cached scenarios; shared quakes alone exhaust both quake covers: 50% (exact 47%); each country's own quakes alone exhaust its cover: 40% (exact 40%); a shared storm triggers both rain covers: 86% (exact 89%)

Quantum queries for the library: 68,317.

Run time 0 s. Noiseless emulation; a query is not a unit of time.
