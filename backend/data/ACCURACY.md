# Accuracy report

Generated 2026-09-04T10:47:31 on 340 held-out validation images.

**These numbers describe synthetic labels through a simulated imaging chain — they measure the architecture, not real retail packaging.** See `0_population_this_measures` in the JSON.


## Gates, derived from the calibration split

- resolution floor: **6.0 px/mm** (below this the height is refused)
- error band: **±0.15645 mm** (bias +0.00877, sigma 0.03692, k=4.0)
- coplanarity: slope ≤ 0.63°, spread ≤ 5.212°
- max marker tilt: 35.0°
- bias subtracted from measurements: **False**


### Why that floor (error vs capture resolution)

| floor px/mm | n | bias mm | sigma mm | MAE mm | p95 abs mm |
|---|---|---|---|---|---|
| ≥6.0 | 431 | +0.0088 | 0.0369 | 0.0235 | 0.0815 |
| ≥8.0 | 414 | +0.0076 | 0.0359 | 0.0225 | 0.0810 |
| ≥10.0 | 366 | +0.0052 | 0.0306 | 0.0185 | 0.0697 |
| ≥12.0 | 321 | +0.0026 | 0.0255 | 0.0148 | 0.0490 |
| ≥14.0 | 274 | +0.0010 | 0.0240 | 0.0131 | 0.0439 |
| ≥16.0 | 231 | +0.0011 | 0.0202 | 0.0106 | 0.0370 |
| ≥18.0 | 179 | +0.0003 | 0.0183 | 0.0078 | 0.0222 |

## 1. Field extraction

Legible-frame rate: **82.7%** (frames the system judged readable enough to claim a declaration absent)


### legible frames only

| declaration | recall | precision | adjudicated | abstained | value exact |
|---|---|---|---|---|---|
| manufacturer | 0.966 | 1.000 | 281 | 0 | — |
| commodity_name | 1.000 | 1.000 | 254 | 0 | — |
| net_quantity | 0.908 | 1.000 | 281 | 0 | 0.901 |
| mfg_date | 0.989 | 0.968 | 281 | 0 | 0.952 |
| mrp | 0.933 | 1.000 | 281 | 0 | 0.888 |
| consumer_care | 0.978 | 1.000 | 281 | 0 | — |

### all frames

| declaration | recall | precision | adjudicated | abstained | value exact |
|---|---|---|---|---|---|
| manufacturer | 0.969 | 1.000 | 302 | 38 | — |
| commodity_name | 1.000 | 1.000 | 275 | 0 | — |
| net_quantity | 0.914 | 0.996 | 302 | 38 | 0.904 |
| mfg_date | 0.990 | 0.965 | 317 | 23 | 0.911 |
| mrp | 0.939 | 1.000 | 307 | 33 | 0.887 |
| consumer_care | 0.980 | 1.000 | 309 | 31 | — |

## 2. Height estimation (mm)

Overall: bias **+0.0207 mm**, sigma **0.2213 mm**, MAE **0.0551 mm**, p95 |e| **0.1409 mm** (n=787)
Yield: **93.2%** of located declarations produced a height; **80.4%** end-to-end of declarations actually printed.

Conditioned on the extractor having read the RIGHT text (classification failures excluded): bias **+0.0016 mm**, sigma **0.1288 mm**, MAE **0.0327 mm**, p95 |e| **0.1102 mm** (n=758). The gap between this and the line above is misclassification, not mis-measurement.


| split | n | bias mm | sigma mm | MAE mm | p95 abs mm | rel MAE % |
|---|---|---|---|---|---|---|
| tier: clean | 239 | -0.0041 | 0.0707 | 0.0099 | 0.0161 | 0.49 |
| tier: harsh | 109 | +0.1401 | 0.4740 | 0.2097 | 1.3983 | 12.58 |
| tier: realistic | 439 | +0.0046 | 0.1602 | 0.0413 | 0.0965 | 2.43 |
| field: net_quantity | 244 | -0.0118 | 0.1957 | 0.0416 | 0.1270 | 1.81 |
| field: mrp | 257 | +0.0079 | 0.1497 | 0.0409 | 0.1120 | 2.57 |
| field: mfg_date | 286 | +0.0601 | 0.2823 | 0.0793 | 0.2139 | 5.08 |

### by capture resolution

| src px/mm | n | bias mm | sigma mm | MAE mm |
|---|---|---|---|---|
| 6-9 | 46 | +0.0089 | 0.2962 | 0.1340 |
| 9-12 | 150 | +0.0242 | 0.0625 | 0.0498 |
| 12-16 | 162 | -0.0140 | 0.2192 | 0.0401 |
| 16-22 | 288 | -0.0088 | 0.1161 | 0.0295 |
| >=22 | 123 | -0.0051 | 0.1317 | 0.0247 |

### by marker tilt

| tilt deg | n | bias mm | sigma mm | p95 abs mm |
|---|---|---|---|---|
| 0-5 | 243 | -0.0016 | 0.0938 | 0.0189 |
| 5-10 | 51 | +0.0074 | 0.1442 | 0.1305 |
| 10-15 | 206 | -0.0080 | 0.2263 | 0.1403 |
| 15-20 | 159 | -0.0113 | 0.1639 | 0.1786 |
| 20-25 | 83 | +0.0166 | 0.0620 | 0.1531 |
| 25-30 | 23 | +0.0325 | 0.0562 | 0.1317 |
| >=30 | 4 | -0.0112 | 0.0631 | 0.0936 |

## 3. Overall verdict — the output the user actually sees

| ground truth | packs | non-compliant | REVIEW rate | decided | acc. when decided | violating pack CLEARED | compliant pack ACCUSED |
|---|---|---|---|---|---|---|---|
| strict | 340 | 203 | 0.432 | 0.568 | 0.927 | 0 (0.000) | 14 (0.041) |
| assessable | 340 | 199 | 0.432 | 0.568 | 0.922 | 0 (0.000) | 15 (0.044) |

## 3b. Verdict agreement per field

| field | n | decisive | acc. on decisive | false CLEAR | false VIOLATION | borderline | not assessed |
|---|---|---|---|---|---|---|---|
| net_quantity | 265 | 0.672 | 1.000 | 0 (0.000) | 0 (0.000) | 0.159 | 0.170 |
| mrp | 275 | 0.673 | 1.000 | 0 (0.000) | 0 (0.000) | 0.084 | 0.244 |
| mfg_date | 302 | 0.791 | 1.000 | 0 (0.000) | 0 (0.000) | 0.000 | 0.209 |

- **net_quantity** — within 0.3 mm of the threshold (n=89): 47.2% answered BORDERLINE, 0 wrong decisive calls. Clear of the threshold (n=176): accuracy 1.000, 0 wrong decisive calls.
- **mrp** — within 0.3 mm of the threshold (n=72): 31.9% answered BORDERLINE, 0 wrong decisive calls. Clear of the threshold (n=203): accuracy 1.000, 0 wrong decisive calls.
- **mfg_date** — within 0.3 mm of the threshold (n=27): 0.0% answered BORDERLINE, 0 wrong decisive calls. Clear of the threshold (n=275): accuracy 1.000, 0 wrong decisive calls.

### Does the band cover the cases it adjudicates?

On the 667 validation measurements that actually received a verdict: p95 |error| = **0.0869 mm**, p99 = 0.15908 mm, max = 0.304 mm, against a fitted band of **±0.15645 mm**.
1.05% fall outside the band. **band is adequate for the adjudicated population**


## 4. Coplanarity statistic, null distribution

On n=320 frames that ARE coplanar by construction: |median baseline slope| p99 = 0.4882°, max = 0.815°; spread p99 = 3.6192°.

## Throughput

Median **6543 ms** per scan (2 page-segmentation modes; `fast=true` uses one).
