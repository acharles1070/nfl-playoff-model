# Experiment registry and multiple-testing control

Every out-of-sample comparison run in this project (37 in total), as a single family.
`delta` is the paired log-loss difference versus the model it was tested against
(negative = better). p-values are recovered from the bootstrap intervals (normal
approximation); Holm controls the family-wise error rate, BH the false-discovery rate.

- nominally significant at 0.05: **4** (about 1.9 would be expected by chance alone)
- surviving Holm: **1**; surviving BH: **1**

| id | experiment | delta | 95% interval | p | p (Holm) | survives |
|---|---|---|---|---|---|---|
| E01 | Kalman team ratings vs season-to-date EPA | -0.0200 | [-0.0284, -0.0117] | 0.000 | 0.000 | **yes** |
| E02 | QB layer vs no-QB ratings (all games) | -0.0037 | [-0.0076, +0.0003] | 0.066 | 1.000 | no |
| E03 | QB layer, games where a starter changed | -0.0132 | [-0.0246, -0.0016] | 0.024 | 0.856 | no |
| E04 | QB layer, regular season only | -0.0044 | [-0.0084, -0.0003] | 0.033 | 1.000 | no |
| E05 | QB layer, postseason only | +0.0108 | [-0.0011, +0.0229] | 0.078 | 1.000 | no |
| E06 | Train from 2000 vs from 2010 | +0.0004 | [-0.0008, +0.0016] | 0.514 | 1.000 | no |
| E07 | Re-tune filter on 2002-2016 vs 2011-2016 values | -0.0005 | [-0.0020, +0.0010] | 0.514 | 1.000 | no |
| E08 | Margin regression + normal CDF vs logistic | -0.0008 | [-0.0022, +0.0004] | 0.228 | 1.000 | no |
| E09 | Heteroscedastic (uncertainty-widened) margin model | -0.0003 | [-0.0024, +0.0017] | 0.774 | 1.000 | no |
| E10 | Margin Kalman rating added | -0.0002 | [-0.0028, +0.0024] | 0.880 | 1.000 | no |
| E11 | Pass/rush split filters | +0.0000 | [-0.0023, +0.0022] | 1.000 | 1.000 | no |
| E12 | + rest difference | -0.0001 | [-0.0013, +0.0012] | 0.875 | 1.000 | no |
| E13 | + bye flags | -0.0001 | [-0.0019, +0.0018] | 0.916 | 1.000 | no |
| E14 | + neutral site | +0.0007 | [-0.0011, +0.0025] | 0.446 | 1.000 | no |
| E15 | + bye flags + neutral | +0.0006 | [-0.0018, +0.0031] | 0.631 | 1.000 | no |
| E16 | + rest + bye + neutral | +0.0007 | [-0.0017, +0.0032] | 0.575 | 1.000 | no |
| E17 | + all context features | +0.0011 | [-0.0016, +0.0037] | 0.416 | 1.000 | no |
| E18 | Competitive plays only, wp 10-90% | +0.0026 | [-0.0021, +0.0073] | 0.278 | 1.000 | no |
| E19 | Competitive plays only, wp 5-95% | +0.0020 | [-0.0018, +0.0058] | 0.302 | 1.000 | no |
| E20 | Leverage-weighted plays | +0.0026 | [-0.0018, +0.0070] | 0.247 | 1.000 | no |
| E21 | Injury-report burden, totals | -0.0014 | [-0.0032, +0.0005] | 0.138 | 1.000 | no |
| E22 | Injury-report burden, by position group | -0.0011 | [-0.0040, +0.0020] | 0.472 | 1.000 | no |
| E23 | Injury burden, OL only | +0.0002 | [-0.0003, +0.0006] | 0.384 | 1.000 | no |
| E24 | Injury burden, DB only | -0.0004 | [-0.0016, +0.0007] | 0.495 | 1.000 | no |
| E25 | Two-stage EPA-residual injury adjustment | -0.0018 | [-0.0045, +0.0009] | 0.191 | 1.000 | no |
| E26 | Injured-reserve burden | +0.0011 | [-0.0031, +0.0052] | 0.603 | 1.000 | no |
| E27 | Game-day inactive burden | -0.0018 | [-0.0034, -0.0003] | 0.023 | 0.822 | no |
| E28 | IR + inactives | -0.0005 | [-0.0051, +0.0040] | 0.829 | 1.000 | no |
| E29 | IR + inactives + report | +0.0002 | [-0.0044, +0.0047] | 0.931 | 1.000 | no |
| E30 | IR + inactives by position group | +0.0027 | [-0.0026, +0.0077] | 0.304 | 1.000 | no |
| E31 | Injury-report burden (pooled 2014-25) | -0.0006 | [-0.0024, +0.0012] | 0.514 | 1.000 | no |
| E32 | Calibrated market vs raw market | -0.0001 | [-0.0010, +0.0008] | 0.828 | 1.000 | no |
| E33 | Stacked market + my model vs raw market | +0.0002 | [-0.0008, +0.0012] | 0.695 | 1.000 | no |
| E35 | Recency-weighted training, half-life 8 seasons (post hoc) | -0.0003 | [-0.0011, +0.0005] | 0.462 | 1.000 | no |
| E36 | Recency-weighted training, half-life 5 seasons (post hoc) | -0.0003 | [-0.0015, +0.0009] | 0.624 | 1.000 | no |
| E37 | Recency-weighted training, half-life 3 seasons (post hoc) | -0.0001 | [-0.0019, +0.0017] | 0.913 | 1.000 | no |
| E38 | Recency-weighted training, half-life 2 seasons (post hoc) | +0.0002 | [-0.0021, +0.0025] | 0.865 | 1.000 | no |

## How clean is the holdout?

Every comparison above is scored on the holdout window or later: 31 on 2017-2025, 6 on 2014-2025 and 0 on 2023-2025. The registry holds no development-window numbers. The Kalman ratings, the quarterback layer and roughly 20 of the early experiments were developed on 2026-09-30, when 2017-2025 was the only test window; the formal protocol (select ideas on 2002-2016, confirm on 2017-2025) was adopted on 2026-10-06 when history was extended to 1999. Hyperparameters were tuned on 2011-2016 only and frozen. The holdout is therefore not sealed: it has been examined for every idea, which is exactly why the family-wise corrections above are applied to these looks.

E34 is intentionally absent: that comparison used a licensed third-party dataset that cannot be published. It showed no detectable effect and is not counted above; including it would not change the survivor.

## Replication of E01 on seasons outside its design (pre-specified 2026-10-07)

E01 (Kalman ratings vs season-to-date EPA) is the only experiment that survives correction. Test seasons 2003-2009 played no part in designing it or tuning it, so they are the cleanest check. Primary criterion fixed in advance: a 95% paired-bootstrap interval excluding zero on 2003-2009. Both models train walk-forward from 2000, so the holdout row differs slightly from E01, which uses the live model's 2010 training start.

| window | games | change in log loss | 95% interval | seasons Kalman better |
|---|---|---|---|---|
| primary: 2003-2009 (untouched by design and tuning) | 1,869 | -0.0076 | [-0.0158, +0.0004] | 4/7 |
| secondary: dev window 2002-2016 | 4,005 | -0.0146 | [-0.0204, -0.0089] | 8/15 |
| secondary: holdout 2017-2025 (= E01) | 2,494 | -0.0175 | [-0.0253, -0.0099] | 4/9 |

Result: the primary check does NOT meet the pre-specified criterion (the effect has the same sign and is smaller than on later windows). The survivor is supported across windows but not independently confirmed at the 5% level.

## Statistical power

The SD of per-game log-loss differences between two similar models is 0.100. With 80% power at 5% significance, the smallest true improvement a paired test can detect is:

| games | minimum detectable improvement |
|---|---|
| 111 | 0.0267 |
| 500 | 0.0126 |
| 1,000 | 0.0089 |
| 2,494 | 0.0056 |
| 5,000 | 0.0040 |
| 10,000 | 0.0028 |
| 25,000 | 0.0018 |

Confirming a 0.005 improvement needs about 3,157 games (~11 seasons); a 0.002 improvement needs about 19,728 games (~69 seasons). The 111 holdout playoff games can only detect improvements above ~0.027.
