# Could the model make money? A closing-line betting simulation

*A research exercise, not betting advice. Bets are assumed to be placed at the closing
moneyline, which is the hardest realistic test of a probability model.*

## The math in four lines

- A moneyline price converts to a payout multiplier ("decimal odds"). Odds of -110 pay
  1.909 per 1 staked.
- The two sides of a game never add up to 100%: at the closing line they add up to
  about **103.2%** in my data. That extra ~3% is the bookmaker's margin (the *vig*). A
  bettor with no edge loses it.
- **Expected value** of a bet at win probability p and decimal odds d: `EV = p(d - 1) - (1 - p)`.
  I bet the side with the larger EV when it exceeds a threshold.
- **Kelly sizing** stakes the fraction `f = (p*d - 1)/(d - 1)` of the bankroll, which
  maximizes long-run growth *if p is correct*. I use quarter-Kelly capped at 5% of the
  bankroll, because full Kelly is brutal when p is wrong.

## Setup

- Model probabilities: my Kalman team + QB ratings, walk-forward (each season predicted
  from earlier seasons only).
- Prices: closing moneylines from both sides (so the vig is real). 2,582 games in the dev
  window (2007-2016) and 2,493 in the holdout (2017-2025).
- **Threshold chosen on dev (2007-2016), scored once on the holdout (2017-2025).**
  Seven EV thresholds (0 to 8 percentage points) x two staking rules were tried. Flat and
  Kelly place the *same* bets and differ only in stake size, so there are seven distinct bet
  sets; the Deflated Sharpe Ratio below counts all 14 trials, which is conservative.

## Results (holdout 2017-2025)

Naive baselines, flat 1 unit per game, same games (what the vig alone does):

| Strategy | bets | ROI per bet [95% CI] |
|---|---|---|
| Bet every home team | 2,493 | -5.5% [-9.4, -1.5] |
| Bet every away team | 2,493 | -2.8% [-7.7, +2.2] |
| Bet every favorite | 2,493 | -2.8% [-5.6, -0.0] |
| Bet every underdog | 2,493 | -5.5% [-11.1, +0.5] |
| **Model: bet its larger-EV side, threshold chosen on dev (0)** | **2,054** | **-9.0% [-14.5, -3.2]** |

- Every one of the seven bet sets loses on the holdout (ROI from -7.3% to -10.3%).
- Flat staking: **-185 units** over 2,054 bets; worst peak-to-trough drop -217 units.
- Quarter-Kelly: bankroll **100 to 0.1**. That is ruin.
- Sharpe ratio per bet -0.068. The **Deflated Sharpe Ratio is 0.00**; luck alone across the 14
  trials would have produced a best Sharpe of about +0.011.
- Versus simply betting the favorite on the *same* 2,054 games, the model's picks do 6.0
  points worse [-13.6, +1.9]. That gap is not statistically distinguishable from zero, so
  the claim is "no better than naive", not "reliably worse".
- The model bets the underdog 72% of the time: it mostly thinks the market overprices favorites.

The model "sees" a positive-EV side in 82% of games and disagrees with the de-vigged
closing probability by 6.4 percentage points on average, roughly twice the vig. That is a
symptom of a noisy disagreement, not a large edge: in the encompassing regression the
market absorbs the model's information (the model's coefficient is -0.18 [-0.42, +0.06]),
and the correlation between model and market probabilities is 0.89.

## Dev to holdout: what changed?

The same rule returned -0.5% per bet on 2007-2016 [-6.4, +5.3], which is consistent with
zero, and -9.0% on 2017-2025.

| Season | bets | ROI | home share of bets |
|---|---|---|---|
| 2007-2016 (avg) | 2,290 | -0.5% | 43% |
| 2017 | 238 | -6.2% | 51% |
| 2018 | 230 | -0.8% | 50% |
| 2019 | 226 | -5.3% | 54% |
| 2020 | 228 | -15.9% | 66% |
| 2021 | 245 | -5.1% | 54% |
| 2022 | 234 | -9.9% | 57% |
| 2023 | 217 | -8.7% | 57% |
| 2024 | 223 | -11.3% | 57% |
| 2025 | 213 | -19.0% | 57% |

Single seasons are noisy (each one's interval is about +/-10 points; the dev decade swings
between -9% and +10%), but the holdout is negative in all nine seasons. Two explanations were
tested:

- *The empty-stadium seasons (2019-2021).* ROI is -8.7% inside them and -9.2% outside them
  [-16.2, -2.1], so they do not explain it.
- *A shrinking home edge the model failed to track.* The model's average home-win
  probability in 2017-2025 is 0.569 versus the market's 0.554 and the actual home-win rate of
  0.545, so the model *does* over-credit home teams, and its share of home bets rose from 43% to
  56%. But training with exponentially down-weighted old seasons (half-lives of 8, 5, 3 and 2
  seasons) leaves holdout log loss within 0.0003 of the baseline and ROI between -8.7% and
  -9.5%. (This hypothesis came from looking at holdout diagnostics, so it is flagged post hoc
  in the experiment registry, E35-E38.)

I did **not** establish why the dev decade looks better than the holdout. One untested
hypothesis is that closing lines became more efficient over this period; the 2007-2016 result
is also statistically indistinguishable from the holdout's sign, so a regime change is not
required to explain it.

## Takeaways

1. Beating the vig requires an edge of a few percentage points on the probability, as a rough
   guide (the 103.2% overround costs about 1.6 points per side), applied consistently. My model
   shows none in the holdout.
2. Pre-vig skill is invisible after costs: a ~3% fee turns a model that is merely "close"
   into a reliable loser. This is the same reason a quantitative trader needs an alpha larger
   than transaction costs.
3. Evaluating with probability scores (log loss) and with P&L are different questions; both
   say the model has no exploitable edge over the closing line.

```
python -m src.betting.run
python -m src.models.hfa_experiment
```
