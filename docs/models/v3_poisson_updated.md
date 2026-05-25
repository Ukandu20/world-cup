# V3 Poisson Model (Updated)

## Changelog

| Version | Changes |
|---|---|
| v3.1 | Added `wc_l5_goal_difference` to active feature set per correlation analysis |
| v3.1 | Added Dixon-Coles low-score correction (rho parameter) |
| v3.1 | Added stage-specific lambda scaling factors |
| v3.1 | Expanded backtesting to rolling three-tournament holdout (2014, 2018, 2022) |
| v3.1 | Added calibration curve reporting to backtest suite |
| v3.1 | Added time-decay sample weights to training pipeline |
| v3.1 | Added `is_knockout` flag as prediction-time feature |
| v3.1 | Tuned regularization alpha via cross-validation |
| v3.1 | Added `has_wc_l5_history` missingness flag for teams without sufficient WC history |
| v3.2 | Rebuilt `v3_strength` with correlation-anchored coefficients |
| v3.2 | Added `wc_l5_goal_difference` to `v3_strength` formula |
| v3.2 | Reduced host flag bonus from 25 to 15 in `v3_strength` |
| v3.2 | Added per-fold metric reporting with std dev to backtest output |
| v3.2 | Documented three-holdout rationale and data availability constraints |

---

## Purpose

V3 is the project's expected-goals model. It predicts separate expected goal
rates for the two teams in a match, converts those rates into score and
win/draw/loss probabilities, and simulates the full tournament.

V3 is the most realistic current probability model because group-table outcomes
depend on simulated goals, goal difference, and goals scored rather than on
outcome classes alone.

---

## Inputs

V3 uses:

- Current team metadata:
  - team id
  - display name
  - group code
  - confederation
  - Elo rating
  - host flag
- Lead-in match features (linearly recency-weighted over last k matches):
  - result score
  - goals for
  - goals against
  - capped goal difference
  - performance versus Elo expectation
  - pre-tournament Elo
- Recent World Cup history (last 5 editions):
  - weighted finish score
  - goal difference
  - goal difference per match (weighted and unweighted)
  - goals against per match
  - appearances and appearance rate
  - best finish score
  - WC history missingness flag
- Match context:
  - competition importance
  - neutral-site flag
  - net host flag
  - knockout stage flag
  - stage lambda multiplier
- Historical international or World Cup-only training data.
- 2026 fixtures.

The default training scope is:

```text
all_international_since_anchor
```

---

## Feature Selection Methodology

Active features were selected through a two-stage empirical process:

**Stage 1 — Correlation analysis.** Pearson and Spearman correlations were
computed between each candidate feature and the tournament performance target,
across all World Cup editions since 1930. Features with absolute Spearman
correlation below 0.10 were dropped. ELO change features were consistently
near-zero across all window definitions and were eliminated.

**Stage 2 — Partial correlation against `start_elo`.** Remaining features were
tested for independent signal after partialling out `start_elo`, the strongest
single predictor. Features that lost significance after this adjustment were
flagged for removal or consolidation.

Key findings from the correlation analysis:

```text
start_elo                               Spearman = 0.477   anchor feature
wc_l5_goal_difference                   Spearman = 0.424   2nd strongest overall
weighted_wc_l5_finish_score             Spearman = 0.387
prior_best_finish_score                 Spearman = 0.384
weighted_form_l10_goal_diff_per_match   Spearman = 0.348
wc_l5_goals_against_per_match           Spearman = -0.303
prior_world_cup_participations          Spearman = 0.339
is_host                                 Spearman = 0.225
```

Features eliminated by correlation analysis:

```text
form_l10_elo_change               Spearman = 0.011   noise across all windows
form_l10_elo_change_per_match     Spearman = 0.008   noise
weighted_form_elo_change          Spearman = 0.047   noise
wc_l5_elo_change                  Spearman = 0.100   noise in WC window too
wc_l5_goals_against               Spearman = 0.208   Simpson's paradox (raw total)
form_l10_matches                  Spearman = 0.087   no signal
```

Note on `wc_l5_goals_against` sign flip: the raw total shows positive
correlation because teams that play more WC matches (stronger teams) concede
more total goals. The per-match version correctly shows negative correlation
and is retained instead.

---

## Core Method

V3 trains two separate Poisson regression models:

```text
home_goal_model predicts home_score
away_goal_model predicts away_score
```

Both models use the same standardized feature columns. For prediction, the
models output:

```text
lambda_home = expected home goals
lambda_away = expected away goals
```

Those lambdas are then:

1. Scaled by a stage-specific multiplier.
2. Fed into the Dixon-Coles corrected probability matrix.

---

## Mathematical Formulas

### V3 Strength Score

V3 computes a scalar `v3_strength` for display ordering and tie-break
fallbacks only. It is not the Poisson model itself and does not influence
probability calculations.

#### Formula

```text
v3_strength =
    elo_rating
  + 200 * results_form
  + 180 * perf_vs_exp
  + 150 * placement
  + 100 * wc_l5_goal_diff_norm
  + 50  * gd_form
  + 30  * goals_for
  - 25  * goals_against
  + 20  * appearance
  + 15  * host_flag
```

All features except `elo_rating` and `host_flag` are on a 0..1 scale before
the coefficient is applied.

#### Coefficient Derivation

Coefficients are anchored to the empirical Spearman correlations from the
feature selection analysis. The principle is that a feature's coefficient
should be proportional to its predictive signal relative to `start_elo`,
scaled so that each component contributes a meaningful share of the total
score without any single non-ELO feature dominating.

```text
Feature                         Spearman    Relative to ELO    Coefficient
─────────────────────────────────────────────────────────────────────────
elo_rating                      0.477       1.000              1.0 (anchor)
results_form                    0.313       0.656              200
perf_vs_exp                     0.347       0.728              180
placement (wc_l5_finish)        0.387       0.812              150
wc_l5_goal_diff_norm            0.424       0.889              100
gd_form                         0.348       0.730               50
goals_for                       0.281       0.589               30
goals_against                  -0.238      -0.499              -25
appearance                      0.312       0.654               20
host_flag                       0.225       0.472               15
```

The coefficients are smaller for features where the 0..1 range captures
less variance (e.g. `appearance` has a narrower effective spread than
`results_form`), and larger for features with wider spreads that need
more scaling to register against ELO's natural 1200–1800 range.

#### Changes from Previous Formula

```text
Feature             Old coefficient    New coefficient    Reason
────────────────────────────────────────────────────────────────────────
results_form        120                200                was underweighted
perf_vs_exp          80                180                most sophisticated
                                                          signal, was underweighted
placement            60                150                WC pedigree underweighted
wc_l5_goal_diff_norm —                 100                new feature added
gd_form              18                 50                was severely underweighted
goals_for            22                 30                minor upward adjustment
goals_against       -18                -25                minor upward adjustment
appearance            8                 20                was severely underweighted
host_flag            25                 15                was too generous;
                                                          flat bonus overstated
                                                          home advantage
```

The most significant changes are to `perf_vs_exp` (from 80 to 180) and
`gd_form` (from 18 to 50). The original formula heavily underweighted both
despite `perf_vs_exp` being the most analytically sophisticated component
and `gd_form` having comparable predictive signal to `results_form`.

The host flag reduction from 25 to 15 reflects that a flat ELO-equivalent
bonus of 25 points was generous for a display score — typical empirical
home advantage in football is 50–70 ELO points in match probability terms,
but that applies to a genuine home fixture, not all WC host-nation matches
which include neutral-site group games for co-hosts.

#### Limitation

`v3_strength` is a heuristic ranking score. Its coefficients are informed
by correlation analysis but are not derived from regression on a target
variable. Two teams with similar strength scores may have very different
win probabilities depending on matchup — the Poisson model is the
authoritative source for all probability outputs.

### Match Feature Differences

For a predicted match between home-slot team H and away-slot team A, V3 uses:

```text
elo_diff                  = elo_H - elo_A
results_form_diff         = results_form_H - results_form_A
goals_for_diff            = goals_for_H - goals_for_A
goals_against_diff        = goals_against_H - goals_against_A
gd_form_diff              = gd_form_H - gd_form_A
perf_vs_exp_diff          = perf_vs_exp_H - perf_vs_exp_A
placement_diff            = placement_H - placement_A
appearance_diff           = appearance_H - appearance_A
wc_l5_goal_diff_diff      = wc_l5_goal_difference_H - wc_l5_goal_difference_A
competition_importance    = World Cup finals weight
neutral_site_flag         = 1 if neutral, else 0
net_host_flag             = host_flag_H - host_flag_A
is_knockout               = 1 if knockout stage, else 0
```

The active feature columns are:

```text
elo_diff
results_form_diff
goals_for_diff
goals_against_diff
gd_form_diff
perf_vs_exp_diff
placement_diff
appearance_diff
wc_l5_goal_diff_diff          ← added in v3.1
competition_importance
neutral_site_flag
net_host_flag
is_knockout                   ← added in v3.1
```

### Handling Missing WC L5 History

Teams with fewer than 5 World Cup appearances have no `wc_l5_goal_difference`
value. This missingness is non-random — debut teams and historically weaker
nations are overrepresented. Two steps are applied:

```text
1. A binary flag has_wc_l5_history is added as a feature:
   has_wc_l5_history = 1 if wc_l5_goal_difference is not null, else 0

2. Missing wc_l5_goal_difference values are imputed with the historical
   mean of all debut and first-appearance teams in the training data,
   not the overall mean.
```

This allows the model to learn that missing WC history is itself a signal
of relative weakness, rather than treating it as neutral.

### Weighted Form Inputs

The V3 current-team feature table uses linear recency weighting over the last
k matches, sorted oldest to newest:

```text
oldest selected match weight = 1
newest selected match weight = k
weighted_metric = sum(metric_i * i) / (k * (k + 1) / 2)
```

For k = 10: total weight = 55, newest share = 18.2%, oldest share = 1.8%.

This gives a 10:1 ratio between the most recent and oldest match. The softer
linear decay (vs. quadratic used for WC edition weighting) is deliberate:
form matches involve the same squad weeks or months apart, so older matches
remain more informative than in the 4-year WC edition context.

The form inputs are:

```text
results_form
gd_form
perf_vs_exp
goals_for
goals_against
pre_tournament_elo
```

### Poisson Regression

The training feature matrix is standardized with `StandardScaler`. V3 fits:

```text
PoissonRegressor(alpha = optimized via cross-validation, max_iter = 1000)
```

for home goals and away goals separately.

Alpha is tuned via `TimeSeriesSplit(n_splits=5)` cross-validation over the
range `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]`, scored by
`neg_mean_poisson_deviance`. Given the limited WC training sample, the optimal
alpha is expected to be higher than the previous default of 0.1.

Poisson regression models a non-negative expected count:

```text
lambda = exp(intercept + beta dot x)
```

Predicted lambdas are scaled by the stage multiplier before being passed to
the probability matrix:

```text
lambda_home_adj = clip(lambda_home * stage_multiplier, 0.05, 4.5)
lambda_away_adj = clip(lambda_away * stage_multiplier, 0.05, 4.5)
```

### Stage Lambda Scaling

Goals per match decline in later knockout rounds due to increased defensive
conservatism and higher stakes. Empirical multipliers are computed from
historical World Cup data relative to the group stage baseline:

```text
STAGE_GOAL_MULTIPLIER = {
    'group':       1.00,   # baseline
    'round_of_16': 0.91,
    'quarter':     0.85,
    'semi':        0.81,
    'final':       0.80,
}
```

These multipliers are applied to both `lambda_home` and `lambda_away` before
the Dixon-Coles correction and probability matrix calculation. They are
re-estimated each training cycle from the historical WC match dataset:

```python
stage_avg = df.groupby('stage')['total_goals'].mean()
multipliers = stage_avg / stage_avg['group']
```

### Dixon-Coles Low-Score Correction

Independent Poisson models systematically underestimate the probability of
low-scoring draws (0-0, 1-1) and slightly miscalibrate 1-0 and 0-1 outcomes.
The Dixon-Coles correction applies a single parameter `rho` to the 2x2
low-score corner of the score probability matrix.

The correction factor tau is:

```text
For home_goals h and away_goals a, given lambda_h, lambda_a, rho:

tau(0, 0) = 1 - (lambda_h * lambda_a * rho)
tau(0, 1) = 1 + (lambda_h * rho)
tau(1, 0) = 1 + (lambda_a * rho)
tau(1, 1) = 1 - rho
tau(h, a) = 1.0   for all h > 1 or a > 1
```

Applied to the score matrix:

```text
score_probability[h, a] = P(h) * P(a) * tau(h, a, lambda_h, lambda_a, rho)
```

The matrix is renormalized after correction so probabilities sum to 1.0.

`rho` is estimated from training data by maximum likelihood. For football it
typically falls in the range 0.05 to 0.15. A value of 0.0 reduces the model
to standard independent Poisson.

The correction is applied in all match probability calculations including
group stage, knockout, and backtest evaluation.

### Poisson Score Probabilities

For a team with adjusted expected goals `lambda_adj`, the Poisson probability
of scoring `g` goals is:

```text
P(G = g) = exp(-lambda_adj) * lambda_adj^g / g!
```

V3 computes probabilities for goals 0..10:

```text
V3_POISSON_GOAL_CAP = 10
```

The probability mass above 10 is folded into the 10-goal bucket.

### Home, Draw, Away Probability Matrix

V3 builds a score matrix from the Dixon-Coles corrected goal distributions:

```text
score_probability[h, a] = P(home_goals = h) * P(away_goals = a) * tau(h, a)
```

Then:

```text
home_win_prob = sum(score_probability[h, a] where h > a)
draw_prob     = sum(score_probability[h, a] where h = a)
away_win_prob = sum(score_probability[h, a] where h < a)
```

The three probabilities are normalized to sum to 1.0.

---

## Training Pipeline

### Competition Importance Sample Weights

Each training match is weighted by competition type to emphasize the target
domain (World Cup finals):

```text
World Cup finals    = 3.0
Continental finals  = 2.5
Qualifiers          = 2.0
Other competitive   = 1.5
Friendlies          = 1.0
```

These weights are applied as `sample_weight` in the Poisson regression fit.
They are treated as tunable hyperparameters and should be validated against
rolling holdout log loss. The current values are informed priors, not
empirically optimized.

Note on qualifiers: qualification match quality varies significantly by
confederation. CONMEBOL qualifiers involve stronger opposition on average than
some other confederation qualification routes. Confederation-adjusted qualifier
weights are a candidate future improvement.

### Time-Decay Sample Weights

In addition to competition importance, each training match receives a
time-decay weight based on its distance from the training cutoff date:

```text
time_weight = exp(-decay_rate * days_before_cutoff)
```

The decay rate is tuned via rolling holdout cross-validation. A larger decay
rate concentrates learning on recent matches. The final sample weight combines
both signals:

```text
final_weight = competition_weight * time_weight
```

This ensures that a recent friendly is not necessarily penalized below an
old World Cup qualifier.

### Training Cutoff Policy

For live 2026 prediction, training includes all matches up to the day before
the first 2026 World Cup fixture.

For holdout backtests, the cutoff is set to the day before the first match
of the holdout tournament.

---

## Simulation Flow

### Group Stage

For each 2026 group-stage fixture:

1. Build the V3 feature row.
2. Predict `lambda_home` and `lambda_away`.
3. Apply group stage multiplier (1.00 baseline).
4. Apply Dixon-Coles tau correction to score matrix.
5. Sample goals:

```text
home_goals ~ Poisson(lambda_home_adj)
away_goals ~ Poisson(lambda_away_adj)
```

6. Award points and update goals for/against.
7. Rank the group table.

The 48-team tournament advances the top two teams from each of 12 groups plus
the best eight third-place teams (ranked by points, then goal difference, then
goals scored across all third-place finishers).

### Knockout Stage

For knockout matches:

1. Build feature row with `is_knockout = 1`.
2. Predict lambdas.
3. Apply knockout-stage multiplier.
4. Apply Dixon-Coles correction.
5. Simulate regulation goals.
6. If tied, simulate extra time:

```text
extra_time_lambda = regulation_lambda_adj * (1 / 3)
```

7. If still tied, choose a random penalty winner.

The `is_knockout` flag allows the model to learn systematic differences in
match dynamics between group and knockout football beyond the stage multiplier.

The knockout simulator caches matchup probability and lambda calculations so
repeated matchups within the same simulation flow are not recomputed.

### Host and Neutral-Site Logic

V3 has explicit host flags for:

```text
2022: Qatar
2026: Canada, Mexico, United States
```

For 2026 fixture prediction, a match is treated as neutral unless one of the
two teams has a host flag. The feature `net_host_flag` is:

```text
host_flag_home - host_flag_away
```

---

## Probability Outputs

V3 outputs:

- group finish probabilities: `prob_1`, `prob_2`, `prob_3`, `prob_4`
- best-third qualification: `top8_third_prob`
- knockout qualification: `ko_prob`
- reaching Round of 16: `r16_prob`
- reaching quarter-finals: `qf_prob`
- reaching semi-finals: `sf_prob`
- reaching final: `final_prob`
- winning the tournament: `champion_prob`

Each is a Monte Carlo estimate:

```text
stage_probability = stage_count / simulation_count * 100
```

V3 also exposes match-level quantities in backtests:

```text
lambda_home
lambda_away
lambda_home_adj       (post stage-multiplier)
lambda_away_adj       (post stage-multiplier)
home_win_prob
draw_prob
away_win_prob
rho                   (fitted Dixon-Coles parameter)
```

---

## Backtesting

### Rolling Three-Tournament Holdout

V3 is evaluated on a rolling holdout covering the last three World Cups:

```text
Fold 1: train through 2013-06-11, test on 2014 World Cup (64 matches)
Fold 2: train through 2017-06-13, test on 2018 World Cup (64 matches)
Fold 3: train through 2021-11-20, test on 2022 World Cup (64 matches)
```

Metrics are computed per fold and averaged across all three. This gives 192
matches of evaluation data and three independent bracket simulations, reducing
the variance inherent in any single tournament holdout.

The previous single-tournament (2022-only) holdout is retained for comparison
but is not the primary validation benchmark.

#### Why Three Folds

Three is a pragmatic constraint, not a statistical optimum. The limiting
factor is feature availability as holdouts extend further back:

```text
2022 holdout → train through Nov 2021   ✅  full feature coverage
2018 holdout → train through Jun 2017   ✅  full feature coverage
2014 holdout → train through Jun 2013   ✅  usable feature coverage
2010 holdout → train through Jun 2009   ⚠️  wc_l5 features degrade;
                                             many teams have < 5 WC appearances
2006 holdout → train through Jun 2005   ⚠️  WC L5 window breaks down
                                             for the majority of the field
2002 holdout → train through Jun 2001   ❌  insufficient WC L5 history
                                             to train a consistent model
```

Going beyond three folds would evaluate a model with materially different
feature coverage than the one deployed for 2026, making comparisons
misleading. When pre-2010 data density improves (e.g. if match-date
information is backfilled), the 2010 fold should be added as a fourth.

#### Variance Acknowledgement

Even across three folds, metric variance is non-trivial. Each World Cup
is a structurally different sample — different field composition, host
nation, era of football, and bracket draw. The standard deviation across
folds is reported alongside the average precisely because a model that
performs well on 2022 but poorly on 2014 and 2018 is not the same as one
that is consistently solid across all three.

A model cannot be fully validated on three samples. The fold metrics are
a directional reliability check, not a definitive benchmark.

### Per-Fold Metric Reporting

All metrics are reported per fold and as a three-fold summary. The
standard deviation column is as important as the average — high std dev
indicates sensitivity to tournament-specific conditions.

**Per-match metrics:**

```text
Metric                              2014    2018    2022    Mean    Std Dev
──────────────────────────────────────────────────────────────────────────
Multiclass log loss                 X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
Multiclass Brier score              X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
Top-1 match accuracy                XX%     XX%     XX%     XX%     XX%
Predicted draw rate                 XX%     XX%     XX%     XX%     —
Actual draw rate                    XX%     XX%     XX%     XX%     —
Predicted home win rate             XX%     XX%     XX%     XX%     —
Actual home win rate                XX%     XX%     XX%     XX%     —
MAE lambda_home                     X.XX    X.XX    X.XX    X.XX    X.XX
MAE lambda_away                     X.XX    X.XX    X.XX    X.XX    X.XX
ECE (home win)                      X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
ECE (draw)                          X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
ECE (away win)                      X.XXX   X.XXX   X.XXX   X.XXX   X.XXX
```

**Per-tournament metrics (one row per fold):**

```text
Metric                              2014    2018    2022
────────────────────────────────────────────────────────
Champion hit                        Y/N     Y/N     Y/N
Semifinal hits (of 4)               X/4     X/4     X/4
Round of 16 hits (of 16)            X/16    X/16    X/16
Group advancement accuracy          XX%     XX%     XX%
```

**Stage-split metrics (group vs. knockout, averaged across folds):**

```text
Metric                  Group stage     Knockout
────────────────────────────────────────────────
Log loss                X.XXX           X.XXX
Brier score             X.XXX           X.XXX
Accuracy                XX%             XX%
Draw rate (pred/actual) XX% / XX%       XX% / XX%
ECE                     X.XXX           X.XXX
```

The stage-split draw rate comparison is the primary diagnostic for
whether the stage lambda multipliers are correctly calibrated. Knockout
predicted draw rate should be higher than group stage predicted draw rate,
and both should track their respective actual rates.

### Reported Metrics

**Per-match metrics (averaged across folds):**

```text
multiclass log loss
multiclass Brier score
top-1 match accuracy
predicted vs. actual draw rate
predicted vs. actual home win rate
predicted vs. actual away win rate
mean absolute error on lambda_home
mean absolute error on lambda_away
```

**Per-tournament metrics (per fold):**

```text
champion hit (correct winner predicted as top probability)
semifinal hit count (correct 4 semifinalists in top 8 by prob)
Round of 16 hit count
group stage advancement accuracy
```

**Calibration curves (per fold):**

Calibration is evaluated separately for home win, draw, and away win
probabilities. For each outcome:

```text
1. Bin predicted probabilities into 10 equal-width buckets.
2. Compute actual outcome rate within each bucket.
3. Plot predicted probability vs. actual rate.
4. A well-calibrated model lies on the diagonal.
```

The draw calibration curve is the primary diagnostic for the Dixon-Coles
correction. If predicted draw rate is systematically below actual draw rate,
the rho parameter requires recalibration.

**Expected Calibration Error (ECE):**

```text
ECE = sum_bins( |bin_size / n| * |actual_rate - predicted_prob| )
```

ECE is reported per outcome class and per tournament stage (group vs. knockout)
to identify stage-specific miscalibration.

### Rho Validation

The Dixon-Coles rho parameter is fit on training data and its effect on
draw rate calibration is reported explicitly:

```text
Predicted draw rate (rho = 0.0):     baseline independent Poisson
Predicted draw rate (rho = fitted):  corrected
Actual draw rate:                    from holdout matches
```

---

## Assumptions

- Home and away goal counts are conditionally independent given the features,
  modulo the Dixon-Coles low-score correction.
- A Poisson count model is appropriate for football goals after feature
  adjustment.
- Difference features capture most matchup strength information.
- Historical international results improve the training sample for World Cup
  forecasting when weighted by competition importance and time proximity.
- Host advantage can be summarized by a binary host flag and neutral-site flag.
- Extra time is approximately one third of a full match in goal rate terms.
- Goal rates decline in knockout rounds and can be summarized by a scalar
  stage multiplier.
- The linear recency weighting within the form window (10:1 newest:oldest)
  is appropriate given that form matches involve the same squad over
  weeks or months, warranting softer decay than the quadratic scheme used
  for WC edition weighting.

---

## Known Weaknesses

- Independent Poisson goal models can miscalibrate low-score correlations.
  The Dixon-Coles correction partially addresses 0-0, 1-1, 0-1, and 1-0
  outcomes but does not fully model score correlation.
- V3 does not learn separate team attack and defense latent strengths.
  Difference features conflate attacking quality vs. defensive weakness.
  A Dixon-Coles or Elo-based attack/defense decomposition would address this
  but requires significant architectural change.
- Penalty shootouts are random. Weak but real evidence suggests historical
  penalty performance matters; data constraints prevent modelling this.
- Host effects are simple binary indicators. Partial home advantage for
  co-hosts (2026: Canada, Mexico, United States) is not modelled at the
  match level.
- Squad quality, injuries, market values, player minutes, and tactical
  matchups are not included.
- `wc_l5_goal_difference` has n=382 vs. n=489 for full-history features.
  The higher correlation (0.424) may be partially inflated because teams
  missing this feature are systematically weaker. The `has_wc_l5_history`
  flag partially mitigates this.
- Competition importance sample weights are assumed priors, not empirically
  optimized. Confederation-level qualifier quality differences are not
  accounted for.
- The 48-team 2026 format introduces distribution shift — no historical
  training data exists for this format. Third-place qualification incentive
  effects on late group-stage match behavior are not modelled.
- Three-tournament rolling holdout validation covers only three independent
  samples due to feature availability constraints (WC L5 features degrade
  before 2010). Metric variance remains non-trivial. The std dev across
  folds should be inspected alongside the mean.
- `v3_strength` coefficients are correlation-anchored heuristics, not
  regression-derived. The formula provides a principled ranking but does
  not have the statistical grounding of the Poisson model outputs.

---

## Potential Improvements

- **Attack/defense decomposition.** Replace difference features with explicit
  per-team attack and defense strength parameters, estimated via maximum
  likelihood over the training corpus. This is the highest-value architectural
  improvement but requires significant refactoring.
- **Full bivariate Poisson.** Replace independent Poisson with a bivariate
  model that natively handles score correlation without a post-hoc correction.
- **Tune competition sample weights.** Treat the 1.0/1.5/2.0/2.5/3.0 scheme
  as hyperparameters and optimize via rolling holdout deviance.
- **Confederation-adjusted qualifier weights.** Down-weight qualifiers from
  confederations with weaker opposition to reduce noise from low-quality
  matches.
- **Quadratic recency weighting for form window.** Current linear weighting
  adds +0.019 Spearman over uniform. Test whether quadratic (aligning with
  WC edition weighting) adds further signal. One-character code change.
- **Time-based form weighting.** Replace match-count-based linear decay with
  day-count-based exponential decay. More principled for historical data
  where match frequency varied significantly, particularly pre-1970.
- **Strength-weighted penalty probabilities.** Replace random penalty outcomes
  with a probability weighted by team strength or historical penalty data.
- **Stage or knockout conservatism as a learned feature.** The current
  `is_knockout` flag and stage multiplier handle this partially; a richer
  stage encoding (round number, cumulative match load) could improve it.
- **Calibration post-processing.** Apply Platt scaling or isotonic regression
  to output probabilities after fitting, improving calibration without
  changing the model architecture.
- **Expanded holdout.** Extend rolling holdout to include 2010 once
  sufficient pre-2010 training data is available, giving a fourth fold.
