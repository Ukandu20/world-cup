# V4 Enhanced Poisson Model

## Purpose

V4 is the current primary dashboard model. It keeps the V3 expected-goals structure but adds the main improvements identified in `v3_poisson_updated.md`: quadratic recent-form weighting, World Cup last-5 goal-difference features, Dixon-Coles low-score correction, stage-specific lambda multipliers, time-decayed training weights, alpha cross-validation, and a rolling-backtest dashboard surface.

The 2022 holdout in `docs/model_card.md` now includes V4. V4 should still be read as a richer production-facing model, not as a universal winner on every validation metric.

## Inputs

V4 uses current 2026 team metadata, lead-in match results, 2026 fixtures, historical World Cup placement data, historical international results, Elo ratings, host flags, and match context.

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
wc_l5_goal_diff_diff
has_wc_l5_history_diff
competition_importance
neutral_site_flag
net_host_flag
is_knockout
```

The default training scope is:

```text
world_cup_only
```

## Core Method

V4 fits two regularized Poisson regressions:

```text
home_goals ~ V4 feature differences
away_goals ~ V4 feature differences
```

Features are standardized with `StandardScaler` before fitting. The regularization parameter `alpha` is selected from:

```text
0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0
```

When enough rows exist, V4 uses time-aware cross-validation and selects the alpha with the lowest combined home-plus-away mean Poisson deviance. If cross-validation cannot run, it falls back to `alpha = 0.1`.

## Mathematical Formulas

Quadratic recent form uses the last `k` matches sorted oldest to newest:

```text
recency_weight_i = i^2
weighted_metric = sum(metric_i * i^2) / sum(i^2)
```

For `k = 10`, the weights are:

```text
1, 4, 9, 16, 25, 36, 49, 64, 81, 100
```

The newest match receives `100 / 385 = 25.97%` of the total weight. This V4-only change applies to result form, goal-difference form, performance versus expectation, goals for, and goals against. Pre-tournament Elo remains the latest prior Elo value.

Training sample weights combine competition importance and time decay:

```text
final_weight = competition_weight * time_weight
time_weight = 0.5 ^ (days_before_cutoff / half_life_days)
```

The implemented half-life default is `1095` days. The configured grid is:

```text
365, 730, 1095, 1460, 2190
```

V4 converts fitted lambdas into a score probability matrix using independent Poisson probabilities, then applies the Dixon-Coles low-score correction:

```text
P(x, y) = Pois(x; lambda_home) * Pois(y; lambda_away) * tau(x, y)
```

The fitted `rho` is bounded to:

```text
-0.20 <= rho <= 0.20
```

The corrected score matrix is normalized to sum to `1.0`. `rho = 0.0` reproduces the independent Poisson matrix.

Stage multipliers are applied after lambda prediction and before clipping:

```text
lambda_stage = clip(lambda * stage_multiplier, 0.05, 4.5)
```

The goal cap is `10`.

## Simulation Flow

For each match, V4 predicts home and away expected goals, applies the stage multiplier, builds the Dixon-Coles corrected score matrix, and samples a regulation score from that matrix.

In group play, sampled scores drive points, goal difference, goals scored, group ranking, best-third qualification, and 2026 knockout routing.

In knockout play, regulation scores are sampled from the corrected matrix. If tied, extra time uses corrected lambdas scaled by:

```text
EXTRA_TIME_FACTOR = 1/3
```

If still tied, penalties use the V4 strength fallback:

```text
p_home_penalty_win = sigmoid((v4_strength_home - v4_strength_away) / 250)
```

clipped to `[0.35, 0.65]`.

## Probability Outputs

V4 returns the same tournament probability columns expected by the dashboard:

```text
prob_1, prob_2, prob_3, prob_4
ko_prob, r16_prob, qf_prob, sf_prob, final_prob, champion_prob
```

It also exposes model metadata, including selected alpha, alpha source, rho, rho source, time-decay half-life, training scope, training dates, training match count, and stage multipliers.

## Strength Score

`v4_strength` is used for display ordering and tie-break fallback only. It is not the probability model.

```text
v4_strength =
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

Non-Elo continuous inputs are normalized to `0..1` before coefficients are applied.

## Assumptions

V4 assumes that team-level historical signals, recent form, Elo strength, host context, and match-stage context are enough to produce useful pre-tournament probabilities. It does not use player injuries, squad values, tactical matchups, betting markets, or live tournament information.

Spearman correlations remain feature-selection and documentation evidence. They are not direct coefficients in V4.

## Known Weaknesses

V4 has more moving parts than V3, so it has higher overfitting risk. Dixon-Coles rho and stage multipliers are estimated from limited low-score and knockout-stage data. World Cup last-5 history can overweight old tournament pedigree for teams whose current quality has changed.

The current rolling backtest dashboard exposes 2014, 2018, and 2022 fold slots, but only the 2022 leakage-free tournament reconstruction is implemented in code. The 2014 and 2018 folds are marked as unavailable until a generic historical fixture reconstruction helper is added.

## Potential Improvements

Add complete leakage-free 2014 and 2018 tournament builders, tune the time-decay half-life in backtest utilities, compare fitted rho against `rho = 0.0`, add calibration plots by stage and outcome class, and run ablations for linear versus quadratic recency weighting.

Future model versions could add player availability, squad market value, confederation-specific calibration, travel/rest effects, and posterior uncertainty around lambdas rather than only point estimates.
