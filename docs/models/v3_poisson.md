# V3 Poisson Model

## Purpose

V3 is the project's expected-goals model. It predicts separate expected goal
rates for the two teams in a match, converts those rates into score and
win/draw/loss probabilities, and simulates the full tournament.

V3 is the most realistic current probability model because group-table outcomes
depend on simulated goals, goal difference, and goals scored rather than on
outcome classes alone.

## Inputs

V3 uses:

- Current team metadata:
  - team id
  - display name
  - group code
  - confederation
  - Elo rating
  - host flag
- Lead-in match features:
  - result score
  - goals for
  - goals against
  - capped goal difference
  - performance versus Elo expectation
  - pre-tournament Elo
- Recent World Cup history:
  - placement feature
  - appearance feature
- Match context:
  - competition importance
  - neutral-site flag
  - net host flag
- Historical international or World Cup-only training data.
- 2026 fixtures.

The default training scope is:

```text
world_cup_only
```

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

Those lambdas drive both probability calculation and score simulation.

## Mathematical Formulas

### V3 Strength Score

V3 also computes a scalar `v3_strength` for display ordering and tie-break
fallbacks. It is not the Poisson model itself. The current formula is:

```text
v3_strength =
    elo_rating
  + 120 * results_form
  + 18 * gd_form
  + 80 * perf_vs_exp
  + 22 * goals_for
  - 18 * goals_against
  + 60 * placement
  + 8 * appearance
  + 25 * host_flag
```

This score is a stable ranking helper. Match probabilities come from the fitted
Poisson regressors.

### Match Feature Differences

For a predicted match between home-slot team `H` and away-slot team `A`, V3 uses:

```text
elo_diff = elo_H - elo_A
results_form_diff = results_form_H - results_form_A
goals_for_diff = goals_for_H - goals_for_A
goals_against_diff = goals_against_H - goals_against_A
placement_diff = placement_H - placement_A
appearance_diff = appearance_H - appearance_A
gd_form_diff = gd_form_H - gd_form_A
perf_vs_exp_diff = perf_vs_exp_H - perf_vs_exp_A
competition_importance = World Cup finals weight
neutral_site_flag = 1 if neutral, else 0
net_host_flag = host_flag_H - host_flag_A
```

The active feature columns are:

```text
elo_diff
results_form_diff
goals_for_diff
goals_against_diff
placement_diff
appearance_diff
gd_form_diff
perf_vs_exp_diff
competition_importance
neutral_site_flag
net_host_flag
```

### Weighted Form Inputs

The V3 current-team feature table uses the same weighted recent form snapshot
logic as V2:

```text
oldest selected match weight = 1
newest selected match weight = n
weighted_metric = sum(metric_i * i) / sum(i)
```

The form inputs include:

```text
results_form
gd_form
perf_vs_exp
goals_for
goals_against
pre_tournament_elo
```

### Poisson Regression

The training feature matrix is standardized with `StandardScaler`. V3 then fits:

```text
PoissonRegressor(alpha = 0.1, max_iter = 1000)
```

for home goals and away goals separately.

Poisson regression models a non-negative expected count. Conceptually:

```text
lambda = exp(intercept + beta dot x)
```

The implementation uses scikit-learn's Poisson regression. Predicted lambdas are
clipped for stability:

```text
lambda_home = clip(lambda_home, 0.05, 4.5)
lambda_away = clip(lambda_away, 0.05, 4.5)
```

### Poisson Score Probabilities

For a team with expected goals `lambda`, the Poisson probability of scoring `g`
goals is:

```text
P(G = g) = exp(-lambda) * lambda^g / g!
```

V3 computes probabilities for goals `0..10`:

```text
V3_POISSON_GOAL_CAP = 10
```

The probability mass above 10 is folded into the 10-goal bucket.

### Home, Draw, Away Probability Matrix

V3 builds a score matrix from independent home and away goal distributions:

```text
score_probability[h, a] = P(home_goals = h) * P(away_goals = a)
```

Then:

```text
home_win_prob = sum(score_probability[h, a] where h > a)
draw_prob = sum(score_probability[h, a] where h = a)
away_win_prob = sum(score_probability[h, a] where h < a)
```

The three probabilities are normalized to sum to `1.0`.

## Simulation Flow

### Training

Training data is built from either World Cup-only results or all international
results since the anchor policy. For each historical match, V3 creates
pre-match features using only information available before that match date.

Training examples use the shared competition-importance sample weights:

```text
World Cup finals = 3.0
continental finals = 2.5
qualifiers = 2.0
other competitive = 1.5
friendlies = 1.0
```

For the 2022 holdout, the training end date is the day before the first 2022
World Cup match.

### Group Stage

For each 2026 group-stage fixture:

1. Build the V3 feature row.
2. Predict `lambda_home` and `lambda_away`.
3. Sample goals:

```text
home_goals ~ Poisson(lambda_home)
away_goals ~ Poisson(lambda_away)
```

4. Award points and update goals for/against.
5. Rank the group table.

The 48-team tournament then advances the top two teams from each group plus the
best eight third-place teams.

### Knockout Stage

For knockout matches:

1. Predict lambdas.
2. Simulate regulation goals.
3. If tied, simulate extra time:

```text
extra_time_lambda = regulation_lambda * (1 / 3)
```

4. If still tied, choose a random penalty winner.

The knockout simulator caches matchup probability/lambda calculations so repeated
matchups do not need to be recomputed inside the same simulation flow.

### Host And Neutral-Site Logic

V3 has explicit host flags for:

```text
2022: Qatar
2026: Canada, Mexico, United States
```

For 2026 fixture prediction, a match is treated as neutral unless one of the two
teams has a host flag. The feature `net_host_flag` is:

```text
host_flag_home - host_flag_away
```

This lets the model account for host-side advantage when the training data
contains relevant home/non-neutral signal.

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
home_win_prob
draw_prob
away_win_prob
```

## Backtesting

The 2022 V3 backtest trains through the eve of the 2022 World Cup and then
simulates the actual 32-team tournament format. It reports:

- multiclass log loss
- multiclass Brier score
- top-1 match accuracy
- predicted versus actual draw rate
- champion hit
- semifinal hit count
- Round of 16 hit count
- match-level predicted probabilities and actual outcomes
- group and team advancement comparisons

The model card currently includes validation rows for V3 under both
`world_cup_only` and `all_international_since_anchor`.

## Assumptions

- Home and away goal counts are conditionally independent given the features.
- A Poisson count model is appropriate for football goals after feature
  adjustment.
- Difference features capture most matchup strength information.
- Historical international results can improve the training sample for World Cup
  forecasting when weighted by competition importance.
- Host advantage can be summarized by a binary host flag and neutral-site flag.
- Extra time is approximately one third of a full match.

## Known Weaknesses

- Independent Poisson goal models can miscalibrate low-score correlations such
  as `0-0` and `1-1`.
- V3 does not learn separate team attack and defense latent strengths.
- The same model handles group and knockout match behavior unless stage effects
  are represented indirectly through training data.
- Penalty shootouts are random.
- Host effects are simple binary indicators.
- Squad quality, injuries, market values, player minutes, and tactical matchups
  are not included.
- One-tournament holdout validation is not enough to fully compare V2 and V3.

## Potential Improvements

- Add a Dixon-Coles or bivariate-Poisson correction for low-score dependence.
- Add explicit attack and defense strength features.
- Add stage or knockout conservatism features at prediction time.
- Replace random penalties with strength-weighted penalty probabilities.
- Add time-decay sample weights so newer matches matter more in training.
- Tune competition sample weights through rolling holdout validation.
- Add player/squad quality features when reliable data is available.
- Evaluate calibration by probability bin, favorite/underdog status, and stage.
