# V2 Probabilities Model

## Purpose

V2 Probabilities is the trained match-outcome model for the V2 family. It
predicts three regulation-time-style outcome classes for each matchup:

```text
home_win
draw
away_win
```

It then uses those probabilities, empirical scoreline distributions, and the
real tournament fixture structure to simulate group tables and knockout
progression.

V2 is more statistical than V1 because it trains a multinomial logistic
regression on historical match-level examples.

## Inputs

V2 uses:

- Current team metadata and Elo ratings.
- Lead-in match data for weighted recent form.
- Recent World Cup history features.
- Historical World Cup results, or broader international results depending on
  training scope.
- 2026 group and knockout fixtures.

The default training scope is:

```text
world_cup_only
```

The available scopes are:

```text
world_cup_only
all_international_since_anchor
```

## Core Method

V2 has two related pieces:

1. A current-team feature table, built from Elo, recent form, and history.
2. A trained multinomial logistic regression model for match outcomes.

The model predicts outcome probabilities. It does not directly predict goals.
When scorelines are needed for group tables, V2 samples historical empirical
scorelines conditional on the sampled outcome class.

## Mathematical Formulas

### Match Feature Differences

For a match between home-slot team `H` and away-slot team `A`, V2 creates
difference features:

```text
elo_diff = elo_H - elo_A
results_form_diff = results_form_H - results_form_A
gd_form_diff = gd_form_H - gd_form_A
perf_vs_exp_diff = perf_vs_exp_H - perf_vs_exp_A
goals_for_diff = goals_for_H - goals_for_A
goals_against_diff = goals_against_H - goals_against_A
placement_diff = placement_H - placement_A
appearance_diff = appearance_H - appearance_A
```

These are the active feature columns:

```text
elo_diff
results_form_diff
gd_form_diff
perf_vs_exp_diff
goals_for_diff
goals_against_diff
placement_diff
appearance_diff
```

The form features use the weighted last-k logic described in
[V2 Form](v2_form.md).

### Multinomial Logistic Regression

The training data matrix is standardized with `StandardScaler`. The scaled
feature vector is passed to scikit-learn `LogisticRegression` with:

```text
solver = lbfgs
C = 1.0
max_iter = 5000
random_state = 20260403
```

Conceptually, the model estimates one score per class:

```text
score_c = intercept_c + beta_c dot x
```

Those scores are converted to probabilities with softmax:

```text
P(class c) = exp(score_c) / sum_j exp(score_j)
```

The output probabilities sum to `1.0`:

```text
P(home_win) + P(draw) + P(away_win) = 1
```

### Sample Weights

Training examples are weighted by competition importance:

```text
World Cup finals = 3.0
continental finals = 2.5
qualifiers = 2.0
other competitive = 1.5
friendlies = 1.0
```

This makes higher-stakes matches count more in the fitted model.

### Empirical Scoreline Distributions

V2 builds historical scoreline samplers by:

```text
stage bucket: group or knockout
outcome class: home_win, draw, away_win
```

For a sampled group-stage outcome, V2 samples a real historical scoreline from
the matching bucket and outcome class. If a bucket has no examples, it falls
back to the available examples for the outcome, and then to a default scoreline.

Default scorelines are:

```text
home_win = 1-0
draw = 0-0
away_win = 0-1
```

## Simulation Flow

### Group Stage

For each group-stage fixture:

1. Build the home-away feature difference row.
2. Predict `home_win`, `draw`, and `away_win` probabilities.
3. Sample an outcome for each Monte Carlo run.
4. Sample a scoreline from the empirical distribution for that outcome.
5. Update group points, goals for, and goals against.

Groups are ranked with the shared group ranking function. The 2026 format then
advances:

```text
top 2 teams from each group
best 8 third-place teams
```

The best-third routing map resolves the Round of 32 bracket slots.

### Knockout Stage

For knockout fixtures, V2 again predicts the three outcome probabilities. If the
sampled outcome is `home_win` or `away_win`, the winner is immediate.

If the sampled outcome is `draw`, the model treats the match as level before
penalties and resolves the winner using the non-draw split:

```text
P(home wins shootout proxy) =
    P(home_win) / (P(home_win) + P(away_win))
```

If both non-draw probabilities are zero, penalties fall back to a random winner.

This is not a true penalty model. It is a pragmatic way to avoid randomizing all
drawn knockout matches equally.

### Deterministic Bracket

The V2 deterministic bracket uses modal group rankings, best-third routing, and
repeated head-to-head simulations to produce one stable presentation bracket.

## Probability Outputs

V2 outputs:

- group finish probabilities: `prob_1`, `prob_2`, `prob_3`, `prob_4`
- best-third qualification: `top8_third_prob`
- knockout qualification: `ko_prob`
- reaching Round of 16: `r16_prob`
- reaching quarter-finals: `qf_prob`
- reaching semi-finals: `sf_prob`
- reaching final: `final_prob`
- winning the tournament: `champion_prob`

Each probability is:

```text
count / simulation_count * 100
```

## Backtesting

The 2022 holdout backtest trains with 2022 excluded and cuts training data off
before the tournament starts. It reports:

- multiclass log loss
- multiclass Brier score
- top-1 match accuracy
- champion hit
- semifinal hit count
- Round of 16 hit count
- match predictions
- group finish backtest
- team advancement backtest

The current model card reports both `world_cup_only` and
`all_international_since_anchor` V2 validation rows.

## Assumptions

- The three-class outcome structure is sufficient for group-stage simulation.
- Historical scoreline distributions are a reasonable way to attach scores to
  predicted outcomes.
- Difference features capture the relevant matchup information.
- Competition-importance weights improve training signal quality.
- Prior World Cup history adds useful context beyond current Elo and form.

## Known Weaknesses

- V2 predicts outcome classes, not goal rates.
- Empirical scoreline sampling can inherit historical scoring patterns that may
  not match 2026 conditions.
- Home/away slot labels are not true home advantage in most World Cup matches.
- Feature effects are linear in the logistic regression after scaling.
- The model does not explicitly separate team attack and defense.
- Penalty handling is approximate.
- One-tournament holdout validation is useful but not enough for final model
  selection.

## Potential Improvements

- Add rolling World Cup holdouts across multiple editions.
- Add stage-specific features or separate group and knockout models.
- Replace empirical scoreline sampling with a calibrated goal model.
- Add squad quality, market value, player availability, or club-strength
  features.
- Tune sample weights and recency weights through validation rather than fixed
  defaults.
- Compare calibration curves for favorites, underdogs, and draw probabilities.
