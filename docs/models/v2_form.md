# V2 Form Model

## Purpose

V2 Form is a team ranking and diagnostic surface. It does not directly simulate
tournament probabilities. Instead, it builds a richer pre-tournament strength
score that combines:

- baseline team rating
- weighted recent form
- recent World Cup history

The dashboard uses this page to inspect why teams are rated strongly or weakly
before moving to the trained V2 probability model.

## Inputs

V2 Form uses:

- Current team metadata:
  - `team_id`
  - `display_name`
  - `group_code`
  - `confederation`
  - `elo_rating`
  - `fifa_points`
  - `world_rank`
  - World Cup history fields
- Lead-in match data with Elo fields:
  - team score
  - opponent score
  - result
  - team Elo start
  - opponent Elo start
  - team Elo delta
- Historical World Cup placement data for prior-edition pedigree.

The default match window is:

```text
k = 10
```

## Core Method

V2 Form creates three normalized components:

```text
rating component
weighted form component
history component
```

Then it blends them into:

```text
v2_strength_index_0to1
v2_strength
```

The model is a deterministic feature-engineering pipeline, not a trained
regression model.

## Mathematical Formulas

### Rating Component

The rating component combines Elo and FIFA points. The default remains Elo-only:

```text
rating_score = 1.0 * zscore(elo_rating) + 0.0 * zscore(fifa_points)
```

The result is scaled to a bounded index:

```text
rating_index_0to1 = scale_to_range(rating_score, 0, 1, neutral = 0.5)
```

### Linear Recency Weighting

For each team, matches are sorted from oldest to newest and the last `k` are
selected. If `n` matches are available, weights are:

```text
oldest selected match = 1
newest selected match = n
```

For any metric:

```text
weighted_metric = sum(metric_i * i) / sum(i)
```

For `k = 10`, the weights are:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

The newest match contributes `10 / 55 = 18.18%` of the weighted average, while
the oldest contributes `1 / 55 = 1.82%`.

### Weighted Recent Form Components

For each selected match:

```text
actual_score = 1.0 for win, 0.5 for draw, 0.0 for loss
goal_difference = team_score - opponent_score
gd_capped = clip(goal_difference, -4, 4)
```

Elo expected score is:

```text
expected_score = 1 / (1 + 10 ^ ((opponent_elo_start - team_elo_start) / 400))
```

Performance over expectation is:

```text
perf_vs_exp = actual_score - expected_score
```

The weighted form table computes:

```text
results_form = weighted_average(actual_score)
gd_form = weighted_average(gd_capped)
perf_vs_exp = weighted_average(actual_score - expected_score)
elo_delta_form = weighted_average(team_elo_delta)
```

### Component Scaling

Each component is converted to a `0..1` score:

```text
results_score = clip(results_form, 0, 1)
gd_score = scale gd_form from [-4, 4] to [0, 1]
perf_score = scale perf_vs_exp from [-0.5, 0.5] to [0, 1]
elo_score = scale elo_delta_form from [-15, 15] to [0, 1]
```

The default weighted form blend is:

```text
form_index_0to1 =
    0.40 * results_score
  + 0.25 * gd_score
  + 0.25 * perf_score
  + 0.10 * elo_score
```

It is converted to a `1..10` display score:

```text
form = 1 + 9 * form_index_0to1
```

### World Cup History Component

V2 Form uses recent World Cup history over the previous five editions by
default:

```text
V2_PREVIOUS_EDITION_LOOKBACK = 5
```

The history component combines:

- weighted placement score
- weighted participation ratio

The default history blend is:

```text
history_score =
    0.70 * weighted_world_cup_placement_score
  + 0.30 * weighted_world_cup_participation_ratio
```

Non-qualification editions are scored as zero in the placement aggregate.

### V2 Composite Strength

The top-level blend is:

```text
v2_strength_index_0to1 =
    0.40 * rating_index_0to1
  + 0.40 * form_index_0to1
  + 0.20 * history_score
```

The display strength is:

```text
v2_strength = 1 + 9 * v2_strength_index_0to1
```

## Simulation Flow

V2 Form does not run a tournament simulation. Its flow is:

1. Load current teams and lead-in results.
2. Compute rating index.
3. Compute weighted recent form.
4. Compute recent World Cup history features.
5. Blend the three components into `v2_strength`.
6. Sort teams by V2 strength, form, Elo, and world rank.

The V2 Form dashboard provides controls for:

- simulation label, used for consistent export metadata
- last-k match window
- result weight
- goal-difference weight
- performance-over-expectation weight
- Elo-delta weight
- view mode by all countries or confederation

The form component weights are normalized internally, so slider values behave as
relative weights.

## Probability Outputs

V2 Form does not produce tournament probabilities. It produces ranking and
diagnostic outputs:

- `rating_index_0to1`
- `results_form`
- `gd_form`
- `perf_vs_exp`
- `elo_delta_form`
- `form`
- `history_score`
- `v2_strength`

Tournament probabilities for the V2 family come from the separate
[V2 Probabilities](v2_probabilities.md) model.

## Assumptions

- Recent matches are more informative than older matches within the same window.
- Last-k form is summarized well by results, goal difference, performance versus
  Elo expectation, and Elo movement.
- The fixed `40/40/20` blend between rating, form, and history is reasonable for
  ranking teams before the tournament.
- Prior World Cup participation and placement are useful pedigree signals, even
  though teams and player pools change across editions.

## Known Weaknesses

- The top-level weights are handcrafted, not learned from a validation objective.
- The form score can be sensitive to the chosen match window.
- The same recency schedule is used for every team, regardless of match spacing
  or opponent quality beyond Elo expectation.
- World Cup history can reward national-program pedigree even when the current
  squad is very different.
- It does not directly estimate match probabilities or simulate bracket paths.

## Potential Improvements

- Tune the `40/40/20` and `40/25/25/10` blends using rolling historical
  validation.
- Add opponent-strength-adjusted attacking and defensive form splits.
- Use time-decay by days instead of match-order-only recency.
- Add uncertainty bands for teams with fewer usable lead-in matches.
- Compare V2 Form rankings against V2 and V3 simulated stage probabilities to
  identify teams whose bracket path differs from underlying strength.
