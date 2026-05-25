# V1 Team Strength Model

## Purpose

V1 is the original tournament simulator. It converts pre-tournament team ratings
and simple recent form into a scalar `team_strength`, maps strength differences
to expected goals, and runs a fixture-by-fixture Monte Carlo simulation of the
World Cup 2026 tournament.

V1 is intentionally simple. It is useful as a transparent baseline because every
major assumption is explicit: rating strength, recent form, expected goals, group
table ranking, and knockout resolution.

## Inputs

V1 uses three main inputs:

- Team metadata from the 2026 teams table:
  - `team_id`
  - `display_name`
  - `group_code`
  - `elo_rating`
  - `fifa_points`
  - `world_rank`
- Lead-in match data:
  - result
  - goal difference
  - match date
  - qualified team id
- Fixture data:
  - group-stage matchups
  - knockout slot definitions

The default recent match window is:

```text
k = 10
```

## Core Method

V1 has three modeling layers:

1. Build each team's pre-tournament strength.
2. Use strength differences to simulate match scores.
3. Aggregate simulated tournament outcomes into probabilities.

It does not train a statistical model from historical match outcomes. Instead,
it uses handcrafted formulas and Monte Carlo sampling.

## Mathematical Formulas

### Rating Score

V1 first creates a rating score from Elo and FIFA points. The current default is
Elo-only:

```text
rating_score = 1.0 * zscore(elo_rating) + 0.0 * zscore(fifa_points)
```

The z-score transformation centers and scales the column across teams:

```text
zscore(x_i) = (x_i - mean(x)) / std(x)
```

This makes rating values comparable to form values.

### Simple Recent Form

For each team, V1 takes the last `k` lead-in matches and computes:

```text
points_per_match = average match points
goal_diff_per_match = average goal difference
```

Match points use:

```text
win = 3
draw = 1
loss = 0
```

Both components are z-scored across teams:

```text
points_form_z = zscore(points_per_match)
goal_diff_form_z = zscore(goal_diff_per_match)
```

The default V1 form blend is:

```text
form_score = 0.70 * points_form_z + 0.30 * goal_diff_form_z
```

### Team Strength

The final V1 team strength combines rating and form:

```text
team_strength = 0.50 * rating_score + 0.50 * form_score
```

This strength can be negative because it is based on standardized values.
Higher is better.

### Strength To Expected Goals

For a match between a home-slot team and an away-slot team:

```text
delta = home_team_strength - away_team_strength
home_xg = clip(1.20 + 0.40 * delta, 0.20, 3.00)
away_xg = clip(1.20 - 0.40 * delta, 0.20, 3.00)
```

The constants are:

```text
EXPECTED_GOALS_BASE = 1.20
EXPECTED_GOALS_SCALE = 0.40
EXPECTED_GOALS_MIN = 0.20
EXPECTED_GOALS_MAX = 3.00
```

The `clip` operation prevents extreme strength gaps from producing unrealistic
expected-goal values.

### Poisson Score Simulation

Goals are sampled from independent Poisson distributions:

```text
home_goals ~ Poisson(home_xg)
away_goals ~ Poisson(away_xg)
```

The Poisson assumption means a team's goal count is a non-negative integer whose
average rate is the expected-goals value.

## Simulation Flow

### Group Stage

For each group:

1. Extract the six group fixtures.
2. Compute expected goals for every fixture.
3. Simulate all fixture scores for each Monte Carlo run.
4. Award points:
   - win = 3
   - draw = 1
   - loss = 0
5. Rank teams using group-table rules.

The group ranking function considers points, goal difference, goals for,
head-to-head information, and strength-based fallbacks where needed.

### Best Third-Place Qualification

World Cup 2026 has 12 groups of 4 teams. V1 advances:

```text
top 2 teams from each group = 24 teams
best 8 third-place teams = 8 teams
total knockout teams = 32 teams
```

Third-place teams are ranked by their simulated group-table records. A static
third-place routing map assigns the qualifying third-place groups to the correct
Round of 32 fixture slots.

### Knockout Stage

For each knockout match:

1. Simulate regulation goals from the same Poisson expected-goals logic.
2. If tied, simulate extra-time goals.
3. If still tied, resolve penalties randomly.

Extra time uses:

```text
EXTRA_TIME_FACTOR = 1 / 3
```

So extra-time expected goals are one third of regulation expected goals.

### Deterministic Bracket

The deterministic bracket is not a separate probability model. It uses:

- modal group rankings from the simulation output
- average third-place statistics
- repeated head-to-head simulations for each knockout matchup

It produces one stable bracket view for presentation.

## Probability Outputs

V1 outputs percentages for:

- group finish positions: `prob_1`, `prob_2`, `prob_3`, `prob_4`
- best-third qualification: `top8_third_prob`
- knockout qualification: `ko_prob`
- reaching Round of 16: `r16_prob`
- reaching quarter-finals: `qf_prob`
- reaching semi-finals: `sf_prob`
- reaching final: `final_prob`
- winning the tournament: `champion_prob`

For any output stage:

```text
stage_probability = stage_count / simulation_count * 100
```

## Assumptions

- Elo is the only active rating input by default.
- Recent form is summarized only through points and goal difference.
- Goals are independent Poisson draws.
- Strength differences map linearly to expected goals before clipping.
- Penalty shootouts are random.
- The model does not know player quality, injuries, tactics, travel, rest, or
  squad announcements.

## Known Weaknesses

- V1 is not statistically trained against historical match outcomes.
- It has no explicit draw model; draws arise only from matching Poisson score
  samples.
- It does not separate attack and defense.
- It ignores opponent strength inside the simple form calculation except through
  the Elo values already embedded in `team_strength`.
- Random penalties can understate stronger teams in knockout ties.
- The linear strength-to-xG mapping is easy to explain but not calibrated from
  data.

## Potential Improvements

- Calibrate the expected-goals mapping from historical match outcomes.
- Replace random penalties with Elo- or model-weighted penalty probabilities.
- Split team strength into attack and defense components.
- Add match context such as stage, rest days, travel, and host effects.
- Compare V1 against V2 and V3 over rolling historical World Cup holdouts.
