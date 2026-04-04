# World Cup 2026 Dashboard

This repository contains dataset-building scripts and a Streamlit dashboard for preseason World Cup 2026 group-stage projections.

## Current Probability Simulation Logic

The dashboard in `apps/home.py` now uses a fixture-by-fixture Monte Carlo simulation for the 2026 group stage.

### Inputs

- Elo rating from `INT-World Cup/world_cup/2026/elo_snapshots.csv`
- FIFA points from `INT-World Cup/world_cup/2026/fifa_rank_snapshots.csv`
- Group-stage fixtures from `INT-World Cup/world_cup/2026/fixtures.csv`
- Lead-in results from `INT-World Cup/world_cup/2026/team_results_lead_in.csv`

### Strength Model

Each team receives a pre-tournament `team_strength` built in two stages.

Baseline rating score:

- `65%` Elo rating
- `35%` FIFA points

Both inputs are standardized with a z-score first so they are on a comparable scale.

Recent form score:

- uses each team's last `8` lead-in matches
- `points_per_match` is based on win/draw/loss points
- `goal_diff_per_match` is based on average goal difference
- `form_score = 70% * z(points_per_match) + 30% * z(goal_diff_per_match)`

Final blend:

- `team_strength = 75% * rating_score + 25% * form_score`

### Group Simulation

For each group separately, the dashboard runs `20,000` simulations.

In each simulation:

- the real six group fixtures are simulated in kickoff order
- each match uses a Poisson goal model driven by the two teams' `team_strength` values
- points, goals scored, and goals conceded are updated after every fixture
- final standings are resolved with:
  - total points
  - goal difference
  - goals scored
  - head-to-head points among tied teams
  - head-to-head goal difference among tied teams
  - head-to-head goals scored among tied teams
  - pre-tournament `team_strength` as the final deterministic fallback

### Output Probabilities

After all simulations:

- `prob_1` is the percentage of runs where the team finishes 1st
- `prob_2` is the percentage of runs where the team finishes 2nd
- `prob_3` is the percentage of runs where the team finishes 3rd
- `prob_4` is the percentage of runs where the team finishes 4th
- `ko_prob` is the percentage of runs where the team reaches the Round of 32 either via a top-two finish or as one of the eight best third-place teams

The group cards continue to show `prob_1` to `prob_4`, while the combined `All Countries` table also shows `KO %`.

### Current Limitations

The current model does not yet simulate:

- host advantage or venue-specific effects
- fair-play points or drawing of lots as final FIFA tiebreakers
- squad availability or injuries
- knockout rounds

This means the current probabilities should be interpreted as pre-tournament group-stage forecasts, not full tournament forecasts.
