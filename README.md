# World Cup 2026 Dashboard

This repository contains dataset-building scripts and a Streamlit dashboard for preseason World Cup 2026 group-stage projections.

## Current Probability Simulation Logic

The dashboard in `apps/home.py` uses a simple rating-based group finish simulation. It is not yet a fixture-by-fixture tournament model.

### Inputs

- Elo rating from `INT-World Cup/world_cup/2026/elo_snapshots.csv`
- FIFA points from `INT-World Cup/world_cup/2026/fifa_rank_snapshots.csv`

### Strength Model

Each team receives a single `strength_score` built from:

- `65%` Elo rating
- `35%` FIFA points

Both inputs are standardized with a z-score first so they are on a comparable scale.

### Group Simulation

For each group separately, the dashboard runs `20,000` simulations.

In each simulation:

- every team starts with its `strength_score`
- random Gumbel noise is added to that score
- the four teams in the group are ranked by the noisy score
- the ranking determines 1st, 2nd, 3rd, and 4th place for that run

### Output Probabilities

After all simulations:

- `prob_1` is the percentage of runs where the team finishes 1st
- `prob_2` is the percentage of runs where the team finishes 2nd
- `prob_3` is the percentage of runs where the team finishes 3rd
- `prob_4` is the percentage of runs where the team finishes 4th

These values are shown in the Streamlit group cards and combined team tables.

### Current Limitations

The current model does not yet simulate:

- actual group fixtures
- match scores or goal difference
- draws and tiebreakers
- host advantage
- squad availability, injuries, or recent form
- knockout rounds

This means the current probabilities should be interpreted as preseason rating-based finish likelihoods, not full tournament forecasts.
