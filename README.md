# World Cup 2026 Dashboard

This repository contains dataset-building scripts and a Streamlit dashboard for preseason World Cup 2026 tournament projections.

## Setup

Install the dashboard and dataset dependencies with:

```bash
python -m pip install -r requirements.txt
```

The app-ready dataset is committed under `data/processed/`, so a clean clone can run without your local Kaggle download cache:

```bash
streamlit run apps/home.py
```

Raw Kaggle/source downloads remain ignored and rebuild-only. To refresh the local Kaggle raw files used by the builders:

```bash
python scripts/bootstrap_kaggle_data.py
```

See [`data/README.md`](data/README.md) for the data layout and environment-variable overrides.

Reference notes:

- [`docs/elo_rating_reference.md`](docs/elo_rating_reference.md): stored Elo rating methodology reference based on the provided `eloratings.net` summary

## Validation Summary

The published model validation is a 2022 World Cup holdout using `20,000` simulations, match window `10`, and seed `20260403`. See the full [`model card`](docs/model_card.md) and the reproducible artifact at [`data/processed/validation/model_validation_2022.json`](data/processed/validation/model_validation_2022.json).

| Model | Scope | Matches | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo-only baseline | all_international_since_anchor | 22,843 | 1.0545 | 0.6358 | 42.2% | 24.1% / 23.4% | 0 | 0 | No |
| V2 World Cup only | world_cup_only | 384 | 1.0170 | 0.5991 | 51.6% | 24.0% / 23.4% | 10 | 1 | Yes |
| V2 all international since anchor | all_international_since_anchor | 22,843 | 1.0124 | 0.6004 | 48.4% | 22.0% / 23.4% | 9 | 2 | No |
| V3 World Cup only | world_cup_only | 384 | 1.0204 | 0.6050 | 51.6% | 24.5% / 23.4% | 9 | 1 | No |
| V3 all international since anchor | all_international_since_anchor | 22,843 | 1.0210 | 0.6033 | 50.0% | 21.6% / 23.4% | 10 | 2 | No |

## Current Probability Simulation Logic

The dashboard in `apps/home.py` now uses a fixture-by-fixture Monte Carlo simulation for the full 2026 tournament.

### Inputs

- Elo rating from `data/processed/world_cup/2026/elo_snapshots.csv`
- FIFA points from `data/processed/world_cup/2026/fifa_rank_snapshots.csv`
- Group-stage and knockout fixtures from `data/processed/world_cup/2026/fixtures.csv`
- Lead-in results from `data/processed/world_cup/2026/team_results_lead_in.csv`

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

### Group and Knockout Simulation

By default, the dashboard runs `100,000` simulations per group.

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

After the group stage in each simulation:

- the top two teams in each group qualify automatically for the Round of 32
- the 12 third-placed teams are ranked by points, goal difference, goals scored, then `team_strength`
- the best eight third-placed teams qualify
- the Round of 32 bracket is routed using the published 2026 knockout-stage combinations table
- knockout matches from the Round of 32 through the final use the same strength-driven Poisson goal model
- tied knockout matches go to extra time using one-third of regulation expected goals, then to a 50/50 penalty shootout if still level

### Deterministic Bracket Logic

The dashboard also builds one stable predicted bracket from the Monte Carlo output.

- each group stores its most common full finishing order across all simulations as `modal_group_rankings`
- the bracket uses those modal group finishers, not the display sort from the probability tables
- the 12 modal third-placed teams are ranked again by average simulated third-place points, goal difference, goals scored, then `team_strength`
- the best eight third-placed groups are mapped into the Round of 32 using the fixed 2026 third-place routing combinations
- every knockout matchup in that bracket is then simulated head-to-head multiple times to estimate the likely winner and win percentage

This means the bracket is position-based: it follows projected group winners, runners-up, and qualifying third-place teams through the official knockout slots rather than taking a global top-N list of teams.

### Output Probabilities

After all simulations:

- `prob_1` is the percentage of runs where the team finishes 1st
- `prob_2` is the percentage of runs where the team finishes 2nd
- `prob_3` is the percentage of runs where the team finishes 3rd
- `prob_4` is the percentage of runs where the team finishes 4th
- `top8_third_prob` is the percentage of runs where the team qualifies as one of the eight best third-placed teams
- `ko_prob` is the percentage of runs where the team reaches the Round of 32 either via a top-two finish or as one of the eight best third-place teams
- `r16_prob` is the percentage of runs where the team reaches the Round of 16
- `qf_prob` is the percentage of runs where the team reaches the quarter-finals
- `sf_prob` is the percentage of runs where the team reaches the semi-finals
- `final_prob` is the percentage of runs where the team reaches the final
- `champion_prob` is the percentage of runs where the team wins the tournament

The `Single group` and `All groups` views now use the bracket-aligned `Projected Order` by default, so those tables match the modal group rankings that feed the deterministic bracket. The combined `All Countries` table also shows `Top 8 3rd %`, `KO %`, `R16 %`, `QF %`, `SF %`, `Final %`, and `Champion %`.

Each table card and the bracket card display the simulation count used for that render.

### Current Limitations

The current model does not yet simulate:

- host advantage or venue-specific effects
- fair-play points or drawing of lots as final FIFA tiebreakers
- squad availability or injuries

This means the current probabilities should still be interpreted as pre-tournament forecasts rather than match-specific predictions.

## V2 Multinomial Probabilities

The dashboard also includes separate `V2 Form`, `V2 Probabilities`, and `V2 2022 Backtest` pages.

### Current V2 Model Card

Model type:

- multinomial logistic regression for match outcomes
- output classes: `home_win`, `draw`, `away_win`
- tournament probabilities are produced by Monte Carlo simulation on top of those match probabilities

Training window:

- uses the previous `5` completed World Cup editions
- includes both group-stage and knockout matches
- the `2022` backtest excludes `2022` from training, so it uses the previous `5` editions before that

V2 team-strength inputs:

- rating block: currently `Elo-only` in practice because `BASELINE_RATING_WEIGHTS = (1.0, 0.0)`
- weighted recent form block:
  - `results_form`
  - `gd_form`
  - `perf_vs_exp`
  - `elo_delta_form`
- World Cup history block from the previous `5` editions:
  - weighted placement score
  - weighted participation ratio

V2 team-strength blend:

- `40%` rating
- `40%` weighted recent form
- `20%` World Cup history

History sub-weights:

- `70%` weighted placement score
- `30%` weighted participation ratio

Weighted form sub-weights:

- `40%` results
- `25%` goal difference
- `25%` performance versus Elo expectation
- `10%` Elo delta

Match-model feature set:

- `elo_diff`
- `results_form_diff`
- `gd_form_diff`
- `perf_vs_exp_diff`
- `goals_for_diff`
- `goals_against_diff`
- `placement_diff`
- `appearance_diff`

Behavior:

- `V2 Form` exposes the history-aware team ranking surface
- `V2 Probabilities` uses the multinomial match model to simulate the 2026 tournament
- `V2 2022 Backtest` evaluates the same model family on the real 2022 tournament with `2022` held out from training

The original `V1 Probabilities` page remains unchanged.
