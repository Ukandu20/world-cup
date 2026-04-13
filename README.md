# World Cup 2026 Dashboard

This repository contains dataset-building scripts and a Streamlit dashboard for preseason World Cup 2026 tournament projections.

Reference notes:

- [`docs/elo_rating_reference.md`](docs/elo_rating_reference.md): stored Elo rating methodology reference based on the provided `eloratings.net` summary

## Current Probability Simulation Logic

The dashboard in `apps/home.py` now uses a fixture-by-fixture Monte Carlo simulation for the full 2026 tournament.

### Inputs

- Elo rating from `INT-World Cup/world_cup/2026/elo_snapshots.csv`
- FIFA points from `INT-World Cup/world_cup/2026/fifa_rank_snapshots.csv`
- Group-stage and knockout fixtures from `INT-World Cup/world_cup/2026/fixtures.csv`
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
