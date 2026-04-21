# V2 Weighted World Cup History Features

## Summary
- Add two new World Cup history features for each 2026 qualified team: `weighted_world_cup_placement_score` and `weighted_world_cup_participations`.
- Compute both across every World Cup edition from `1930` through `2022` using quadratic edition weights `w_i = (i + 1)^2` in chronological order.
- Count non-qualification as `DNQ = 0` for the placement aggregate, and keep the current V1 simulator unchanged; the new history component is V2-only.

## Implementation Changes
- In `scripts/build_world_cup_2026_dataset.py`, replace the raw appearance-only helper with a history-feature builder that:
  - loads `INT-World Cup/world_cup/all_editions/placement.csv` and `INT-World Cup/world_cup/fifa_world_cup_history.csv`;
  - reuses the existing alias/former-name canonicalization so current qualified teams map cleanly onto historical records;
  - builds one per-team, per-edition timeline for all editions, filling missing editions as `qualified=False`, `rank=None`, `n_teams=<history teams>`, and placement score `0.0`.
- Add a placement-score helper with defaults `epsilon=0.05` and `gamma=0.8`:
  - `DNQ -> 0.0`
  - qualified teams use `epsilon + (1 - epsilon) * (1 - ((rank - 1) / (n_teams - 1)) ** gamma)`
  - winner returns `1.0`
- Persist these columns into `teams.csv`:
  - keep existing `world_cup_participations` as the raw integer count for backward compatibility;
  - add `weighted_world_cup_participations` as the raw quadratic-weighted appearance sum;
  - add `weighted_world_cup_placement_score` as the weighted mean of per-edition placement scores over all editions, including DNQ zeros.
- Update `apps/home.py` loading so `base_df` retains the new history columns from `teams.csv`.
- In `world_cup_simulation.py`, add a V2 strength path that:
  - reuses the current baseline rating component unchanged;
  - reuses the current weighted-form composite as the V2 form component;
  - builds a history component as `0.7 * weighted_world_cup_placement_score + 0.3 * weighted_world_cup_participation_ratio`, where `weighted_world_cup_participation_ratio = weighted_world_cup_participations / total_edition_weight`;
  - blends `rating/form/history` as `40% / 40% / 20%`;
  - leaves the existing V1 `build_team_strengths` and `simulate_probabilities` behavior untouched.
- Extend the current V2 page so it ranks teams by the new V2 composite and displays the two history features alongside the existing weighted-form metrics.

## Public Interfaces
- `teams.csv` gains:
  - `weighted_world_cup_participations`
  - `weighted_world_cup_placement_score`
- The V2 table output gains:
  - the two new history columns
  - a final V2 composite strength/rank column
- No breaking changes to V1 simulator inputs or defaults.

## Test Plan
- Unit-test the placement-score helper for `DNQ`, winner, last-place qualifier, monotonicity by `rank`, and varying `n_teams`.
- Unit-test the history aggregation for quadratic weights, all-editions DNQ handling, raw-count preservation, and no duplicate edition counting.
- Unit-test historical name/alias resolution for current teams with renamed historical entries.
- Add V2 model tests to verify:
  - new columns load from `teams.csv`
  - the `40/40/20` top-level blend and `70/30` history mix are applied
  - V1 defaults and probability outputs remain unchanged.

## Assumptions
- Default `gamma` is `0.8`.
- Quadratic recency weights use global World Cup edition order, not per-team appearance order.
- The current rating component stays as-is; this change does not retune Elo/FIFA weighting.
- “V2-only” means the history-aware composite is exposed on the current V2 page, while the existing V1 probabilities page remains the default simulator until a separate V2 simulation surface is added.
