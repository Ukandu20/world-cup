# Build Historical World Cup Edition Files

## Summary
Generate per-edition folders under `INT-World Cup/world_cup/<year>/` for every completed World Cup from 1930 through 2022. Each folder will contain:
- `placement.csv`
- `schedule.csv`
- `results.csv`

Primary inputs:
- `INT-World Cup/world_cup/results.csv`
- `INT-World Cup/world_cup/world_cup_historical_tournament_formats_1930_2022.csv`

Support inputs:
- `INT-World Cup/world_cup/fifa_world_cup_history.csv`
- `INT-World Cup/world_cup/shootouts.csv`
- `INT-World Cup/world_cup/former_names.csv`

## Output Interfaces
`placement.csv`
- Columns: `country,placement,position,matches_played,gs,ga`
- `country` uses normalized names for direct renames and simple aliases, while dissolved historical entities remain distinct
- `placement` values: `Winner`, `Runner-up`, `Third Place`, `Fourth Place`, `Semi-final`, `Quarter-final`, `Round of 16`, `Group Stage`
- `position` uses stage-size encoding:
  - `1` winner
  - `2` runner-up
  - `3` third place
  - `4` fourth place
  - `8` quarter-final-equivalent exits
  - `16` round-of-16 exits
  - tournament team count for group-stage exits

`schedule.csv`
- Columns: `match_number,date,stage,home_team,away_team,city,country,neutral`
- `stage` is required on every row
- `stage` uses standardized values: `Group Stage`, `Round of 16`, `Quarter-final`, `Semi-final`, `Third Place`, `Final`

`results.csv`
- Columns: `match_number,date,stage,home_team,away_team,home_score,away_score,city,country,neutral,decided_by_shootout,shootout_winner`
- `stage` is required on every row
- `stage` uses the same standardized values as `schedule.csv`

## Implementation Changes
- Filter global `results.csv` to `tournament = FIFA World Cup`, group by year, preserve row order, and assign `match_number` within each edition.
- Add derived `stage` to both per-edition `schedule.csv` and `results.csv`.
- Normalize team names with direct rename mappings and simple aliases such as `USA -> United States` and `West Germany -> Germany`; keep dissolved historical entities such as `Czechoslovakia`, `Yugoslavia`, `Soviet Union`, and `German DR` as separate countries.
- Build per-team aggregates for each edition:
  - `matches_played`
  - `gs`
  - `ga`
- Derive `stage` and final placements by format family:
  - `1934, 1938`: straight knockout with replay matches; replay rows keep their actual knockout stage
  - `1954-1970, 1986-2022`: standard group stage plus knockout
  - `1930`: non-semi top-four are resolved from official history; remaining matches labeled `Group Stage`, `Semi-final`, `Final`
  - `1950`: first phase labeled `Group Stage`; final-round group matches also labeled `Final` for standardized output
  - `1974, 1978`: first phase labeled `Group Stage`; second group stage also labeled `Quarter-final` for standardized output; title/bronze matches keep `Final` and `Third Place`
  - `1982`: first phase labeled `Group Stage`; second group stage labeled `Quarter-final`; later rounds use standard knockout labels
- Use `shootouts.csv` to populate `decided_by_shootout` and `shootout_winner` where applicable.

## Test Plan
- Each generated edition `results.csv` row count matches `Matches_Played` in `fifa_world_cup_history.csv`.
- Each generated edition `placement.csv` row count matches `Teams` in `fifa_world_cup_history.csv`.
- `schedule.csv` and `results.csv` have identical match counts and identical `stage` values row-for-row.
- Sum of team `matches_played` equals `2 * official match count`.
- Sum of team `gs` equals official tournament goals, and sum of `ga` equals the same value.
- Top 4 placements match `fifa_world_cup_history.csv` after alias normalization.
- Spot-check irregular years: `1930`, `1950`, `1974`, `1978`, `1982`, and a penalty-shootout edition such as `2002` or `2022`.

## Assumptions
- Output path is `INT-World Cup/world_cup/<year>/`.
- `stage` is included in both generated `schedule.csv` and generated `results.csv`.
- Standardized stage labels are preferred over historically exact labels for irregular editions.
- `position` uses stage-size encoding, not tied ordinal ranks.
