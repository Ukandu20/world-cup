# Build a Normalized FIFA World Cup 2026 Dataset With Team Lead-In Results

## Summary
- Create a curated output under `INT-World Cup/world_cup/2026/` built from official 2026 FIFA sources plus the local international results base.
- Keep normalized tables for `teams`, `groups`, `rounds`, `venues`, `fixtures`, `fifa_rank_snapshots`, `elo_snapshots`, `edition_metadata`, `editions_summary`, `squads`, and add a qualified-team `team_results_lead_in` table.
- Use dual team identity on relevant tables: stable canonical name plus source/tournament-era label.
- Rankings remain freeze-aware: provisional before the final pre-tournament release exists, then lock to the final pre-tournament snapshot.

## Interface Changes
- `teams.csv`: `team_id`, `canonical_name`, `tournament_name`, `fifa_code`, `confederation`, `group_code`, `is_host`, `qualification_path`, `world_cup_participations`, `squad_status`, `source_url`, `source_as_of`.
- `fixtures.csv`: official 2026 match schedule and results/status with stable foreign keys to rounds, groups, venues, and teams.
- `fifa_rank_snapshots.csv` and `elo_snapshots.csv`: one row per qualified team per tournament freeze snapshot, with `snapshot_date`, `freeze_target_date`, and `is_final_freeze`.
- `team_results_lead_in.csv`: every recorded international match for each 2026 qualified team up to `2026-06-10`, regardless of competition. Include match metadata from `data/results.csv`, canonical/source team fields, outcome fields, and optional joins to goalscorers/shootouts where keys match.
- `edition_metadata.csv`, `editions_summary.csv`, and `squads.csv`: 2026 edition record, historical edition summary through 2026, and an empty squad scaffold until official squads are published.

## Implementation Changes
- Source 2026 tournament structure from official FIFA pages for teams, groups, fixtures, venues, and official status/results.
- Build a canonical team mapping layer from `former_names.csv` plus a small football-specific override map for cases like `Germany/West Germany`, `Russia/Soviet Union`, `Serbia/Yugoslavia`, `South Korea/Korea Republic`, `IR Iran/Iran`, and `USA/United States`.
- Derive `world_cup_participations` from `data/results.csv` filtered to `FIFA World Cup`, using canonical identity rules rather than raw source names.
- Build `team_results_lead_in.csv` by filtering `data/results.csv` to matches dated `<= 2026-06-10` where either side is one of the 48 qualified teams. Preserve both original match labels and canonicalized team ids so the table is analysis-safe.
- Add optional companion enrichment for lead-in matches:
  - goals from `data/goalscorers.csv`
  - shootout outcomes from `data/shootouts.csv`
  - derived fields such as `result_for_team`, `goal_difference_for_team`, `is_home_perspective`, and `days_before_tournament`
- Use FIFA rankings with the chosen policy: store the official 1 April 2026 ranking as provisional now, then refresh and mark the 10 June 2026 release as the final pre-tournament freeze.
- Use Elo similarly: current daily snapshot until the tournament freeze date exists, then refresh and lock to the final pre-tournament daily Elo snapshot.

## Test Plan
- Validate tournament structure counts: 48 teams, 12 groups, 104 fixtures, 16 venues, and correct round counts.
- Validate identity normalization on known historical edge cases so participation counts and lead-in rows do not split the same football entity incorrectly.
- Validate lead-in extraction:
  - every row is dated `<= 2026-06-10`
  - every row includes at least one qualified team
  - qualified-team totals reconcile against raw `data/results.csv`
- Validate ranking coverage: every qualified team has one FIFA snapshot and one Elo snapshot, with provisional/final freeze flags behaving correctly.
- Validate fixture integrity: all foreign keys resolve, scheduled matches stay unresolved where appropriate, and official results can later replace placeholders without breaking joins.
- Validate historical edition summary against the existing World Cup summary/champions files.

## Assumptions And Defaults
- Output root: `INT-World Cup/world_cup/2026/`.
- Lead-in window means all prior recorded internationals for the 48 qualified teams up to the pre-tournament cutoff, not just recent form.
- Pre-tournament cutoff date is `2026-06-10`, the day before the tournament begins on `2026-06-11`.
- FIFA rankings are release-based rather than daily; the build must support a provisional snapshot now and a final pre-tournament refresh later.
- Squads are not yet available, so only the schema and team-level squad status are populated for now.
