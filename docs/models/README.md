# Model Documentation Index

This directory documents the modeling surfaces used by the World Cup forecasting
project. The project currently exposes five related but distinct model views:

- [V1 Team Strength](v1_team_strength.md): the original handcrafted
  team-strength Monte Carlo simulator.
- [V2 Form](v2_form.md): a diagnostic ranking model that combines ratings,
  weighted recent form, and World Cup history into one team strength score.
- [V2 Probabilities](v2_probabilities.md): a trained three-class multinomial
  match model used for tournament probability simulation.
- [V3 Poisson](v3_poisson.md): a trained expected-goals model using paired
  Poisson regressions.
- [V4 Enhanced Poisson](v4_enhanced_poisson.md): the current primary dashboard
  model, extending V3 with quadratic recent form, Dixon-Coles correction, stage
  effects, and time-decayed training weights.

The documents are written for a technical audience, but they avoid assuming
advanced statistical training. Each page explains the purpose, inputs, formulas,
simulation flow, assumptions, weaknesses, and improvement opportunities for that
model surface.

## Shared Concepts

All tournament probability models estimate outcomes for the World Cup 2026
fixture structure. They use repeated Monte Carlo simulation: run the tournament
many times, count how often each team reaches each stage, and convert those
counts into percentages.

For any event `E`, such as "team reaches the quarter-finals":

```text
P(E) = number of simulations where E occurs / total simulation count
```

The dashboard default simulation count is `20,000`, with named lower-count
options available for faster exploration. Deterministic bracket views are built
from modal group outcomes and repeated head-to-head simulations, so they are
representative single brackets rather than the full probability distribution.

## Shared Data Inputs

The models draw from these broad data groups:

- Current 2026 team metadata: team id, display name, group, confederation, Elo,
  FIFA rank/points, host flag, and World Cup history fields.
- 2026 fixture data: group stage and knockout slot definitions.
- Lead-in match data: recent pre-tournament results, goals, Elo starts, opponent
  Elo starts, and Elo deltas.
- Historical World Cup data: past placements, historical fixtures, and stage
  outcomes.
- International results data: broader match history used by all-international
  training scopes.

## Training Scopes

The trained V2, V3, and V4 models support these scopes:

```text
world_cup_only
all_international_since_anchor
```

`world_cup_only` uses World Cup finals matches from the anchor World Cup onward.
`all_international_since_anchor` uses broader international results from the
same anchor policy. For the 2022 holdout validation documented in
`docs/model_card.md`, training starts at the 1998 World Cup kickoff and ends
before the first 2022 World Cup match.

The current defaults are:

```text
V2 default training scope = world_cup_only
V3 default training scope = world_cup_only
V4 default training scope = world_cup_only
```

The trained models use this sample-weight policy:

```text
World Cup finals = 3.0
continental finals = 2.5
qualifiers = 2.0
other competitive = 1.5
friendlies = 1.0
```

## Correlation Handling

The historical EDA pages compute Pearson and Spearman correlations to study
which historical features relate to World Cup finish score. These correlations
are diagnostics. They are not direct model coefficients and they do not train
V1, V2, or V3.

The historical correlation feature names mean:

```text
form_l10_* = unweighted last-up-to-10 matches before a World Cup
weighted_form_l10_* = the same last-up-to-10 matches with linear recency weights
```

For the weighted historical form features, the oldest selected match receives
weight `1`, the newest receives weight `n`, and the weighted average is:

```text
weighted_metric = sum(metric_i * i) / sum(i)
```

V2 and V3 live model form uses the richer weighted form implementation in
`world_cup_sim/shared.py`, including result score, capped goal difference,
performance versus Elo expectation, and Elo delta. The EDA correlation table is
useful for model interpretation and feature discovery, but it is not itself the
live probability model.

V4 uses a separate quadratic recent-form implementation. For the same
last-up-to-10 match window, V4 weights the oldest-to-newest matches as:

```text
1, 4, 9, 16, 25, 36, 49, 64, 81, 100
```

This quadratic weighting is part of V4 only and does not change V2, V3, or the
historical EDA `weighted_form_l10_*` columns.

## Validation Snapshot

The current validation reference is `docs/model_card.md`, backed by the
committed artifact `data/processed/validation/aggregate_validation.json`.
That validation uses 2014, 2018, and 2022 World Cup holdout folds. It is useful
for comparing model families across multiple tournament snapshots, while the
single-fold 2022 drilldown remains available under `data/processed/validation/`.
