# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA Men's World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

The current dashboard primary model is V4, an enhanced Poisson expected-goals model. V4 is documented as the production-facing model because it includes the richest match-generation logic, but the validation table should be read across multiple metrics rather than as a single winner-takes-all leaderboard.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/model_validation_folds.json`. The validation window is 2014/2018/2022 FIFA World Cup folds. Each trained row uses a cutoff before the first match in its holdout World Cup.

- Match window: `10`
- Monte Carlo simulations: `20,000`
- Seed: `20260403`

### Match-Level Metrics

#### Per-Fold

| fold_year | model | scope | log_loss | brier | top1_acc | draw_pred | draw_actual | r16_hits | sf_hits | champion_hit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2014 | Elo-only baseline | all_international_since_anchor | 0.9232 | 0.5457 | 62.5% | 27.4% | 20.3% | 0 | 0 | No |
| 2014 | V2 all international since anchor | all_international_since_anchor | 1.0641 | 0.6174 | 54.7% | 26.8% | 20.3% | 11 | 2 | No |
| 2014 | V2 World Cup only | world_cup_only | 0.9903 | 0.5882 | 54.7% | 23.9% | 20.3% | 10 | 2 | No |
| 2014 | V3 all international since anchor | all_international_since_anchor | 1.0380 | 0.6161 | 56.2% | 25.2% | 20.3% | 10 | 2 | No |
| 2014 | V3 World Cup only | world_cup_only | 0.9841 | 0.5857 | 54.7% | 25.4% | 20.3% | 9 | 2 | No |
| 2014 | V4 all international since anchor | all_international_since_anchor | 1.0377 | 0.6180 | 50.0% | 26.1% | 20.3% | 8 | 2 | No |
| 2014 | V4 World Cup only | world_cup_only | 1.0069 | 0.6016 | 51.6% | 27.3% | 20.3% | 8 | 2 | No |
| 2018 | Elo-only baseline | all_international_since_anchor | 1.0008 | 0.5981 | 53.1% | 25.9% | 20.3% | 0 | 0 | No |
| 2018 | V2 all international since anchor | all_international_since_anchor | 0.9780 | 0.5813 | 53.1% | 25.8% | 20.3% | 12 | 1 | No |
| 2018 | V2 World Cup only | world_cup_only | 0.9657 | 0.5737 | 50.0% | 25.8% | 20.3% | 10 | 2 | No |
| 2018 | V3 all international since anchor | all_international_since_anchor | 0.9641 | 0.5743 | 64.1% | 25.4% | 20.3% | 11 | 1 | No |
| 2018 | V3 World Cup only | world_cup_only | 0.9613 | 0.5724 | 54.7% | 25.9% | 20.3% | 10 | 2 | No |
| 2018 | V4 all international since anchor | all_international_since_anchor | 0.9762 | 0.5814 | 60.9% | 25.5% | 20.3% | 12 | 1 | No |
| 2018 | V4 World Cup only | world_cup_only | 0.9702 | 0.5782 | 56.2% | 27.0% | 20.3% | 11 | 2 | No |
| 2022 | Elo-only baseline | all_international_since_anchor | 1.0555 | 0.6198 | 48.4% | 25.0% | 23.4% | 0 | 0 | No |
| 2022 | V2 all international since anchor | all_international_since_anchor | 1.0390 | 0.6165 | 50.0% | 24.8% | 23.4% | 10 | 2 | No |
| 2022 | V2 World Cup only | world_cup_only | 1.0292 | 0.6073 | 57.8% | 24.7% | 23.4% | 10 | 1 | No |
| 2022 | V3 all international since anchor | all_international_since_anchor | 1.0340 | 0.6172 | 53.1% | 24.6% | 23.4% | 8 | 2 | No |
| 2022 | V3 World Cup only | world_cup_only | 1.0356 | 0.6171 | 53.1% | 25.5% | 23.4% | 10 | 2 | No |
| 2022 | V4 all international since anchor | all_international_since_anchor | 1.0223 | 0.6103 | 54.7% | 25.5% | 23.4% | 9 | 2 | No |
| 2022 | V4 World Cup only | world_cup_only | 0.9965 | 0.5914 | 53.1% | 22.3% | 23.4% | 11 | 2 | No |

#### Aggregate

| model | scope | log_loss mean+/-std | brier mean+/-std | top1_acc mean+/-std | champion_hits/3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Elo-only baseline | all_international_since_anchor | 0.9932 +/- 0.0665 | 0.5879 +/- 0.0381 | 54.7% +/- 7.2% | 0/3 |
| V2 World Cup only | world_cup_only | 0.9951 +/- 0.0320 | 0.5897 +/- 0.0169 | 54.2% +/- 3.9% | 0/3 |
| V2 all international since anchor | all_international_since_anchor | 1.0270 +/- 0.0443 | 0.6051 +/- 0.0206 | 52.6% +/- 2.4% | 0/3 |
| V3 World Cup only | world_cup_only | 0.9937 +/- 0.0381 | 0.5917 +/- 0.0230 | 54.2% +/- 0.9% | 0/3 |
| V3 all international since anchor | all_international_since_anchor | 1.0120 +/- 0.0416 | 0.6025 +/- 0.0245 | 57.8% +/- 5.7% | 0/3 |
| V4 World Cup only | world_cup_only | 0.9912 +/- 0.0189 | 0.5904 +/- 0.0117 | 53.6% +/- 2.3% | 0/3 |
| V4 all international since anchor | all_international_since_anchor | 1.0121 +/- 0.0320 | 0.6032 +/- 0.0193 | 55.2% +/- 5.5% | 0/3 |

### Calibration

| model | home_win ECE | draw ECE | away_win ECE |
| --- | ---: | ---: | ---: |
| baseline_elo | 0.1367 | 0.0693 | 0.1365 |
| v2_world_cup_only | 0.1036 | 0.0872 | 0.1435 |
| v2_all_international_since_anchor | 0.1120 | 0.0660 | 0.1333 |
| v3_world_cup_only | 0.1075 | 0.0620 | 0.1041 |
| v3_all_international_since_anchor | 0.0775 | 0.0407 | 0.1115 |
| v4_world_cup_only | 0.1207 | 0.0562 | 0.1265 |
| v4_all_international_since_anchor | 0.1045 | 0.0560 | 0.1145 |

### Anomaly Flags

- Elo-only baseline (all_international_since_anchor) has elevated fold dispersion: log-loss std 0.0665, Brier std 0.0381.
- Elo-only baseline (all_international_since_anchor) has mean draw prediction 4.8 percentage points from actual.
- V2 World Cup only (world_cup_only) has elevated fold dispersion: log-loss std 0.0320, Brier std 0.0169.
- V2 World Cup only (world_cup_only) has mean draw prediction 3.5 percentage points from actual.
- V2 all international since anchor (all_international_since_anchor) has elevated fold dispersion: log-loss std 0.0443, Brier std 0.0206.
- V2 all international since anchor (all_international_since_anchor) has mean draw prediction 4.5 percentage points from actual.
- V3 World Cup only (world_cup_only) has elevated fold dispersion: log-loss std 0.0381, Brier std 0.0230.
- V3 World Cup only (world_cup_only) has mean draw prediction 4.3 percentage points from actual.
- V3 all international since anchor (all_international_since_anchor) has elevated fold dispersion: log-loss std 0.0416, Brier std 0.0245.
- V3 all international since anchor (all_international_since_anchor) has mean draw prediction 3.7 percentage points from actual.
- V4 World Cup only (world_cup_only) has mean draw prediction 4.2 percentage points from actual.
- V4 all international since anchor (all_international_since_anchor) has elevated fold dispersion: log-loss std 0.0320, Brier std 0.0193.
- V4 all international since anchor (all_international_since_anchor) has mean draw prediction 4.4 percentage points from actual.
- 2014 all_international_since_anchor: Elo-only baseline beat V4 all international since anchor on both log loss and Brier.
- 2014 world_cup_only: V2 World Cup only beat V4 World Cup only on both log loss and Brier.
- 2014 world_cup_only: V3 World Cup only beat V4 World Cup only on both log loss and Brier.
- 2018 all_international_since_anchor: V3 all international since anchor beat V4 all international since anchor on both log loss and Brier.
- 2018 world_cup_only: V2 World Cup only beat V4 World Cup only on both log loss and Brier.
- 2018 world_cup_only: V3 World Cup only beat V4 World Cup only on both log loss and Brier.

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## How To Read The Metrics

- **Log loss** rewards calibrated probabilities assigned to the actual class; lower is better.
- **Brier score** measures multiclass probability error; lower is better.
- **Top-1 accuracy** measures whether the highest-probability match outcome occurred.
- **Draw Pred./Actual** checks whether the model's average draw probability is close to the observed draw rate.
- **R16, SF, and Champion hits** evaluate tournament simulation outputs against actual holdout advancement outcomes.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference, trained on all international matches since the anchor date with tournament sample weights.
- **V2:** multinomial logistic regression using Elo, recent form, goal profile, and prior World Cup history differences. It is validated under both World-Cup-only and all-international training scopes.
- **V3:** Poisson expected-goals model using Elo, form, historical pedigree, host/neutral-site context, and competition importance. It is validated under both World-Cup-only and all-international training scopes.
- **V4:** enhanced Poisson expected-goals model using quadratic recent form, World Cup last-5 goal-difference features, Dixon-Coles low-score correction, stage multipliers, time-decayed training weights, and alpha selection. It is the current primary dashboard model and is validated under both World-Cup-only and all-international training scopes.

## Training Scopes And Weights

- `world_cup_only`: historical World Cup finals matches from the anchor World Cup onward.
- `all_international_since_anchor`: all international matches from the anchor World Cup kickoff onward.
- Sample-weight policy: `World Cup finals=3.0; continental finals=2.5; qualifiers=2.0; other competitive=1.5; friendlies=1.0`.

## Leakage Controls

- All validation rows use a cutoff before the first holdout World Cup match.
- Team features for each holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual holdout outcomes after simulation.
- 2026 forecasts use pre-tournament team metadata, fixtures, rankings, Elo snapshots, and lead-in results only.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- The rolling holdouts are useful sanity checks, not a complete validation of every tournament format.
- Penalty shootouts and extra time are simplified relative to real match dynamics.
- V4 has more components than V2/V3, so it carries higher overfitting risk until more rolling holdout folds are implemented.
