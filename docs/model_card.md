# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA Men's World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

The current dashboard primary model is V4, an enhanced Poisson expected-goals model. V4 is documented as the production-facing model because it includes the richest match-generation logic, but the validation table should be read across multiple metrics rather than as a single winner-takes-all leaderboard.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/model_validation_2022.json`. The holdout is the 2022 FIFA World Cup. Each trained row uses the same anchor policy: for the 2022 holdout, training starts at the 1998 World Cup kickoff and ends before the first 2022 World Cup match.

- Match window: `10`
- Monte Carlo simulations: `20,000`
- Seed: `20260403`

| Model | Scope | Matches | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo-only baseline | all_international_since_anchor | 22843 | 1.0539 | 0.6353 | 42.2% | 24.2% / 23.4% | 0 | 0 | No |
| V2 World Cup only | world_cup_only | 384 | 1.0640 | 0.6135 | 51.6% | 23.5% / 23.4% | 9 | 1 | Yes |
| V2 all international since anchor | all_international_since_anchor | 22843 | 1.0148 | 0.6018 | 48.4% | 22.1% / 23.4% | 9 | 2 | No |
| V3 World Cup only | world_cup_only | 384 | 1.0276 | 0.6039 | 56.2% | 24.1% / 23.4% | 9 | 2 | No |
| V3 all international since anchor | all_international_since_anchor | 22843 | 1.0225 | 0.6042 | 50.0% | 21.6% / 23.4% | 10 | 2 | No |
| V4 all international since anchor | all_international_since_anchor | 22843 | 1.0207 | 0.5999 | 51.6% | 23.1% / 23.4% | 9 | 2 | No |

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## How To Read The Metrics

- **Log loss** rewards calibrated probabilities assigned to the actual class; lower is better.
- **Brier score** measures multiclass probability error; lower is better.
- **Top-1 accuracy** measures whether the highest-probability match outcome occurred.
- **Draw Pred./Actual** checks whether the model's average draw probability is close to the observed draw rate.
- **R16, SF, and Champion hits** evaluate tournament simulation outputs against actual 2022 advancement outcomes.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference, trained on all international matches since the anchor date with tournament sample weights.
- **V2:** multinomial logistic regression using Elo, recent form, goal profile, and prior World Cup history differences. It is validated under both World-Cup-only and all-international training scopes.
- **V3:** Poisson expected-goals model using Elo, form, historical pedigree, host/neutral-site context, and competition importance. It is validated under both World-Cup-only and all-international training scopes.
- **V4:** enhanced Poisson expected-goals model using quadratic recent form, World Cup last-5 goal-difference features, Dixon-Coles low-score correction, stage multipliers, time-decayed training weights, and alpha selection. It is the current primary dashboard model and is validated under the all-international training scope.

## Training Scopes And Weights

- `world_cup_only`: historical World Cup finals matches from the anchor World Cup onward.
- `all_international_since_anchor`: all international matches from the anchor World Cup kickoff onward.
- Sample-weight policy: `World Cup finals=3.0; continental finals=2.5; qualifiers=2.0; other competitive=1.5; friendlies=1.0`.

## Leakage Controls

- All validation rows use a cutoff before the first 2022 World Cup match.
- Team features for the 2022 holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual 2022 outcomes after simulation.
- 2026 forecasts use pre-tournament team metadata, fixtures, rankings, Elo snapshots, and lead-in results only.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- The 2022 holdout is a useful sanity check, not a full multi-tournament validation suite.
- Penalty shootouts and extra time are simplified relative to real match dynamics.
- V4 has more components than V2/V3, so it carries higher overfitting risk until more rolling holdout folds are implemented.
