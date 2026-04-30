# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/model_validation_2022.json`. The holdout is the 2022 FIFA World Cup. Each trained row uses the same anchor policy: for the 2022 holdout, training starts at the 1998 World Cup kickoff and ends before the first 2022 World Cup match.

- Match window: `10`
- Monte Carlo simulations: `20,000`
- Seed: `20260403`

| Model | Scope | Matches | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo-only baseline | all_international_since_anchor | 22843 | 1.0545 | 0.6358 | 42.2% | 24.1% / 23.4% | 0 | 0 | No |
| V2 World Cup only | world_cup_only | 384 | 1.0170 | 0.5991 | 51.6% | 24.0% / 23.4% | 10 | 1 | Yes |
| V2 all international since anchor | all_international_since_anchor | 22843 | 1.0124 | 0.6004 | 48.4% | 22.0% / 23.4% | 9 | 2 | No |
| V3 World Cup only | world_cup_only | 384 | 1.0204 | 0.6050 | 51.6% | 24.5% / 23.4% | 9 | 1 | No |
| V3 all international since anchor | all_international_since_anchor | 22843 | 1.0210 | 0.6033 | 50.0% | 21.6% / 23.4% | 10 | 2 | No |

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference, trained on all international matches since the anchor date with tournament sample weights.
- **V2:** multinomial logistic regression using Elo, recent form, goal profile, and prior World Cup history differences. It is validated under both World-Cup-only and all-international training scopes.
- **V3:** Poisson expected-goals model using Elo, form, historical pedigree, host/neutral-site context, and competition importance. It is validated under both World-Cup-only and all-international training scopes.

## Training Scopes And Weights

- `world_cup_only`: historical World Cup finals matches from the anchor World Cup onward.
- `all_international_since_anchor`: all international matches from the anchor World Cup kickoff onward.
- Sample-weight policy: `World Cup finals=3.0; continental finals=2.5; qualifiers=2.0; other competitive=1.5; friendlies=1.0`.

## Leakage Controls

- All validation rows use a cutoff before the first 2022 World Cup match.
- Team features for the 2022 holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual 2022 outcomes after simulation.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- The 2022 holdout is a useful sanity check, not a full multi-tournament validation suite.
- Penalty shootouts and extra time are simplified relative to real match dynamics.
