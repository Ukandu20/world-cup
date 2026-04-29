# World Cup Forecasting Model Card

## Purpose

This project estimates preseason FIFA World Cup 2026 team and tournament probabilities. It is intended as a forecasting and portfolio dashboard, not as betting advice or a match-day injury-aware prediction service.

## Validation Snapshot

The committed validation artifact is `data/processed/validation/model_validation_2022.json`. The holdout is the 2022 FIFA World Cup, with 2022 excluded from V2 and baseline training. The V3 model trains only on international results before the first 2022 World Cup match.

- Match window: `10`
- Monte Carlo simulations: `20,000`
- Seed: `20260403`

| Model | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo-only baseline | 1.0859 | 0.6248 | 53.1% | 21.9% / 23.4% | 0 | 0 | No |
| V2 multinomial model | 1.0610 | 0.6123 | 53.1% | 22.2% / 23.4% | 9 | 1 | No |
| V3 Poisson expected-goals model | 1.0214 | 0.6036 | 50.0% | 21.6% / 23.4% | 10 | 2 | No |

The Elo-only baseline is match-level only. Its tournament-stage fields are set to zero and flagged with `tournament_simulated=false` in the JSON artifact.

## Model Families

- **Baseline:** multinomial logistic regression using only pre-match Elo difference.
- **V2:** multinomial logistic regression trained on historical World Cup matches, using Elo, recent form, goal profile, and prior World Cup history differences.
- **V3:** Poisson expected-goals model trained on broader international results, using Elo, form, historical pedigree, host/neutral-site context, and competition importance.

## Leakage Controls

- V2 and the baseline exclude all 2022 World Cup matches from training.
- V3 uses a cutoff before the first 2022 World Cup match.
- Team features for the 2022 holdout are built from pre-tournament data.
- Tournament probabilities are evaluated against actual 2022 outcomes after simulation.

## Limitations

- The model does not ingest player-level squad quality, injuries, lineups, market odds, or tactical matchups.
- The 2026 forecast is preseason-oriented and should not be interpreted as live match pricing.
- The 2022 holdout is a useful sanity check, not a full multi-tournament validation suite.
- Penalty shootouts and extra time are simplified relative to real match dynamics.
