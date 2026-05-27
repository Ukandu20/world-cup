# World Cup 2026 Forecasting Dashboard

An end-to-end data science portfolio project for preseason FIFA Men's World Cup 2026 forecasting. The project combines cleaned football datasets, historical EDA, feature engineering, trained match models, Monte Carlo tournament simulation, validation artifacts, and a Streamlit dashboard for exploring team advancement probabilities.

The current dashboard primary model is **V4 Enhanced Poisson**, which estimates expected goals with Elo, recent form, World Cup history, host/context features, Dixon-Coles correction, stage multipliers, and time-decayed training weights. Earlier V1/V2/V3 model surfaces remain available for comparison and explainability.

## Results At A Glance

- **Forecasting product:** Streamlit dashboard with group probabilities, all-country rankings, deterministic bracket projections, backtest pages, and team report cards.
- **Primary model:** V4 enhanced Poisson expected-goals model with Monte Carlo tournament simulation.
- **Validation design:** 2022 World Cup holdout; training data ends before the first 2022 tournament match.
- **Simulation settings:** `20,000` tournament simulations, match window `10`, seed `20260403`.
- **V4 holdout snapshot:** `1.0207` log loss, `0.5999` Brier score, `51.6%` top-1 match accuracy, and draw calibration of `23.1%` predicted vs `23.4%` actual.

See the full [model card](docs/model_card.md) and committed validation artifact at [data/processed/validation/model_validation_2022.json](data/processed/validation/model_validation_2022.json).

## Project Visuals

![All countries probability view](assets/charts/generated/all_Countries_20260423_094740_649518.png)

The dashboard ranks teams across group-stage placement, knockout qualification, deep-run probabilities, and championship odds.

![Projected bracket view](assets/charts/generated/bracket_view_20260405_122411_025406.png)

The bracket view turns simulated group outcomes into a position-based knockout path instead of simply selecting a global top-N list.

![Goals and finish quadrants](assets/visualizations/goals_finish_quadrants.png)

The historical EDA layer connects tournament outcomes to interpretable signals such as scoring profile, finishing strength, host context, and prior World Cup performance.

## Skills Demonstrated

- **Data preparation:** cleans and stages World Cup, international match, ranking, Elo, fixture, squad, and country/entity datasets into app-ready processed files.
- **Feature engineering:** builds recent-form, goal-profile, historical-pedigree, host/context, competition-importance, and tournament-structure features.
- **Modeling:** compares Elo-only baseline, multinomial logistic regression, paired Poisson expected-goals models, and enhanced Poisson simulation.
- **Validation:** uses leakage-aware 2022 holdout evaluation with log loss, Brier score, top-1 accuracy, draw calibration, and tournament-stage hit metrics.
- **Product analytics:** converts model output into a Streamlit dashboard, probability tables, bracket projections, and team report cards.
- **Engineering hygiene:** includes reusable modules, dataset builders, model documentation, validation artifacts, GitHub Actions, and pytest coverage for simulation invariants.

## Data Sources

- [`Fjelstul, Joshua C. "The Fjelstul World Cup Database v.1.2.0." July 19, 2023`](https://github.com/jfjelstul/worldcup): historical World Cup match, team, squad, placement, and tournament-structure reference data.
- [Kaggle: FIFA World Cup Complete History 1930-2022](https://www.kaggle.com/datasets/mafaqbhatti/fifa-world-cup-complete-history-19302022): historical World Cup summaries and EDA inputs.
- [Kaggle: International Football Results from 1872](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017): international match results, goalscorers, shootouts, and former-name data used for lead-in form and model training.
- [World Football Elo Ratings](https://eloratings.net/): Elo snapshots, Elo changes, and team-strength signals.
- [FIFA.com](https://www.fifa.com/): official 2026 tournament, ranking, fixture, host, and competition reference information.
- [Wikipedia: 2026 FIFA World Cup](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup): 2026 tournament format, group, venue, and fixture cross-checking.
- [Wikipedia: 2026 FIFA World Cup squads](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads): squad/status reference where applicable.

Original sources retain their own licenses and terms. This repository stores processed snapshots for portfolio, research, and reproducibility purposes.

## Validation Summary

The published validation is a 2022 World Cup holdout using `20,000` simulations, match window `10`, and seed `20260403`.

| Model | Scope | Matches | Log Loss | Brier | Top-1 Acc. | Draw Pred./Actual | R16 Hits | SF Hits | Champion Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo-only baseline | all_international_since_anchor | 22,843 | 1.0539 | 0.6353 | 42.2% | 24.2% / 23.4% | 0 | 0 | No |
| V2 World Cup only | world_cup_only | 384 | 1.0640 | 0.6135 | 51.6% | 23.5% / 23.4% | 9 | 1 | Yes |
| V2 all international since anchor | all_international_since_anchor | 22,843 | 1.0148 | 0.6018 | 48.4% | 22.1% / 23.4% | 9 | 2 | No |
| V3 World Cup only | world_cup_only | 384 | 1.0276 | 0.6039 | 56.2% | 24.1% / 23.4% | 9 | 2 | No |
| V3 all international since anchor | all_international_since_anchor | 22,843 | 1.0225 | 0.6042 | 50.0% | 21.6% / 23.4% | 10 | 2 | No |
| V4 all international since anchor | all_international_since_anchor | 22,843 | 1.0207 | 0.5999 | 51.6% | 23.1% / 23.4% | 9 | 2 | No |

V4 is the current dashboard primary because it has the richest match-generation logic and the best Brier score among the trained tournament models in this holdout. It should not be read as universally best on every metric: V2 all-international has the lowest log loss in this run, and V3 World-Cup-only has the highest top-1 accuracy.

## Project Tour

- `main.ipynb`: concise executive notebook with framing, data-quality snapshot, historical findings, and model/app handoff.
- `apps/home.py`: Streamlit dashboard entrypoint.
- `apps/pages/1_Analysis.py`: historical analysis companion page for participation, goals, host effects, winner follow-up, correlations, and 2026 implications.
- `apps/team_report_card.py`: team-level report-card surface for translating model outputs into an analyst-friendly profile.
- `world_cup_sim/`: shared analysis, feature, model, and simulation modules.
- `scripts/`: reproducible dataset and validation builders.
- `docs/model_card.md`: validation, leakage controls, model-family summary, and limitations.
- `docs/models/`: deeper model documentation for V1 through V4.

Recommended reviewer path:

1. Read this README for the project story and setup.
2. Open `main.ipynb` for the concise analytical narrative.
3. Run `streamlit run apps/home.py` for the interactive dashboard.
4. Review `docs/model_card.md` for validation and limitations.

## Setup

Use Python `3.12.x` for local development. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The app-ready dataset is committed under `data/processed/`, so a clean clone can run without a private Kaggle cache:

```bash
streamlit run apps/home.py
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Raw Kaggle/source downloads remain ignored and rebuild-only. To refresh the local raw files used by the builders:

```bash
python scripts/bootstrap_kaggle_data.py
```

See [data/README.md](data/README.md) for the data layout and environment-variable overrides.

## Current Modeling Flow

The dashboard simulates the full 2026 tournament fixture-by-fixture.

- Group matches are sampled in kickoff order.
- Group standings use points, goal difference, goals scored, head-to-head tie-breakers, and deterministic strength fallback.
- Top-two teams in each group qualify automatically.
- The best eight third-place teams qualify through the official 2026 third-place routing map.
- Knockout matches are simulated through the final, including simplified extra time and penalties.
- Output probabilities include group finish, Round of 32, Round of 16, quarter-final, semi-final, final, and champion probabilities.

The deterministic bracket view is position-based: it uses modal group finishers and official knockout slots rather than taking the highest global probabilities.

## Limitations

- Forecasts are preseason estimates, not live match prices.
- The model does not ingest player-level injuries, lineups, market odds, tactical matchups, or live squad availability.
- The current validation artifact uses a 2022 holdout; broader rolling validation is an important next improvement.
- Penalty shootouts, extra time, fair-play tie-breakers, and drawing of lots are simplified.
- V4 has more components than V2/V3, so it carries higher overfitting risk until more holdout folds are implemented.

## Reference Docs

- [Model card](docs/model_card.md)
- [Model documentation index](docs/models/README.md)
- [Notebook and app architecture](docs/notebook_app_architecture.md)
- [Elo rating reference](docs/elo_rating_reference.md)
- [Data layout](data/README.md)
