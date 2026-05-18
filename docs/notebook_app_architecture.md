# Notebook and App Architecture

This project separates the portfolio narrative from the interactive analysis surface.

## Reviewer Path

1. Start with `README.md` for setup and project structure.
2. Open `main.ipynb` for the concise analytical story and key findings.
3. Run `streamlit run apps/home.py` for the interactive dashboard.
4. Use the `Historical EDA` page for detailed historical exploration.
5. Use the V1/V2/V3 and backtest pages to inspect the forecasting workflow.

## Design Split

- `main.ipynb` is intentionally short. It shows the project framing, methodology, representative outputs, limitations, and how the historical analysis connects to the forecasting app.
- `apps/pages/8_Historical_EDA.py` provides the exploratory depth that used to make the notebook long and hard to review.
- `world_cup_sim/analysis.py` contains shared pandas transformations so the notebook and app use the same historical metrics.
- `world_cup_simulation.py` remains the forecasting and simulation layer.

This keeps the project readable for a resume review while preserving the deeper analysis in an interactive product surface.
