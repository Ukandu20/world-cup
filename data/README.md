# Data Layout

This project separates deployable data from rebuild-only raw data.

## Tracked Data

- `data/processed/world_cup/`: cleaned World Cup data consumed by the Streamlit app, simulations, and tests.
- `data/processed/international/results.csv`: international match results used by the V3 Poisson model.

These files are intentionally committed so a clean clone can run the app without a private local Kaggle cache.

## Ignored Data

Raw downloads stay ignored. Keep local Kaggle/source downloads outside `data/processed/`, for example:

- `data/results.csv`
- `data/goalscorers.csv`
- `data/shootouts.csv`
- `data/former_names.csv`
- `INT-World Cup/`

The build scripts can use those local raw files to regenerate the processed snapshot, but the app should not depend on ignored paths at runtime.

To refresh the ignored Kaggle raw files used by the builders:

```bash
python scripts/bootstrap_kaggle_data.py
```

Use `--force` to force a re-download.

## Overrides

Advanced users can point the app at another prepared dataset with:

```bash
set WORLD_CUP_DATA_ROOT=C:\path\to\world_cup
set INTERNATIONAL_RESULTS_PATH=C:\path\to\results.csv
```

On PowerShell:

```powershell
$env:WORLD_CUP_DATA_ROOT = "C:\path\to\world_cup"
$env:INTERNATIONAL_RESULTS_PATH = "C:\path\to\results.csv"
```
