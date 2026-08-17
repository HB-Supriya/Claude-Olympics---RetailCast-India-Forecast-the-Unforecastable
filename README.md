# RetailCast India — Forecast 28 Days of Demand

Forecasts 28 days of daily unit sales for 60 series (6 products × 10 Indian stores) from
`data/sales_train.csv`, scored by `mean_rmsse` (M5-style RMSSE, unweighted mean across series).

## Setup

Requires Python 3.14 (or any Python 3.10+; `pandas`/`numpy` are pinned to versions with native
wheels for 3.14 — see `requirements.txt`).

```bash
make setup
```

This creates `.venv/`, installs pinned dependencies, and registers a Jupyter kernel
(`retailcast-venv`) for the audit notebook.

Populate `data/` with the challenge's starter-kit files (`sales_train.csv`, `calendar.csv`,
`sell_prices.csv`, `market_signal.csv`, `vendor_signal.csv`, `sample_submission.csv`,
`data_dictionary.md`) — `data/` is gitignored, so this repo doesn't carry a copy:

```bash
cp -R /path/to/starter_kit/data/. data/
```

## Reproduce the submission from scratch

```bash
make setup      # venv + deps
make audit      # executes notebooks/01_audit.ipynb, reproduces Section 1's evidence
make backtest   # rolling-origin backtest across all candidate models (prints RMSSE report)
make predict    # trains the final segment-blended model, writes submission.csv
make validate   # runs validate_format.py — should print PASS
make test       # pytest mirror of the same structural checks
```

Or run everything after setup in one shot: `make all`.

## What ships, and why

Full reasoning is in `notebooks/01_audit.ipynb` (data audit) and `approach_summary.md`
(modeling decisions, validation, trust caveats). Short version:

- `market_signal.csv` is **excluded** — it stops at `d_1913` (no horizon coverage) and
  correlates with same-day sales at r≈0.9, ~10.5× its scale: leakage-shaped, not a genuine
  independent signal.
- `vendor_signal.csv` covers the full horizon but is **not used as a model feature** — a
  rolling-origin backtest (`make backtest`) showed adding it made mean RMSSE slightly *worse*
  (0.9064 vs 0.9048), so it fails the "must measurably help" bar from the project blueprint.
- The final model is **segment-blended**, chosen by the same backtest: a global gradient-boosted
  model (`sklearn.ensemble.HistGradientBoostingRegressor`, direct multi-horizon) for the 36
  "regular" series, and TSB (intermittent-demand smoothing) for the 24 series with >50%
  zero-sales days, where TSB beat the global model on every one of 4 backtest folds.
- LightGBM was the original plan but couldn't load its `libomp` dependency on this machine
  (macOS + Python 3.14); `HistGradientBoostingRegressor` was used instead as a like-for-like
  substitute (native categoricals, native quantile loss, native NaN handling).

## Repo layout

```
data/                       # populated locally, gitignored
notebooks/01_audit.ipynb    # executed data audit — Section 1 evidence
src/data.py                 # wide->long melt, calendar/price joins
src/features.py             # calendar/festival/SNAP/price/lag-rolling features
src/validation.py           # rolling-origin folds + M5 RMSSE
src/models/baseline.py      # seasonal-naive, moving-average, vendor-forecast-as-model
src/models/statistical.py   # TSB (intermittent demand)
src/models/global_model.py  # global gradient-boosted model + quantile heads
src/backtest_report.py      # runs the rolling-origin backtest, prints/saves the RMSSE report
src/predict.py              # trains + writes submission.csv + forecast_detail.csv
tests/test_submission_format.py
submission.csv              # generated
forecast_detail.csv         # generated — per-series segment/model/p10/p50/p90
backtest_scores.csv         # generated — per-series/fold/model RMSSE, backing for approach_summary.md
```
