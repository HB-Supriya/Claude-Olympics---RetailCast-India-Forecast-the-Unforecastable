"""Rolling-origin backtest folds + RMSSE (M5 definition).

RMSSE per series i, given training history y_1..y_n and a horizon forecast yhat_{n+1..n+h}
scored against actuals y_{n+1..n+h}:

    RMSSE_i = sqrt( mean_{t=n+1..n+h} (y_t - yhat_t)^2 )
              / sqrt( mean_{t=2..n} (y_t - y_{t-1})^2 )

The denominator is the *training-history* one-step-ahead naive (lag-1) squared error, per the
official M5 definition — this is a fixed scaling term computed once from history, independent
of whatever model produces yhat. It is NOT the same thing as scoring a "repeat 7-days-ago"
seasonal-naive *model* (that's one of the candidate forecasters we back-test in
`models/baseline.py`, evaluated with this same RMSSE). Keeping these two uses of "naive"
distinct is a specific, deliberate choice — worth calling out explicitly since the blueprint's
prose uses "naive" for both.

`mean_rmsse` (per `olympics.json`) is the unweighted mean of RMSSE_i across all 60 series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import N_HISTORY_DAYS, N_HORIZON_DAYS


def rmsse_denominator(train_sales: np.ndarray) -> float:
    diffs = np.diff(train_sales)
    return float(np.sqrt(np.mean(diffs ** 2)))


def rmsse_per_series(
    actual: np.ndarray, forecast: np.ndarray, train_sales: np.ndarray
) -> float:
    denom = rmsse_denominator(train_sales)
    if denom == 0 or np.isnan(denom):
        return np.nan
    numer = np.sqrt(np.mean((actual - forecast) ** 2))
    return numer / denom


def score_forecasts(
    panel: pd.DataFrame, forecasts: pd.DataFrame, cutoff_d_num: int, horizon: int
) -> pd.DataFrame:
    """`forecasts`: columns id, d_num, yhat. Returns one row per id with rmsse."""
    rows = []
    for sid, g in panel[panel["id"].isin(forecasts["id"].unique())].groupby("id"):
        g = g.sort_values("d_num")
        train = g[g["d_num"] <= cutoff_d_num]["sales"].to_numpy(dtype=float)
        actual_window = g[
            (g["d_num"] > cutoff_d_num) & (g["d_num"] <= cutoff_d_num + horizon)
        ].sort_values("d_num")
        actual = actual_window["sales"].to_numpy(dtype=float)
        fc = (
            forecasts[forecasts["id"] == sid]
            .sort_values("d_num")["yhat"]
            .to_numpy(dtype=float)
        )
        rows.append({"id": sid, "rmsse": rmsse_per_series(actual, fc, train)})
    return pd.DataFrame(rows)


def rolling_origin_cutoffs(
    n_folds: int = 4, horizon: int = N_HORIZON_DAYS, step: int = 91, last_history_day: int = N_HISTORY_DAYS
) -> list[int]:
    """Cutoffs stepping back from the last history day, each leaving a full `horizon`-day
    window inside history to score against (so folds never touch the real, unlabeled
    forecast horizon).
    """
    cutoffs = [last_history_day - horizon - step * k for k in range(n_folds)]
    return sorted(c for c in cutoffs if c - 90 > 0)
