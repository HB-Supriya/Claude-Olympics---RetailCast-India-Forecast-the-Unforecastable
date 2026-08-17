"""Tier 1 (targeted): TSB (Teunter-Syntetos-Babai) intermittent-demand forecaster.

Used only as a comparison baseline for the high-zero-share segment identified in the audit —
not the shipped scalable model (that's Tier 2, `global_model.py`, which is one vectorized model
over the whole panel). A per-series smoothing recursion like TSB is inherently sequential per
series regardless of scale, so a plain Python loop here is appropriate; it is not the pattern
used for anything meant to scale to thousands of series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tsb_series_forecast(sales: np.ndarray, horizon: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    nonzero_idx = np.flatnonzero(sales > 0)
    if len(nonzero_idx) == 0:
        return np.zeros(horizon)
    z_hat = sales[nonzero_idx[0]]
    p_hat = len(nonzero_idx) / len(sales)
    for t in range(len(sales)):
        occurred = sales[t] > 0
        if occurred:
            z_hat = z_hat + alpha * (sales[t] - z_hat)
        p_hat = p_hat + beta * (float(occurred) - p_hat)
    level = z_hat * p_hat
    return np.full(horizon, level)


def tsb_forecast(panel: pd.DataFrame, cutoff_d_num: int, horizon: int) -> pd.DataFrame:
    rows = []
    hist = panel[panel["d_num"] <= cutoff_d_num]
    for sid, g in hist.groupby("id"):
        g = g.sort_values("d_num")
        sales = g["sales"].to_numpy(dtype=float)
        fc = _tsb_series_forecast(sales, horizon)
        for h in range(1, horizon + 1):
            rows.append({"id": sid, "d_num": cutoff_d_num + h, "yhat": fc[h - 1]})
    return pd.DataFrame(rows)
