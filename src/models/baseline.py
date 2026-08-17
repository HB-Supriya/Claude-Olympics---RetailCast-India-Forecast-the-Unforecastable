"""Tier 0 baselines: seasonal-naive and moving-average. Required regardless of what model
ships — they're the sanity/honesty check against which everything else is judged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_forecast(panel: pd.DataFrame, cutoff_d_num: int, horizon: int) -> pd.DataFrame:
    """Tile the last 7 days of history across the horizon (repeat 'same weekday last week')."""
    rows = []
    hist = panel[panel["d_num"] <= cutoff_d_num]
    for sid, g in hist.groupby("id"):
        g = g.sort_values("d_num")
        last7 = g["sales"].to_numpy(dtype=float)[-7:]
        for h in range(1, horizon + 1):
            yhat = last7[(h - 1) % 7]
            rows.append({"id": sid, "d_num": cutoff_d_num + h, "yhat": yhat})
    return pd.DataFrame(rows)


def moving_average_forecast(
    panel: pd.DataFrame, cutoff_d_num: int, horizon: int, window: int = 28
) -> pd.DataFrame:
    rows = []
    hist = panel[panel["d_num"] <= cutoff_d_num]
    for sid, g in hist.groupby("id"):
        g = g.sort_values("d_num")
        trailing = g["sales"].to_numpy(dtype=float)[-window:]
        yhat = float(np.nanmean(trailing)) if len(trailing) else 0.0
        for h in range(1, horizon + 1):
            rows.append({"id": sid, "d_num": cutoff_d_num + h, "yhat": yhat})
    return pd.DataFrame(rows)


def vendor_signal_as_forecast(vendor_signal: pd.DataFrame, cutoff_d_num: int, horizon: int) -> pd.DataFrame:
    """Use the vendor's own forecast product directly as a candidate forecast — the
    "benchmark to beat" from Section 1.2 of the blueprint.
    """
    vs = vendor_signal.copy()
    vs["d_num"] = vs["d"].str.replace("d_", "", regex=False).astype(int)
    window = vs[(vs["d_num"] > cutoff_d_num) & (vs["d_num"] <= cutoff_d_num + horizon)]
    return window[["id", "d_num", "vendor_forecast"]].rename(columns={"vendor_forecast": "yhat"})
