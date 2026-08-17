"""Feature engineering: calendar/festival/SNAP features, price features, lag/rolling
statistics computed as-of a cutoff day, and the (gated) vendor_signal feature.

Design constraint driving this file: we're building a **direct multi-horizon** model (one
model predicts all of d+1..d+28 from a horizon-index feature), not a recursive one-step model.
That means every lag/rolling feature must be computed once, as of the cutoff day, and reused
for all 28 horizon rows — never recomputed from "future" days the model hasn't seen yet. This
is what keeps the pipeline leakage-free for both backtesting and the real forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

LAG_WINDOWS = (7, 28)
ROLL_WINDOWS = (7, 28, 90)


def add_calendar_event_distance(cal: pd.DataFrame) -> pd.DataFrame:
    """Add `days_to_nearest_event`: absolute day distance to the closest non-null
    `event_name_1`/`event_name_2`, computed over the *whole* calendar (history + horizon +
    buffer) — this is safe because `calendar.csv` is known in advance for all dates, unlike
    sales.
    """
    cal = cal.sort_values("d_num").reset_index(drop=True)
    has_event = cal["event_name_1"].notna() | cal["event_name_2"].notna()
    event_days = cal.loc[has_event, "d_num"].to_numpy()
    if len(event_days) == 0:
        cal["days_to_nearest_event"] = np.nan
        return cal
    all_days = cal["d_num"].to_numpy()
    idx = np.searchsorted(event_days, all_days)
    idx_clipped_right = np.clip(idx, 0, len(event_days) - 1)
    idx_clipped_left = np.clip(idx - 1, 0, len(event_days) - 1)
    dist_right = np.abs(event_days[idx_clipped_right] - all_days)
    dist_left = np.abs(event_days[idx_clipped_left] - all_days)
    cal["days_to_nearest_event"] = np.minimum(dist_right, dist_left)
    return cal


def calendar_features_for_days(cal: pd.DataFrame) -> pd.DataFrame:
    cal = add_calendar_event_distance(cal)
    out = cal[
        [
            "d", "d_num", "wday", "month", "year",
            "event_type_1", "event_type_2",
            "snap_MH", "snap_KA", "snap_TN",
            "days_to_nearest_event",
        ]
    ].copy()
    out["is_event_day"] = cal["event_name_1"].notna().astype(int)
    for et in ["National", "Cultural", "Religious", "Sporting"]:
        out[f"event_type_{et}"] = (cal["event_type_1"] == et).astype(int)
    out = out.drop(columns=["event_type_1", "event_type_2"])
    return out


def series_asof_features(panel: pd.DataFrame, cutoff_d_num: int) -> pd.DataFrame:
    """One row per series id: lag-N, rolling mean/std over the last N days, and a
    state-price snapshot — all computed using only `d_num <= cutoff_d_num`.
    """
    hist = panel[panel["d_num"] <= cutoff_d_num].sort_values(["id", "d_num"])
    rows = []
    for sid, g in hist.groupby("id", sort=False):
        g = g.sort_values("d_num")
        sales = g["sales"].to_numpy(dtype=float)
        row = {"id": sid}
        for lag in LAG_WINDOWS:
            row[f"lag_{lag}"] = sales[-lag] if len(sales) >= lag else np.nan
        for win in ROLL_WINDOWS:
            window = sales[-win:] if len(sales) >= 1 else sales
            row[f"roll_mean_{win}"] = np.nanmean(window) if len(window) else np.nan
            row[f"roll_std_{win}"] = np.nanstd(window) if len(window) else np.nan
            row[f"roll_zero_share_{win}"] = np.mean(window == 0) if len(window) else np.nan
        last_price_rows = g[g["has_price_row"]]
        row["last_sell_price"] = (
            last_price_rows["sell_price"].iloc[-1] if len(last_price_rows) else np.nan
        )
        if len(last_price_rows) >= 2:
            row["price_change_pct"] = (
                last_price_rows["sell_price"].iloc[-1] / last_price_rows["sell_price"].iloc[-2] - 1
            )
        else:
            row["price_change_pct"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_horizon_frame(
    panel: pd.DataFrame,
    cal_features: pd.DataFrame,
    cutoff_d_num: int,
    horizon: int,
    vendor_signal: pd.DataFrame | None = None,
    use_vendor_feature: bool = False,
) -> pd.DataFrame:
    """Assemble the direct multi-horizon training/prediction frame for one cutoff.

    One row per (series, h) for h in 1..horizon. `sales` is the true target when it exists in
    `panel` (backtest), NaN for the real forecast horizon.
    """
    ids = panel[["id"] + CATEGORICAL_COLS].drop_duplicates("id")
    horizon_idx = pd.DataFrame({"h": range(1, horizon + 1)})
    frame = ids.assign(key=1).merge(horizon_idx.assign(key=1), on="key").drop(columns="key")
    frame["d_num"] = cutoff_d_num + frame["h"]

    frame = frame.merge(cal_features, on="d_num", how="left")

    asof = series_asof_features(panel, cutoff_d_num)
    frame = frame.merge(asof, on="id", how="left")

    target = panel[panel["d_num"] > cutoff_d_num][["id", "d_num", "sales"]]
    frame = frame.merge(target, on=["id", "d_num"], how="left")

    if use_vendor_feature and vendor_signal is not None:
        vs = vendor_signal.copy()
        vs["d_num"] = vs["d"].str.replace("d_", "", regex=False).astype(int)
        frame = frame.merge(
            vs[["id", "d_num", "vendor_forecast"]], on=["id", "d_num"], how="left"
        )

    for c in CATEGORICAL_COLS:
        frame[c] = frame[c].astype("category")

    return frame


FEATURE_COLS_BASE = (
    CATEGORICAL_COLS
    + ["h", "wday", "month", "year", "days_to_nearest_event", "is_event_day",
       "event_type_National", "event_type_Cultural", "event_type_Religious", "event_type_Sporting",
       "snap_MH", "snap_KA", "snap_TN",
       "lag_7", "lag_28",
       "roll_mean_7", "roll_mean_28", "roll_mean_90",
       "roll_std_7", "roll_std_28", "roll_std_90",
       "roll_zero_share_7", "roll_zero_share_28", "roll_zero_share_90",
       "last_sell_price", "price_change_pct"]
)


def training_cutoffs(
    fold_cutoff: int, horizon: int, min_history: int = 90, step: int = 28
) -> list[int]:
    """Pseudo-cutoffs used to build many direct-horizon training examples out of history —
    a single cutoff only yields 60 series x horizon rows, far too little for a GBM to learn
    the horizon-day/calendar/lag relationship. Every pseudo-cutoff here satisfies
    `pseudo_cutoff + horizon <= fold_cutoff`, so no training row ever looks past the fold's own
    origin day — this is what keeps a backtest fold's "future" out of its own training set.
    """
    max_cutoff = fold_cutoff - horizon
    return list(range(min_history, max_cutoff + 1, step))


def build_training_frame(
    panel: pd.DataFrame,
    cal_features: pd.DataFrame,
    cutoffs: list[int],
    horizon: int,
    vendor_signal: pd.DataFrame | None = None,
    use_vendor_feature: bool = False,
) -> pd.DataFrame:
    frames = [
        build_horizon_frame(panel, cal_features, c, horizon, vendor_signal, use_vendor_feature)
        for c in cutoffs
    ]
    return pd.concat(frames, ignore_index=True)


def feature_cols(use_vendor_feature: bool = False) -> list[str]:
    cols = list(FEATURE_COLS_BASE)
    if use_vendor_feature:
        cols.append("vendor_forecast")
    return cols
