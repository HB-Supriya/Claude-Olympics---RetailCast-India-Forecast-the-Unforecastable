"""Generates submission.csv end-to-end from `data/` using the segment-blended final model,
justified by `backtest_report.py`'s rolling-origin results:

  - `regular` series (<=50% zero-days in history): global gradient-boosted model
    (`models/global_model.py`), no vendor_signal feature — the backtest showed adding
    vendor_signal makes mean RMSSE slightly *worse* (0.9064 vs 0.9048), so it's excluded.
  - `low_volume_intermittent` series (>50% zero-days): TSB (`models/statistical.py`) —
    the backtest showed TSB beats the global model on this segment on every one of the 4
    rolling-origin folds (e.g. fold d_1885: TSB 1.0112 vs global 1.0548).

Also writes `forecast_detail.csv` (per series-day: segment, model used, p10/p50/p90) as the
concrete backing for the per-series trust/uncertainty commentary in approach_summary.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import data as data_mod
import features
from models import global_model, statistical

HORIZON = 28
LAST_HISTORY_D = 1913
LOW_VOLUME_ZERO_SHARE_THRESHOLD = 0.5
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def segment_series(panel: pd.DataFrame, cutoff_d_num: int) -> pd.Series:
    hist = panel[panel["d_num"] <= cutoff_d_num]
    zero_share = hist.groupby("id")["sales"].apply(lambda s: (s == 0).mean())
    return (zero_share > LOW_VOLUME_ZERO_SHARE_THRESHOLD).map(
        {True: "low_volume_intermittent", False: "regular"}
    )


def empirical_bounds(panel: pd.DataFrame, cutoff_d_num: int, window: int = 90) -> pd.DataFrame:
    """p10/p90 for the low-volume segment: empirical quantiles of the trailing `window` days
    of history, since TSB has no native uncertainty output.
    """
    hist = panel[panel["d_num"] <= cutoff_d_num].sort_values(["id", "d_num"])
    rows = []
    for sid, g in hist.groupby("id"):
        sales = g["sales"].to_numpy(dtype=float)[-window:]
        rows.append({
            "id": sid,
            "p10_emp": float(np.percentile(sales, 10)) if len(sales) else 0.0,
            "p90_emp": float(np.percentile(sales, 90)) if len(sales) else 0.0,
        })
    return pd.DataFrame(rows)


def build_forecast(panel: pd.DataFrame, calf: pd.DataFrame, cutoff_d_num: int, horizon: int) -> pd.DataFrame:
    seg = segment_series(panel, cutoff_d_num).rename("segment").reset_index()
    regular_ids = seg[seg["segment"] == "regular"]["id"].tolist()
    low_vol_ids = seg[seg["segment"] == "low_volume_intermittent"]["id"].tolist()
    print(f"Segments as of d_{cutoff_d_num}: {len(regular_ids)} regular, {len(low_vol_ids)} low_volume_intermittent")

    train_cutoffs = features.training_cutoffs(cutoff_d_num, horizon)
    train_frame = features.build_training_frame(panel, calf, train_cutoffs, horizon)
    predict_frame = features.build_horizon_frame(panel, calf, cutoff_d_num, horizon)

    gbm_fc = global_model.global_model_forecast(train_frame, predict_frame, features.feature_cols())
    gbm_fc = gbm_fc[gbm_fc["id"].isin(regular_ids)].copy()
    gbm_fc["segment"] = "regular"
    gbm_fc["model"] = "global_gbm"

    tsb_panel = panel[panel["id"].isin(low_vol_ids)]
    tsb_fc = statistical.tsb_forecast(tsb_panel, cutoff_d_num, horizon)
    bounds = empirical_bounds(panel, cutoff_d_num)
    tsb_fc = tsb_fc.merge(bounds, on="id", how="left")
    tsb_fc["p10"] = np.minimum(tsb_fc["p10_emp"], tsb_fc["yhat"])
    tsb_fc["p90"] = np.maximum(tsb_fc["p90_emp"], tsb_fc["yhat"])
    tsb_fc = tsb_fc.drop(columns=["p10_emp", "p90_emp"])
    tsb_fc["segment"] = "low_volume_intermittent"
    tsb_fc["model"] = "tsb"

    combined = pd.concat([gbm_fc, tsb_fc], ignore_index=True)
    combined["yhat"] = np.clip(combined["yhat"], 0, None)
    return combined


def sanity_check_volume(panel: pd.DataFrame, forecast: pd.DataFrame, cutoff_d_num: int, horizon: int) -> None:
    recent_hist = panel[(panel["d_num"] > cutoff_d_num - horizon) & (panel["d_num"] <= cutoff_d_num)]
    recent_sum = recent_hist.groupby("id")["sales"].sum()
    fc_sum = forecast.groupby("id")["yhat"].sum()
    ratio = (fc_sum / recent_sum.replace(0, np.nan)).rename("fc_vs_recent_ratio")
    flagged = ratio[(ratio > 10) | (ratio < 0.1)]
    print(f"\nVolume sanity check: forecast sum vs last {horizon} historical days, per series.")
    print(f"Ratio range: min={ratio.min():.2f} max={ratio.max():.2f} median={ratio.median():.2f}")
    if len(flagged):
        print(f"FLAGGED ({len(flagged)} series outside 0.1x-10x of recent history):")
        print(flagged.to_string())
    else:
        print("No series flagged (all within 0.1x-10x of recent historical volume).")


def main():
    panel = data_mod.build_panel()
    cal = data_mod.load_calendar()
    calf = features.calendar_features_for_days(cal)

    forecast = build_forecast(panel, calf, LAST_HISTORY_D, HORIZON)
    sanity_check_volume(panel, forecast, LAST_HISTORY_D, HORIZON)

    forecast = forecast.sort_values(["id", "d_num"])
    forecast["h"] = forecast.groupby("id").cumcount() + 1
    wide = forecast.pivot(index="id", columns="h", values="yhat")
    wide.columns = [f"F{h}" for h in wide.columns]
    wide = wide.reset_index()

    sample = pd.read_csv(os.path.join(REPO_ROOT, "sample_submission.csv"))
    wide = sample[["id"]].merge(wide, on="id", how="left")
    fcols = [f"F{i}" for i in range(1, HORIZON + 1)]
    wide[fcols] = wide[fcols].round(3)

    out_path = os.path.join(REPO_ROOT, "submission.csv")
    wide.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(wide)} rows)")

    detail_path = os.path.join(REPO_ROOT, "forecast_detail.csv")
    forecast[["id", "d_num", "h", "segment", "model", "yhat", "p10", "p90"]].to_csv(
        detail_path, index=False
    )
    print(f"Wrote {detail_path} ({len(forecast)} rows) — per-series trust/uncertainty backing")


if __name__ == "__main__":
    main()
