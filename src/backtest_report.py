"""Runs the rolling-origin backtest across all candidate models and prints a report:
overall + per-segment RMSSE, with baseline RMSSE alongside every model on every fold.

This is what decides, with numbers (not assumption): whether the vendor_signal feature earns
its place in the global model, and what final tier ships.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import data as data_mod
import features
import validation
from models import baseline, statistical, global_model

N_FOLDS = 4
HORIZON = 28
LOW_VOLUME_ZERO_SHARE_THRESHOLD = 0.5


def segment_series(panel: pd.DataFrame, cutoff_d_num: int) -> pd.Series:
    hist = panel[panel["d_num"] <= cutoff_d_num]
    zero_share = hist.groupby("id")["sales"].apply(lambda s: (s == 0).mean())
    return (zero_share > LOW_VOLUME_ZERO_SHARE_THRESHOLD).map(
        {True: "low_volume_intermittent", False: "regular"}
    )


def run_fold(panel, calf, vendor_signal, cutoff, horizon, use_vendor_feature):
    results = {}

    results["seasonal_naive"] = baseline.seasonal_naive_forecast(panel, cutoff, horizon)
    results["moving_average_28"] = baseline.moving_average_forecast(panel, cutoff, horizon)
    results["vendor_forecast_raw"] = baseline.vendor_signal_as_forecast(vendor_signal, cutoff, horizon)
    results["tsb"] = statistical.tsb_forecast(panel, cutoff, horizon)

    train_cutoffs = features.training_cutoffs(cutoff, horizon)
    train_frame = features.build_training_frame(
        panel, calf, train_cutoffs, horizon, vendor_signal, use_vendor_feature
    )
    predict_frame = features.build_horizon_frame(
        panel, calf, cutoff, horizon, vendor_signal, use_vendor_feature
    )
    fc = global_model.global_model_forecast(
        train_frame, predict_frame, features.feature_cols(use_vendor_feature)
    )
    model_name = "global_gbm_with_vendor" if use_vendor_feature else "global_gbm"
    results[model_name] = fc[["id", "d_num", "yhat"]]

    return results


def main():
    panel = data_mod.build_panel()
    cal = data_mod.load_calendar()
    calf = features.calendar_features_for_days(cal)
    vendor_signal = data_mod.load_vendor_signal()

    cutoffs = validation.rolling_origin_cutoffs(n_folds=N_FOLDS, horizon=HORIZON)
    print(f"Rolling-origin cutoffs (d_num): {cutoffs}\n")

    all_scores = []
    for cutoff in cutoffs:
        print(f"=== Fold cutoff d_{cutoff} (scoring d_{cutoff+1}..d_{cutoff+HORIZON}) ===")
        seg = segment_series(panel, cutoff)

        for use_vendor in (False, True):
            fold_results = run_fold(panel, calf, vendor_signal, cutoff, HORIZON, use_vendor)
            for model_name, fc in fold_results.items():
                if use_vendor and not model_name.startswith("global_gbm"):
                    continue  # baselines don't depend on use_vendor; skip duplicate scoring
                scored = validation.score_forecasts(panel, fc, cutoff, HORIZON)
                scored["model"] = model_name
                scored["cutoff"] = cutoff
                scored["segment"] = scored["id"].map(seg)
                all_scores.append(scored)

    scores = pd.concat(all_scores, ignore_index=True)

    print("\n=== Per-fold mean RMSSE by model ===")
    print(scores.groupby(["cutoff", "model"])["rmsse"].mean().unstack("model").round(4).to_string())

    print("\n=== Overall mean RMSSE by model (mean_rmsse, unweighted across series+folds) ===")
    print(scores.groupby("model")["rmsse"].mean().sort_values().round(4).to_string())

    print("\n=== Per-segment mean RMSSE by model ===")
    print(
        scores.groupby(["segment", "model"])["rmsse"]
        .mean()
        .unstack("model")
        .round(4)
        .to_string()
    )

    scores.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "backtest_scores.csv"), index=False
    )
    print("\nSaved per-series/fold/model scores to backtest_scores.csv")


if __name__ == "__main__":
    main()
