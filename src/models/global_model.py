"""Tier 2 (shipped model): one global gradient-boosted model over the whole panel, direct
multi-horizon (horizon-index `h` is a feature, no recursive re-forecasting), plus Tier 3a
quantile heads for per-series uncertainty.

Uses `sklearn.ensemble.HistGradientBoostingRegressor` rather than LightGBM — this environment
couldn't load LightGBM's compiled `libomp` dependency on macOS/Python 3.14, and the user chose
the sklearn fallback over a Homebrew system install. Same tier, same design: native categorical
support (`categorical_features="from_dtype"`), native NaN handling (no imputation needed for
missing lag/price features on short-history series), and native quantile loss for the
uncertainty heads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = (0.1, 0.5, 0.9)


def train_quantile_models(
    X: pd.DataFrame, y: pd.Series, quantiles=QUANTILES, random_state: int = 42
) -> dict[float, HistGradientBoostingRegressor]:
    models = {}
    for q in quantiles:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            categorical_features="from_dtype",
            random_state=random_state,
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
        )
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(
    models: dict[float, HistGradientBoostingRegressor], X: pd.DataFrame
) -> pd.DataFrame:
    preds = {q: np.clip(m.predict(X), 0, None) for q, m in models.items()}
    out = pd.DataFrame(preds)
    out.columns = [f"p{int(q * 100)}" for q in models.keys()]
    # enforce monotonic quantiles (quantile crossing can happen with independently-fit heads)
    sorted_cols = sorted(out.columns, key=lambda c: int(c[1:]))
    out[sorted_cols] = np.sort(out[sorted_cols].to_numpy(), axis=1)
    return out


def global_model_forecast(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit on `train_frame` (must have non-null `sales`), predict on `predict_frame`.
    Returns predict_frame's id/d_num plus yhat (median) and p10/p90 uncertainty bounds.
    """
    train = train_frame.dropna(subset=["sales"])
    X_train, y_train = train[feature_cols], train["sales"]
    models = train_quantile_models(X_train, y_train, random_state=random_state)
    preds = predict_quantiles(models, predict_frame[feature_cols])
    out = predict_frame[["id", "d_num"]].reset_index(drop=True)
    out["yhat"] = preds["p50"].to_numpy()
    out["p10"] = preds["p10"].to_numpy()
    out["p90"] = preds["p90"].to_numpy()
    return out
