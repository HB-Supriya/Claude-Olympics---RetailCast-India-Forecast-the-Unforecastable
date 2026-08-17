"""Data loading: melt sales_train wide->long, join calendar/prices, build the base panel.

This is the single source of truth for reading `data/` — the audit notebook, the backtest
report, and the prediction pipeline all import from here so there is exactly one melt/join
implementation to keep correct.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

N_HISTORY_DAYS = 1913
N_HORIZON_DAYS = 28
LAST_HISTORY_D = f"d_{N_HISTORY_DAYS}"
HORIZON_DS = [f"d_{N_HISTORY_DAYS + i}" for i in range(1, N_HORIZON_DAYS + 1)]

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_sales_wide(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "sales_train.csv"))


def load_calendar(data_dir: str = DATA_DIR) -> pd.DataFrame:
    cal = pd.read_csv(os.path.join(data_dir, "calendar.csv"), parse_dates=["date"])
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    return cal


def load_sell_prices(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))


def load_market_signal(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "market_signal.csv"))


def load_vendor_signal(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "vendor_signal.csv"))


def melt_sales_long(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (`id, item_id, ..., d_1..d_1913`) -> long (`id, item_id, ..., d, sales`)."""
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide.melt(
        id_vars=ID_COLS, value_vars=d_cols, var_name="d", value_name="sales"
    ).copy()
    long["d_num"] = long["d"].str.replace("d_", "", regex=False).astype(int)
    return long


def build_panel(data_dir: str = DATA_DIR, include_horizon: bool = True) -> pd.DataFrame:
    """Long sales panel joined with calendar and weekly price, for `d_1..d_1913`
    (plus `d_1914..d_1941` horizon rows with `sales = NaN` if `include_horizon`).

    Price is joined on (store_id, item_id, wm_yr_wk) — weekly, not daily; a missing price row
    means the (item, store) was not sold that retail week (per data_dictionary.md).
    """
    sales_wide = load_sales_wide(data_dir)
    cal = load_calendar(data_dir)
    prices = load_sell_prices(data_dir)

    long = melt_sales_long(sales_wide)

    if include_horizon:
        base = sales_wide[ID_COLS]
        horizon_rows = base.assign(key=1).merge(
            pd.DataFrame({"d": HORIZON_DS, "key": 1}), on="key"
        ).drop(columns="key")
        horizon_rows["d_num"] = horizon_rows["d"].str.replace("d_", "", regex=False).astype(int)
        horizon_rows["sales"] = np.nan
        long = pd.concat([long, horizon_rows], ignore_index=True)

    panel = long.merge(cal, on="d", how="left", suffixes=("", "_cal"))
    panel = panel.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    panel["has_price_row"] = panel["sell_price"].notna()
    panel = panel.sort_values(["id", "d_num"]).reset_index(drop=True)
    return panel
