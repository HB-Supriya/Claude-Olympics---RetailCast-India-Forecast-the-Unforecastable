"""Pytest mirror of validate_format.py's structural checks, so `make test` / CI catches a
broken submission before you'd notice via the manual script.
"""
import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_PATH = os.path.join(REPO_ROOT, "submission.csv")
SAMPLE_PATH = os.path.join(REPO_ROOT, "sample_submission.csv")
FCOLS = [f"F{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def submission():
    if not os.path.exists(SUBMISSION_PATH):
        pytest.skip("submission.csv not generated yet — run `make predict` first")
    return pd.read_csv(SUBMISSION_PATH)


@pytest.fixture(scope="module")
def sample_ids():
    return pd.read_csv(SAMPLE_PATH, dtype=str)["id"].str.strip().tolist()


def test_has_required_columns(submission):
    assert "id" in submission.columns
    for c in FCOLS:
        assert c in submission.columns, f"missing {c}"


def test_row_count(submission):
    assert len(submission) == 60


def test_no_duplicate_ids(submission):
    sid = submission["id"].str.strip()
    assert not sid.duplicated().any()


def test_ids_match_sample(submission, sample_ids):
    sid = set(submission["id"].str.strip())
    assert sid == set(sample_ids)


def test_forecast_values_numeric_nonnegative_finite(submission):
    vals = submission[FCOLS].apply(pd.to_numeric, errors="coerce")
    assert not vals.isna().any().any(), "non-numeric/empty forecast cell(s)"
    arr = vals.to_numpy(dtype=float)
    assert (arr >= 0).all(), "negative forecast value(s)"
    assert np.isfinite(arr).all(), "inf/NaN present"
