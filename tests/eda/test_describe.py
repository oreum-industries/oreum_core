"""Tests for eda.describe.describe()"""

import numpy as np
import pandas as pd
import pytest

from oreum_core.eda.describe import describe


@pytest.fixture
def df():
    """Simple two-column float/string DataFrame with a named index"""
    return pd.DataFrame(
        {"score": [1.0, 2.0, 3.0, 4.0, 5.0], "label": ["a", "b", "a", "c", "b"]},
        index=pd.Index(range(5), name="uid"),
    )


class TestDescribe:
    """Happy-path tests for describe()"""

    def test_basic_structure(self, df):
        """Happy: return_df=True → DataFrame with index named 'ft', one row per feature, 'dtype' column"""
        out = describe(df, return_df=True, reset_index=False)
        assert isinstance(out, pd.DataFrame)
        assert out.index.name == "ft"
        assert set(out.index) == {"score", "label"}
        assert "dtype" in out.columns

    def test_count_columns_present_by_default(self, df):
        """Happy: get_counts=True (default) → count_null, count_inf, count_zero present"""
        out = describe(df, return_df=True, reset_index=False)
        assert "count_null" in out.columns
        assert "count_inf" in out.columns
        assert "count_zero" in out.columns

    def test_sum_computed_for_numeric_column(self, df):
        """Happy: 'sum' column has correct total for float64 column"""
        out = describe(df, return_df=True, reset_index=False)
        assert out.loc["score", "sum"] == pytest.approx(15.0)

    def test_nobs_example_columns_prepended(self, df):
        """Happy: nobs=2 prepends 2 extra columns vs nobs=0"""
        out_0 = describe(df, nobs=0, return_df=True, reset_index=False)
        out_2 = describe(df, nobs=2, return_df=True, reset_index=False)
        assert len(out_2.columns) == len(out_0.columns) + 2

    def test_cr94_adds_outer_percentile_columns(self, df):
        """Happy: get_cr94=True adds '3%' and '97%' columns"""
        out = describe(df, get_cr94=True, return_df=True, reset_index=False)
        assert "3%" in out.columns
        assert "97%" in out.columns

    def test_reset_index_prefixes_index_column(self, df):
        """Happy: reset_index=True (default) → named index appears as 'index: uid'"""
        out = describe(df, return_df=True, reset_index=True)
        assert "index: uid" in out.index


class TestDescribeStringMinMax:
    """Tests for the string-like min/max section of describe()"""

    def test_string_col_min_max_populated(self, df):
        """Happy: string column gets correct min and max"""
        out = describe(df, nobs=0, return_df=True, reset_index=False)
        assert out.loc["label", "min"] == "a"
        assert out.loc["label", "max"] == "c"

    def test_numpy_nan_in_object_col_excluded(self):
        """Happy: np.nan mixed into object-dtype column is excluded from min/max"""
        df = pd.DataFrame(
            {"label": pd.array(["z", np.nan, "a", np.nan, "m"], dtype=object)}
        )
        out = describe(df, nobs=0, return_df=True, reset_index=False)
        assert out.loc["label", "min"] == "a"
        assert out.loc["label", "max"] == "z"

    def test_all_null_string_col_returns_na(self):
        """Edge: all-null object-dtype column → pd.NA for min and max"""
        df = pd.DataFrame({"label": pd.array([None, None, None], dtype=object)})
        out = describe(df, nobs=0, return_df=True, reset_index=False)
        assert pd.isna(out.loc["label", "min"])
        assert pd.isna(out.loc["label", "max"])


class TestDescribeGetMode:
    """Tests for the get_mode=True branch of describe()"""

    def test_mode_columns_present_with_correct_values(self, df):
        """Happy: get_mode=True → 'mode' and 'mode_count' present; label mode is 'a' (count=2)"""
        out = describe(df, get_mode=True, nobs=0, return_df=True, reset_index=False)
        assert "mode" in out.columns
        assert "mode_count" in out.columns
        assert out.loc["label", "mode"] == "a"  # "a" and "b" tie; mode() returns lowest
        assert out.loc["label", "mode_count"] == 2

    def test_numeric_col_has_no_mode(self, df):
        """Happy: numeric column is excluded from mode computation → NaN mode"""
        out = describe(df, get_mode=True, nobs=0, return_df=True, reset_index=False)
        assert pd.isna(out.loc["score", "mode"])

    def test_all_null_col_returns_na_mode(self):
        """Edge: all-null non-numeric column → pd.NA mode, count=0"""
        df = pd.DataFrame({"label": pd.array([None, None, None], dtype=object)})
        out = describe(df, get_mode=True, nobs=0, return_df=True, reset_index=False)
        assert pd.isna(out.loc["label", "mode"])
        assert out.loc["label", "mode_count"] == 0


class TestDescribeSadPath:
    """Sad-path tests for describe()"""

    def test_over_limit_no_subsample_returns_none(self):
        """Sad: df exceeds memory limit with subsample=False → returns None"""
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        out = describe(df, limit=0, subsample=False, return_df=True)
        assert out is None
