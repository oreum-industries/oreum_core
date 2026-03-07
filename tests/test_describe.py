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

    def test_returns_dataframe(self, df):
        """Happy: return_df=True → pd.DataFrame"""
        out = describe(df, return_df=True, reset_index=False)
        assert isinstance(out, pd.DataFrame)

    def test_index_name_is_ft(self, df):
        """Happy: output index is named 'ft'"""
        out = describe(df, return_df=True, reset_index=False)
        assert out.index.name == "ft"

    def test_one_row_per_feature(self, df):
        """Happy: one output row per input column (reset_index=False)"""
        out = describe(df, return_df=True, reset_index=False)
        assert set(out.index) == {"score", "label"}

    def test_dtype_column_present(self, df):
        """Happy: 'dtype' column always present"""
        out = describe(df, return_df=True, reset_index=False)
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


class TestDescribeSadPath:
    """Sad-path tests for describe()"""

    def test_over_limit_no_subsample_returns_none(self):
        """Sad: df exceeds memory limit with subsample=False → returns None"""
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        out = describe(df, limit=0, subsample=False, return_df=True)
        assert out is None
