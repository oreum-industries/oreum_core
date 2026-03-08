# Copyright 2026 Oreum Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for eda.calc"""

import numpy as np
import pandas as pd
import pytest

from oreum_core.eda.calc import (
    bootstrap,
    bootstrap_index_only,
    bootstrap_lr,
    calc_geometric_cv,
    calc_location_in_ecdf,
    calc_svd,
    month_diff,
    tril_nan,
)

RNG = np.random.default_rng(seed=42)


@pytest.fixture
def arr():
    """1D array of 50 standard-normal values"""
    return RNG.standard_normal(50)


@pytest.fixture
def df_lr():
    """DataFrame with premium and claim columns for loss-ratio bootstrap tests"""
    return pd.DataFrame(
        {"premium": [100.0, 200.0, 150.0, 250.0], "claim": [50.0, 0.0, np.nan, 200.0]}
    )


@pytest.fixture
def df_numeric():
    """30×5 DataFrame of standard-normal values"""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(rng.standard_normal((30, 5)), columns=list("abcde"))


class TestBootstrapIndexOnly:
    """Tests for bootstrap_index_only()"""

    def test_shape(self, arr):
        """Happy: output shape is (len(a), len(a)) when nboot is None"""
        idx = bootstrap_index_only(arr)
        assert idx.shape == (len(arr), len(arr))

    def test_shape_custom_nboot(self, arr):
        """Happy: output shape is (len(a), nboot) when nboot is specified"""
        idx = bootstrap_index_only(arr, nboot=100)
        assert idx.shape == (len(arr), 100)

    def test_sad_non_array(self):
        """Sad: raises ValueError when passed a list"""
        with pytest.raises(ValueError):
            bootstrap_index_only([1, 2, 3])


class TestBootstrap:
    """Tests for bootstrap()"""

    def test_no_summary_fn_shape(self, arr):
        """Happy: without summary_fn returns 2D array of shape (len(a), nboot)"""
        result = bootstrap(arr, nboot=200)
        assert result.shape == (len(arr), 200)

    def test_with_mean_summary_fn(self, arr):
        """Happy: np.mean summary_fn returns 1D array of length nboot"""
        result = bootstrap(arr, nboot=200, summary_fn=np.mean)
        assert result.shape == (200,)

    def test_values_drawn_from_input(self, arr):
        """Happy: all resampled values exist in original array"""
        result = bootstrap(arr, nboot=50)
        assert np.all(np.isin(result, arr))


class TestBootstrapLR:
    """Tests for bootstrap_lr()"""

    def test_output_columns(self, df_lr):
        """Happy: output DataFrame has premium_sum, claim_sum, lr columns"""
        result = bootstrap_lr(df_lr, prm="premium", clm="claim", nboot=100)
        assert set(result.columns) == {"premium_sum", "claim_sum", "lr"}

    def test_output_length(self, df_lr):
        """Happy: output has nboot rows"""
        result = bootstrap_lr(df_lr, prm="premium", clm="claim", nboot=100)
        assert len(result) == 100

    def test_premium_sum_positive(self, df_lr):
        """Happy: resampled premium_sum always positive"""
        result = bootstrap_lr(df_lr, prm="premium", clm="claim", nboot=100)
        assert (result["premium_sum"] > 0).all()


class TestCalcGeometricCV:
    """Tests for calc_geometric_cv()"""

    def test_output_shape(self):
        """Happy: output shape matches number of observations (rows)"""
        yhat = np.exp(RNG.standard_normal((20, 100)))
        result = calc_geometric_cv(yhat)
        assert result.shape == (20,)

    def test_constant_samples_gives_zero(self):
        """Happy: identical samples → std of log = 0 → geometric CV = 0"""
        yhat = np.ones((5, 50))
        result = calc_geometric_cv(yhat)
        np.testing.assert_allclose(result, 0.0)


class TestCalcLocationInEcdf:
    """Tests for calc_location_in_ecdf()"""

    def test_output_values_in_unit_interval(self):
        """Happy: all outputs lie in (0, 1]"""
        baseline = RNG.standard_normal(100)
        test = RNG.standard_normal(10)
        result = calc_location_in_ecdf(baseline, test)
        assert np.all(result > 0) and np.all(result <= 1)

    def test_median_test_value_near_midpoint_of_ecdf(self):
        """Happy: test value at median of baseline → ECDF position ≈ 0.5"""
        baseline = np.arange(1, 101, dtype=float)
        test = np.array([50.0])
        result = calc_location_in_ecdf(baseline, test)
        assert 0.4 < result[0] < 0.6


class TestMonthDiff:
    """Tests for month_diff()"""

    def test_positive_diff(self):
        """Happy: 3 months forward → 3"""
        a = pd.Series(pd.to_datetime(["2024-01-15"]))
        b = pd.Series(pd.to_datetime(["2024-04-15"]))
        result = month_diff(a, b)
        assert result.iloc[0] == 3

    def test_same_month_gives_zero(self):
        """Happy: same year-month → 0"""
        a = pd.Series(pd.to_datetime(["2024-06-01"]))
        b = pd.Series(pd.to_datetime(["2024-06-30"]))
        result = month_diff(a, b)
        assert result.iloc[0] == 0

    def test_series_name(self):
        """Happy: output series has the supplied series_name"""
        a = pd.Series(pd.to_datetime(["2023-01-01"]))
        b = pd.Series(pd.to_datetime(["2024-01-01"]))
        result = month_diff(a, b, series_name="gap")
        assert result.name == "gap"


class TestTrilNan:
    """Tests for tril_nan()"""

    def test_upper_triangle_is_nan(self):
        """Happy: elements above diagonal are NaN"""
        m = np.ones((4, 4))
        result = tril_nan(m, k=0)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 3])

    def test_lower_triangle_preserved(self):
        """Happy: elements on and below diagonal match the original"""
        m = np.arange(9, dtype=float).reshape(3, 3)
        result = tril_nan(m, k=0)
        assert result[2, 0] == m[2, 0]
        assert result[1, 1] == m[1, 1]


class TestCalcSvd:
    """Tests for calc_svd()"""

    def test_output_shapes(self, df_numeric):
        """Happy: transformed array has shape (nrows, k) and svd has k components"""
        k = 3
        dfx, svd_fit = calc_svd(df_numeric, k=k)
        assert dfx.shape == (len(df_numeric), k)
        assert svd_fit.n_components == k

    def test_null_rows_excluded(self, df_numeric):
        """Happy: rows containing nulls are dropped before SVD"""
        df_with_null = df_numeric.copy()
        df_with_null.iloc[0, 0] = np.nan
        dfx, _ = calc_svd(df_with_null, k=3)
        assert dfx.shape[0] == len(df_numeric) - 1
