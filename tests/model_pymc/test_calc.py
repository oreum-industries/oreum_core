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

"""Tests for model_pymc.calc"""

import numpy as np
import pandas as pd
import pytest

try:
    from oreum_core.model_pymc.calc import (
        calc_2_sample_delta_prop,
        calc_bayesian_r2,
        calc_f_beta,
        calc_r2,
        expand_packed_triangular,
        numpy_invlogit,
    )

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not loadable")


class TestNumpyInvlogit:
    """Tests for numpy_invlogit()"""

    def test_zero_returns_near_half(self):
        """Happy: invlogit(0) ≈ 0.5"""
        assert abs(numpy_invlogit(0.0) - 0.5) < 1e-6

    def test_large_positive_approaches_one(self):
        """Happy: invlogit(large positive) → close to 1"""
        assert numpy_invlogit(100.0) > 0.999

    def test_large_negative_approaches_zero(self):
        """Happy: invlogit(large negative) → close to 0"""
        assert numpy_invlogit(-100.0) < 0.001

    def test_array_input_monotone(self):
        """Happy: array input returns same shape, monotonically increasing"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = numpy_invlogit(x)
        assert result.shape == x.shape
        assert (result[1:] > result[:-1]).all()


class TestCalcFBeta:
    """Tests for calc_f_beta()"""

    def test_f1_equal_precision_recall(self):
        """Happy: when precision==recall==p, f1==p"""
        assert abs(calc_f_beta(0.6, 0.6, beta=1.0) - 0.6) < 1e-9

    def test_zero_inputs_return_zero(self):
        """Edge: zero precision and recall (as arrays) → 0 (not nan)"""
        result = calc_f_beta(np.array([0.0]), np.array([0.0]))
        assert result[0] == 0.0

    def test_beta_2_emphasises_recall(self):
        """Happy: beta=2 weights recall → f2 > f1 when recall > precision"""
        p, r = 0.4, 0.8
        assert calc_f_beta(p, r, beta=2.0) > calc_f_beta(p, r, beta=1.0)

    def test_array_inputs(self):
        """Happy: array inputs return array of same shape"""
        p = np.array([0.5, 0.8])
        r = np.array([0.5, 0.6])
        result = calc_f_beta(p, r, beta=1.0)
        assert result.shape == p.shape


class TestExpandPackedTriangular:
    """Tests for expand_packed_triangular()"""

    def test_non_integer_n_raises(self):
        """Sad: float n → TypeError"""
        with pytest.raises(TypeError, match="n must be an integer"):
            expand_packed_triangular(3.0, np.ones(6))

    def test_non_1d_packed_raises(self):
        """Sad: 2D packed → ValueError"""
        with pytest.raises(ValueError):
            expand_packed_triangular(3, np.ones((3, 2)))

    def test_diagonal_only_lower(self):
        """Happy: lower diagonal indices extracted correctly for n=3"""
        # lower triangular n=3: packed positions [0,1,2,3,4,5]
        # diagonal at cumsum([1,2,3])-1 = [0,2,5]
        packed = np.arange(6, dtype=float)
        result = expand_packed_triangular(3, packed, lower=True, diagonal_only=True)
        np.testing.assert_array_equal(result, [0.0, 2.0, 5.0])

    def test_diagonal_only_upper(self):
        """Happy: upper diagonal indices extracted correctly for n=3"""
        # upper triangular n=3: diagonal at positions [0,3,5]
        packed = np.arange(6, dtype=float)
        result = expand_packed_triangular(3, packed, lower=False, diagonal_only=True)
        np.testing.assert_array_equal(result, [0.0, 3.0, 5.0])


class TestCalcBayesianR2:
    """Tests for calc_bayesian_r2()"""

    def test_returns_dataframe_with_r2_column(self):
        """Happy: output is DataFrame with 'r2' column, one row per posterior sample"""
        rng = np.random.default_rng(0)
        nsamples = 50
        y = rng.normal(size=10)
        yhat = y.reshape(-1, 1) + rng.normal(scale=0.1, size=(10, nsamples))
        result = calc_bayesian_r2(y, yhat)
        assert isinstance(result, pd.DataFrame)
        assert "r2" in result.columns
        assert len(result) == nsamples

    def test_good_fit_r2_near_one(self):
        """Happy: near-perfect predictions → mean r2 > 0.9"""
        rng = np.random.default_rng(0)
        y = rng.normal(size=20)
        yhat = y.reshape(-1, 1) + rng.normal(scale=0.01, size=(20, 500))
        assert calc_bayesian_r2(y, yhat)["r2"].mean() > 0.9


class TestCalcR2:
    """Tests for calc_r2()"""

    def test_returns_scalar_and_series(self):
        """Happy: returns (numpy scalar, pd.Series) with index named 'pct'"""
        rng = np.random.default_rng(0)
        y = rng.normal(size=20)
        yhat = y.reshape(-1, 1) + rng.normal(scale=0.1, size=(20, 100))
        r2_mean, r2_pct = calc_r2(y, yhat)
        assert isinstance(r2_pct, pd.Series)
        assert np.ndim(r2_mean) == 0
        assert r2_pct.index.name == "pct"

    def test_good_fit_r2_mean_near_one(self):
        """Happy: near-perfect predictions → r2_mean > 0.9"""
        rng = np.random.default_rng(0)
        y = rng.normal(size=20)
        yhat = y.reshape(-1, 1) + rng.normal(scale=0.01, size=(20, 500))
        r2_mean, _ = calc_r2(y, yhat)
        assert r2_mean > 0.9


class TestCalc2SampleDeltaProp:
    """Tests for calc_2_sample_delta_prop()"""

    def test_proportions_sum_to_one(self):
        """Happy: proportions across categories sum to 1 per row"""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 200))
        aref = rng.normal(size=(5, 200))
        result = calc_2_sample_delta_prop(a, aref)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_1d_input_auto_reshaped(self):
        """Edge: 1D a is silently treated as a single-row 2D array"""
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        aref = rng.normal(size=(5, 200))
        result = calc_2_sample_delta_prop(a, aref)
        assert result.shape == (1, 3)

    def test_with_a_index_returns_dataframe(self):
        """Happy: a_index provided → DataFrame with correct column names"""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 200))
        aref = rng.normal(size=(5, 200))
        result = calc_2_sample_delta_prop(a, aref, a_index=pd.Index(["x", "y"]))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["subs_lower", "no_difference", "subs_higher"]

    def test_fully_vectorised_matches_loop(self):
        """Happy: fully_vectorised=True matches loop result"""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 200))
        aref = rng.normal(size=(4, 200))
        r_loop = calc_2_sample_delta_prop(a, aref, fully_vectorised=False)
        r_vec = calc_2_sample_delta_prop(a, aref, fully_vectorised=True)
        np.testing.assert_array_almost_equal(r_loop, r_vec)
