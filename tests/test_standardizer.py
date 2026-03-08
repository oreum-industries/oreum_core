"""Tests for curate.data_transform.Standardizer"""

import numpy as np
import pandas as pd
import pytest

from oreum_core.curate.data_transform import Standardizer, Transformer


@pytest.fixture
def fitted():
    """Fitted Transformer, design matrix, and Standardizer for 'score' feature.
    score = [1,2,3,4,5], df_ex columns = ["Intercept", "score"]
    """
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
    tfmr = Transformer()
    df_ex = tfmr.fit_transform("score", df)
    stdz = Standardizer(tfmr)
    return tfmr, df_ex, stdz


class TestStandardizerFitStandardize:
    """Happy-path tests for Standardizer.fit_standardize()"""

    def test_fit_standardize_returns_dataframe(self, fitted):
        """Happy: fit_standardize → returns pd.DataFrame"""
        _, df_ex, stdz = fitted
        out = stdz.fit_standardize(df_ex)
        assert isinstance(out, pd.DataFrame)

    def test_fit_standardize_intercept_column_unchanged(self, fitted):
        """Happy: Intercept is excluded from standardization → remains all 1.0"""
        _, df_ex, stdz = fitted
        out = stdz.fit_standardize(df_ex)
        assert list(out["Intercept"]) == pytest.approx([1.0] * len(df_ex))

    def test_fit_standardize_numeric_col_is_mean_zero(self, fitted):
        """Happy: standardized numeric column has mean ≈ 0"""
        _, df_ex, stdz = fitted
        out = stdz.fit_standardize(df_ex)
        assert out["score"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_fit_standardize_default_scale_is_2(self, fitted):
        """Happy: default scale=2 → get_scale returns scale 2"""
        _, df_ex, stdz = fitted
        stdz.fit_standardize(df_ex)
        _, scale = stdz.get_scale()
        assert scale == 2

    def test_fts_exclude_passes_through_unchanged(self, fitted):
        """Happy: fts_exclude=['score'] → score column not standardized"""
        tfmr, df_ex, _ = fitted
        stdz_excl = Standardizer(tfmr, fts_exclude=["score"])
        out = stdz_excl.fit_standardize(df_ex)
        assert list(out["score"]) == pytest.approx(list(df_ex["score"]))

    def test_set_scale_then_standardize_matches_fit(self, fitted):
        """Happy: set_scale() + standardize() reproduces fit_standardize() output"""
        tfmr, df_ex, stdz = fitted
        out_fit = stdz.fit_standardize(df_ex)
        ms, scale = stdz.get_scale()

        stdz2 = Standardizer(tfmr)
        stdz2.set_scale(ms, scale)
        out_set = stdz2.standardize(df_ex)

        pd.testing.assert_frame_equal(out_fit, out_set)

    def test_fit_standardize_mx_returns_ndarray(self, fitted):
        """Happy: fit_standardize_mx → returns np.ndarray of same shape"""
        _, df_ex, stdz = fitted
        mx = df_ex.values
        out = stdz.fit_standardize_mx(mx)
        assert isinstance(out, np.ndarray)
        assert out.shape == mx.shape


class TestStandardizerSadPath:
    """Sad-path tests for Standardizer"""

    def test_standardize_before_fit_raises_attributeerror(self, fitted):
        """Sad: standardize() before fit_standardize() → AttributeError"""
        tfmr, df_ex, _ = fitted
        stdz_unfitted = Standardizer(tfmr)
        with pytest.raises(AttributeError, match="fit_standardize"):
            stdz_unfitted.standardize(df_ex)

    def test_standardize_mx_before_fit_raises_attributeerror(self, fitted):
        """Sad: standardize_mx() before fit_standardize() → AttributeError"""
        tfmr, df_ex, _ = fitted
        stdz_unfitted = Standardizer(tfmr)
        with pytest.raises(AttributeError, match="fit_standardize"):
            stdz_unfitted.standardize_mx(df_ex.values)
