"""Tests for curate.data_transform.Transformer"""

import pandas as pd
import patsy
import pytest

from oreum_core.curate.data_transform import Transformer


@pytest.fixture
def tfmr():
    """Default Transformer instance"""
    return Transformer()


class TestTransformerFitTransform:
    """Happy-path tests for Transformer.fit_transform()"""

    def test_fit_transform_basic_properties(self, tfmr):
        """Happy: float column → returns DataFrame with Intercept; populates design_info"""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        assert tfmr.design_info is None
        out = tfmr.fit_transform("score", df)
        assert isinstance(out, pd.DataFrame)
        assert "Intercept" in out.columns
        assert tfmr.design_info is not None

    def test_fit_transform_categorical_populates_factor_map(self, tfmr):
        """Happy: categorical column → factor_map entry maps level names to codes"""
        df = pd.DataFrame({"colour": pd.Categorical(["red", "blue", "green"])})
        tfmr.fit_transform("colour", df)
        assert "colour" in tfmr.factor_map
        assert set(tfmr.factor_map["colour"].keys()) == {"red", "blue", "green"}

    def test_fit_transform_int_column_stays_int(self, tfmr):
        """Happy: int column → output int column dtype is int (not float)"""
        df = pd.DataFrame({"count": pd.array([1, 2, 3], dtype=int)})
        out = tfmr.fit_transform("count", df)
        assert out["count"].dtype == int

    def test_fit_transform_bool_indicator_forced_to_int(self, tfmr):
        """Happy: bool column → patsy creates flag[T.True] indicator forced to int"""
        df = pd.DataFrame({"flag": pd.array([True, False, True], dtype=bool)})
        out = tfmr.fit_transform("flag", df)
        assert "flag[T.True]" in out.columns
        assert out["flag[T.True]"].dtype == int

    def test_transform_reuses_fitted_design_info(self, tfmr):
        """Happy: transform() on new data produces same columns as fit_transform()"""
        df_train = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        df_new = pd.DataFrame({"score": [4.0, 5.0]})
        out_train = tfmr.fit_transform("score", df_train)
        out_new = tfmr.transform(df_new)
        assert list(out_new.columns) == list(out_train.columns)


class TestTransformerSadPath:
    """Sad-path tests for Transformer"""

    def test_transform_before_fit_raises_attributeerror(self, tfmr):
        """Sad: transform() before fit_transform() → AttributeError"""
        df = pd.DataFrame({"score": [1.0, 2.0]})
        with pytest.raises(AttributeError, match="No design_info"):
            tfmr.transform(df)

    def test_fit_transform_unknown_column_raises(self, tfmr):
        """Sad: formula references column absent from DataFrame → patsy error"""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        with pytest.raises(patsy.PatsyError):
            tfmr.fit_transform("nonexistent", df)
