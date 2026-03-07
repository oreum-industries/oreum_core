"""Tests for curate.data_transform.DatasetReshaper"""

import pandas as pd
import pytest

from oreum_core.curate.data_transform import DatasetReshaper


@pytest.fixture
def reshaper():
    return DatasetReshaper()


class TestDatasetReshaperCreateDfcmb:
    """Happy-path tests for DatasetReshaper.create_dfcmb()"""

    def test_categorical_column_lists_all_categories(self, reshaper):
        """Happy: category column → dfcmb contains all category values"""
        df = pd.DataFrame(
            {"colour": pd.Categorical(["red", "blue", "green"], ordered=False)}
        )
        out = reshaper.create_dfcmb(df)
        assert set(out["colour"].dropna()) == {"red", "blue", "green"}

    def test_bool_column_has_false_and_true(self, reshaper):
        """Happy: bool column → dfcmb has exactly [False, True]"""
        df = pd.DataFrame({"flag": pd.array([True, False, True], dtype=bool)})
        out = reshaper.create_dfcmb(df)
        assert list(out["flag"]) == [False, True]

    def test_int_column_filled_with_one(self, reshaper):
        """Happy: int column → dfcmb filled with 1"""
        df = pd.DataFrame({"count": pd.array([10, 20, 30], dtype=int)})
        out = reshaper.create_dfcmb(df)
        assert out["count"].iloc[0] == 1

    def test_float_column_filled_with_one(self, reshaper):
        """Happy: float column → dfcmb filled with 1.0"""
        df = pd.DataFrame({"score": pd.array([1.5, 2.5, 3.5], dtype=float)})
        out = reshaper.create_dfcmb(df)
        assert out["score"].iloc[0] == pytest.approx(1.0)

    def test_ragged_output_for_unequal_category_counts(self, reshaper):
        """Edge: two cat cols with different numbers of levels → NaN padding"""
        df = pd.DataFrame(
            {
                "size": pd.Categorical(["s", "m", "l"], ordered=False),
                "flag": pd.Categorical(["yes", "no", "yes"], ordered=False),
            }
        )
        out = reshaper.create_dfcmb(df)
        assert len(out) == 3  # max(3, 2) levels
        assert pd.isna(out["flag"].iloc[2])  # shorter col (2 levels) padded with NaN


class TestDatasetReshaperCreateDfcmbSadPath:
    """Sad-path tests for DatasetReshaper.create_dfcmb()"""

    def test_object_dtype_raises_valueerror(self, reshaper):
        """Sad: object dtype column → raises ValueError"""
        df = pd.DataFrame({"label": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="Valid dtypes are"):
            reshaper.create_dfcmb(df)

    def test_nullable_boolean_dtype_raises_valueerror(self, reshaper):
        """Sad: pd.BooleanDtype (nullable) column → raises ValueError"""
        df = pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean")})
        with pytest.raises(ValueError, match="Valid dtypes are"):
            reshaper.create_dfcmb(df)
