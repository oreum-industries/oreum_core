"""Tests for curate.data_transform.DatatypeConverter"""

import pandas as pd
import pytest

from oreum_core.curate.data_transform import DatatypeConverter


class TestDatatypeConverterConvertDtypes:
    """Tests for DatatypeConverter.convert_dtypes()"""

    def test_fcat_converts_to_unordered_categorical(self):
        """Happy: string column → snl-cleaned unordered pd.Categorical"""
        df = pd.DataFrame({"colour": ["Red", "blue", "RED"]})
        out = DatatypeConverter({"fcat": ["colour"]}).convert_dtypes(df)
        assert out["colour"].dtype.name == "category"
        assert not out["colour"].cat.ordered
        assert set(out["colour"].dropna()) == {"red", "blue"}

    def test_ford_converts_to_ordered_categorical(self):
        """Happy: string column → ordered pd.Categorical with specified levels"""
        df = pd.DataFrame({"size": ["large", "small", "medium"]})
        out = DatatypeConverter(
            {"ford": {"size": ["small", "medium", "large"]}}
        ).convert_dtypes(df)
        assert out["size"].cat.ordered
        assert list(out["size"].cat.categories) == ["small", "medium", "large"]

    def test_fbool_converts_string_representations(self):
        """Happy: bool-like strings → bool dtype"""
        df = pd.DataFrame({"flag": ["yes", "no", "true", "0"]})
        out = DatatypeConverter({"fbool": ["flag"]}).convert_dtypes(df)
        assert out["flag"].dtype == bool
        assert list(out["flag"]) == [True, False, True, False]

    def test_fbool_nan_to_false_fills_nulls(self):
        """Happy: null in fbool_nan_to_false column → False rather than NA"""
        df = pd.DataFrame({"active": ["yes", None, "no"]})
        out = DatatypeConverter({"fbool_nan_to_false": ["active"]}).convert_dtypes(df)
        assert out["active"].dtype == bool
        assert list(out["active"]) == [True, False, False]

    def test_fint_strips_currency_junk(self):
        """Happy: numeric strings with currency/punctuation junk → int"""
        df = pd.DataFrame({"count": ["$1,000", "2,500", "300"]})
        out = DatatypeConverter({"fint": ["count"]}).convert_dtypes(df)
        assert out["count"].dtype == int
        assert list(out["count"]) == [1000, 2500, 300]

    def test_ffloat_strips_currency_junk(self):
        """Happy: numeric strings with currency/punctuation junk → float"""
        df = pd.DataFrame({"price": ["$1,000.50", "€2,500.00"]})
        out = DatatypeConverter({"ffloat": ["price"]}).convert_dtypes(df)
        assert out["price"].dtype == float
        assert list(out["price"]) == pytest.approx([1000.5, 2500.0])

    def test_fdate_parses_iso_strings(self):
        """Happy: ISO date strings → datetime column"""
        df = pd.DataFrame({"dt": ["2024-01-01", "2024-12-31"]})
        out = DatatypeConverter({"fdate": ["dt"]}).convert_dtypes(df)
        assert out["dt"].dtype.kind == "M"
        assert list(out["dt"]) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31"),
        ]

    def test_fcat_null_becomes_na(self):
        """Edge: null values in fcat column → pd.NA, not dropped"""
        df = pd.DataFrame({"label": ["alpha", None, "beta"]})
        out = DatatypeConverter({"fcat": ["label"]}).convert_dtypes(df)
        assert pd.isna(out["label"].iloc[1])


class TestDatatypeConverterConvertDtypesSadPath:
    """Sad-path tests for DatatypeConverter.convert_dtypes()"""

    def test_missing_feature_raises_keyerror(self):
        """Sad: feature listed in ftsd but absent from DataFrame → KeyError"""
        df = pd.DataFrame({"colour": ["red"]})
        with pytest.raises(KeyError):
            DatatypeConverter({"fcat": ["nonexistent"]}).convert_dtypes(df)

    def test_fbool_unmappable_value_raises_valueerror(self):
        """Sad: fbool value that can't map to True/False with no NaNs → ValueError"""
        df = pd.DataFrame({"flag": ["yes", "maybe"]})
        with pytest.raises(ValueError, match="incompatible with np.bool or pd.Boolean"):
            DatatypeConverter({"fbool": ["flag"]}).convert_dtypes(df)

    def test_fint_non_numeric_raises_exception(self):
        """Sad: non-numeric string in fint → Exception wrapping the conversion error"""
        df = pd.DataFrame({"count": ["10", "abc"]})
        with pytest.raises(Exception, match="in ft: count"):
            DatatypeConverter({"fint": ["count"]}).convert_dtypes(df)

    def test_fdate_wrong_format_raises_valueerror(self):
        """Sad: date string in wrong format (dd/mm/yyyy vs default %Y-%m-%d) → ValueError"""
        df = pd.DataFrame({"dt": ["01/01/2024"]})
        with pytest.raises(ValueError):
            DatatypeConverter({"fdate": ["dt"]}).convert_dtypes(df)

    def test_ford_unknown_level_silently_becomes_nan(self):
        """Edge: ford value not in specified levels → NaN (pd.Categorical behaviour)"""
        df = pd.DataFrame({"size": ["small", "unknown"]})
        out = DatatypeConverter(
            {"ford": {"size": ["small", "medium", "large"]}}
        ).convert_dtypes(df)
        assert pd.isna(out["size"].iloc[1])
