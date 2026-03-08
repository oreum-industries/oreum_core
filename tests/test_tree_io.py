"""Tests for model_tree.tree_io.XGBIO"""

import pytest

try:
    from oreum_core.model_tree.tree_io import XGBIO

    HAS_XGB = True
except Exception:
    XGBIO = None  # type: ignore[assignment,misc]
    HAS_XGB = False

pytestmark = pytest.mark.skipif(not HAS_XGB, reason="xgboost not loadable")


class TestXGBIOPaths:
    """Tests for XGBIO path helpers"""

    def test_get_sqlite_uri_default_fn(self, tmp_path):
        """Happy: default fn → sqlite URI containing 'optuna_study'"""
        io = XGBIO(rootdir=tmp_path)
        result = io.get_sqlite_uri_for_optuna_study()
        assert result.startswith("sqlite:////")
        assert "optuna_study" in result

    def test_get_sqlite_uri_custom_fn(self, tmp_path):
        """Happy: custom fn → URI contains sanitized name"""
        io = XGBIO(rootdir=tmp_path)
        result = io.get_sqlite_uri_for_optuna_study("my study")
        assert "my_study" in result

    def test_read_missing_file_raises(self, tmp_path):
        """Sad: file not found → FileNotFoundError"""
        io = XGBIO(rootdir=tmp_path)
        with pytest.raises(FileNotFoundError):
            io.read("nonexistent")


class TestXGBIORoundtrip:
    """Round-trip read/write tests for XGBIO"""

    def test_write_returns_path(self, tmp_path):
        """Happy: write Booster → returns Path with .json suffix"""
        import numpy as np
        import xgboost as xgb

        rng = np.random.default_rng(42)
        dtrain = xgb.DMatrix(rng.normal(size=(20, 2)), label=rng.integers(0, 2, 20))
        bst = xgb.train(
            {"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=1
        )
        io = XGBIO(rootdir=tmp_path)
        result = io.write(bst, "mybst")
        assert result.suffix == ".json"
        assert result.exists()

    def test_roundtrip(self, tmp_path):
        """Happy: write then read returns Booster of same type"""
        import numpy as np
        import xgboost as xgb

        rng = np.random.default_rng(0)
        dtrain = xgb.DMatrix(rng.normal(size=(20, 2)), label=rng.integers(0, 2, 20))
        bst = xgb.train(
            {"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=1
        )
        io = XGBIO(rootdir=tmp_path)
        io.write(bst, "mybst")
        bst2 = io.read("mybst")
        assert isinstance(bst2, xgb.core.Booster)
