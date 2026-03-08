"""Tests for BasePYMCModel build lifecycle: build, extend_build, get_rvs, debug"""

from unittest.mock import MagicMock, patch

import pytest

try:
    import pymc as pm

    from oreum_core.model_pymc.base import BasePYMCModel

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


class TestBuild:
    """Tests for build() and extend_build()"""

    def test_build_assigns_pm_model(self, simple_model):
        """Happy: build() sets self.model to a pm.Model instance"""
        simple_model.build()
        assert isinstance(simple_model.model, pm.Model)

    def test_build_raises_without_build_method(self):
        """Sad: subclass missing _build → NotImplementedError"""

        class NoBuild(BasePYMCModel):
            pass

        with pytest.raises(NotImplementedError):
            NoBuild().build()

    def test_extend_build_raises_without_method(self, simple_model):
        """Sad: subclass missing _extend_build → NotImplementedError"""
        with pytest.raises(NotImplementedError):
            simple_model.extend_build()


class TestGetRVs:
    """Tests for get_rvs()"""

    def test_returns_all_six_rv_categories(self, built_model):
        """Happy: get_rvs() returns dict with all six RV category keys"""
        rvs = built_model.get_rvs()
        assert set(rvs) == {
            "basic",
            "unobserved",
            "observed",
            "free",
            "potentials",
            "deterministics",
        }


class TestDebug:
    """Tests for debug()"""

    def test_no_model_returns_zero_checks(self, simple_model):
        """Happy: debug() before build → zero-check string"""
        assert simple_model.debug() == "Ran 0 checks: []"

    def test_with_model_runs_three_checks(self, built_model):
        """Happy: debug() after build → three-check string (mocked to avoid C compilation)"""
        built_model.model.debug = MagicMock(return_value=None)
        with patch("oreum_core.model_pymc.base.assert_no_rvs"):
            result = built_model.debug()
        assert "Ran 3 checks" in result
