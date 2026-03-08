"""Tests for model_pymc.base.BasePYMCModel"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import arviz as az
    import pymc as pm

    from oreum_core.model_pymc.base import BasePYMCModel

    HAS_PYMC = True
except Exception:
    BasePYMCModel = None  # type: ignore[assignment,misc]
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


class SimpleModel(BasePYMCModel):
    """Minimal concrete subclass for testing"""

    name = "simple"
    version = "1.0"
    obs_nm = "test_obs"

    def _build(self, **kwargs):
        with pm.Model() as self.model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            pm.Normal("obs", mu=mu, sigma=1, observed=np.zeros(10))


@pytest.fixture
def simple_model():
    """Unbuilt SimpleModel instance"""
    return SimpleModel()


@pytest.fixture
def built_model():
    """Built SimpleModel instance"""
    m = SimpleModel()
    m.build()
    return m


@pytest.fixture
def fake_idata():
    """Minimal InferenceData with a posterior group"""
    rng = np.random.default_rng(42)
    return az.from_dict(posterior={"mu": rng.normal(size=(4, 100))})


class TestBasePYMCModelProperties:
    """Tests for mdl_id, mdl_id_fn, idata and posterior properties"""

    def test_mdl_id_defaults(self):
        """Happy: no name/version/obs_nm attrs yield fallback mdl_id"""

        class Anon(BasePYMCModel):
            def _build(self, **kwargs):
                pass

        assert Anon().mdl_id == "unnamed_model_vunversioned_model_unnamed_obs"

    def test_mdl_id_with_set_attrs(self, simple_model):
        """Happy: name/version/obs_nm attrs compose expected mdl_id"""
        assert simple_model.mdl_id == "simple_v1.0_test_obs"

    def test_mdl_id_fn_replaces_dots(self, simple_model):
        """Happy: mdl_id_fn contains no dots (replaced with hyphens)"""
        assert "." not in simple_model.mdl_id_fn

    def test_idata_raises_before_update(self, simple_model):
        """Sad: idata accessed before update_idata → AssertionError"""
        with pytest.raises(AssertionError):
            _ = simple_model.idata

    def test_posterior_raises_before_sampling(self, simple_model):
        """Sad: posterior accessed before sampling → AssertionError or AttributeError"""
        with pytest.raises((AssertionError, AttributeError)):
            _ = simple_model.posterior


class TestBasePYMCModelBuild:
    """Tests for build(), extend_build(), get_rvs() and debug()"""

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

    def test_get_rvs_returns_expected_keys(self, built_model):
        """Happy: get_rvs() returns dict with all six RV categories"""
        rvs = built_model.get_rvs()
        assert set(rvs) == {
            "basic",
            "unobserved",
            "observed",
            "free",
            "potentials",
            "deterministics",
        }

    def test_debug_no_model_returns_zero_checks(self, simple_model):
        """Happy: debug() before build → zero-check string"""
        assert simple_model.debug() == "Ran 0 checks: []"

    def test_debug_with_model_runs_three_checks(self, built_model):
        """Happy: debug() after build → three-check string (mocked to avoid C compilation)"""
        built_model.model.debug = MagicMock(return_value=None)
        with patch("oreum_core.model_pymc.base.assert_no_rvs"):
            result = built_model.debug()
        assert "Ran 3 checks" in result


class TestBasePYMCModelUpdateIdata:
    """Tests for update_idata()"""

    def test_first_call_sets_idata(self, simple_model, fake_idata):
        """Happy: first update_idata() assigns _idata"""
        simple_model.update_idata(fake_idata)
        assert simple_model._idata is fake_idata

    def test_second_call_extends_idata(self, simple_model, fake_idata):
        """Happy: second update_idata() extends rather than replaces"""
        simple_model.update_idata(fake_idata)
        simple_model.update_idata(fake_idata)
        assert simple_model._idata is not None

    def test_replace_true_uses_right_join(self, simple_model, fake_idata):
        """Happy: replace=True passes join='right' to extend()"""
        simple_model.update_idata(fake_idata)
        rng = np.random.default_rng(0)
        idata2 = az.from_dict(posterior={"mu": rng.normal(size=(4, 100))})
        simple_model.update_idata(idata2, replace=True)
        assert simple_model._idata is not None


class TestBasePYMCModelReplaceObs:
    """Tests for replace_obs()"""

    def test_sets_attribute_on_model(self, simple_model):
        """Happy: replace_obs sets each obsd key as an attribute"""
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3]})
        simple_model.replace_obs(obsd={"dfx": df})
        assert hasattr(simple_model, "dfx")

    def test_updates_obs_nm(self, simple_model):
        """Happy: replace_obs with obs_nm updates model.obs_nm"""
        simple_model.replace_obs(obsd={}, obs_nm="new_dataset")
        assert simple_model.obs_nm == "new_dataset"


class TestBasePYMCModelSample:
    """Tests for sample_prior_predictive() and sample()"""

    def test_sample_prior_predictive_updates_idata(self, built_model):
        """Happy: sample_prior_predictive() populates _idata"""
        rng = np.random.default_rng(42)
        fake = az.from_dict(prior={"mu": rng.normal(size=(1, 500))})
        with patch("pymc.sample_prior_predictive", return_value=fake):
            built_model.sample_prior_predictive()
        assert built_model._idata is not None

    def test_sample_updates_idata_with_posterior(self, built_model):
        """Happy: sample() populates _idata containing a posterior group"""
        rng = np.random.default_rng(42)
        fake_posterior = az.from_dict(posterior={"mu": rng.normal(size=(4, 100))})
        with patch("pymc.sample", return_value=fake_posterior):
            built_model.sample(tune=10, draws=10, chains=1, cores=1, progressbar=False)
        assert built_model._idata is not None
        assert "posterior" in built_model._idata

    def test_sample_invalid_sampler_raises(self, built_model):
        """Sad: unsupported nuts_sampler → NotImplementedError"""
        built_model.sample_kws["nuts_sampler"] = "invalid_sampler"
        with pytest.raises(NotImplementedError):
            built_model.sample()
