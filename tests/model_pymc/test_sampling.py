"""Tests for BasePYMCModel sampling: sample_prior_predictive, sample"""

from unittest.mock import patch

import numpy as np
import pytest

try:
    import arviz as az

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


class TestSampling:
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
