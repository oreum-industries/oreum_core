"""Shared fixtures for model_pymc tests"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import arviz as az
    import pymc as pm

    from oreum_core.model_pymc.base import BasePYMCModel

    class SimpleModel(BasePYMCModel):
        """Minimal concrete subclass for testing"""

        name = "simple"
        version = "1.0"
        obs_nm = "test_obs"

        def _build(self, **kwargs):
            with pm.Model() as self.model:
                mu = pm.Normal("mu", mu=0, sigma=1)
                pm.Normal("obs", mu=mu, sigma=1, observed=np.zeros(10))

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to prevent memory accumulation"""
    yield
    plt.close("all")


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
