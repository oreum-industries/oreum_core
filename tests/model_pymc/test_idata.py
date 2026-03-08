"""Tests for BasePYMCModel InferenceData management: update_idata, replace_obs"""

import numpy as np
import pytest

try:
    import arviz as az

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


class TestUpdateIdata:
    """Tests for update_idata()"""

    def test_first_call_sets_idata(self, simple_model, fake_idata):
        """Happy: first update_idata() assigns _idata directly"""
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


class TestReplaceObs:
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
