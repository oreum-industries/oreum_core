"""Tests for BasePYMCModel properties: mdl_id, mdl_id_fn, idata, posterior"""

import pytest

try:
    from oreum_core.model_pymc.base import BasePYMCModel

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")


class TestProperties:
    """Tests for BasePYMCModel properties"""

    def test_mdl_id_defaults(self):
        """Happy: missing name/version/obs_nm attrs yield fallback mdl_id"""

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
