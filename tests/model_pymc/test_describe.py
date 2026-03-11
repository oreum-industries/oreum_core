# Copyright 2026 Oreum Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for model_pymc.describe"""

import numpy as np
import patsy
import pytest

try:
    import arviz as az

    from oreum_core.model_pymc.describe import extract_yobs_yhat, model_desc, print_rvs

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not loadable")


RNG = np.random.default_rng(seed=42)
NOBS, NCHAINS, NDRAWS = 10, 4, 100


@pytest.fixture(scope="module")
def idata_ppc() -> az.InferenceData:
    """Minimal InferenceData with posterior_predictive and constant_data groups."""
    return az.from_dict(
        posterior_predictive={"yhat": RNG.normal(size=(NCHAINS, NDRAWS, NOBS))},
        constant_data={"y": RNG.normal(size=(NOBS,))},
    )


class TestModelDesc:
    """Tests for model_desc()"""

    def test_returns_string_with_patsy_header(self):
        """Happy: returns a string containing the patsy header"""
        result = model_desc("y ~ x")
        assert isinstance(result, str)
        assert "patsy linear model desc" in result

    def test_rhs_only_no_intercept(self):
        """Happy: RHS-only formula (no tilde, no '1 +') processed without error"""
        result = model_desc("x + z")
        assert isinstance(result, str)

    def test_rhs_only_with_intercept(self):
        """Happy: RHS-only formula starting with '1 +' re-inserts intercept marker"""
        result = model_desc("1 + x + z")
        assert "1 +" in result

    def test_two_tildes_raises(self):
        """Sad: formula with two tildes raises PatsyError (patsy rejects it before
        our ValueError guard is reached)"""
        with pytest.raises(patsy.PatsyError):
            model_desc("y ~ x ~ z")


class TestExtractYobsYhat:
    """Tests for extract_yobs_yhat()"""

    def test_returns_tuple(self, idata_ppc):
        """Happy: returns a two-element tuple"""
        result = extract_yobs_yhat(idata_ppc, obs="y", pred="yhat")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_yobs_and_yhat_shapes(self, idata_ppc):
        """Happy: yobs is 1D with length nobs; yhat is 2D with shape (nchains*ndraws, nobs)"""
        yobs, yhat = extract_yobs_yhat(idata_ppc, obs="y", pred="yhat")
        assert yobs.ndim == 1
        assert yobs.shape[0] == NOBS
        assert yhat.ndim == 2
        assert yhat.shape == (NCHAINS * NDRAWS, NOBS)


class TestPrintRvs:
    """Tests for print_rvs()"""

    def test_returns_list(self, built_model):
        """Happy: returns a list for a built model"""
        result = print_rvs(built_model)
        assert isinstance(result, list)

    def test_list_contains_strings(self, built_model):
        """Happy: all items in the returned list are strings"""
        result = print_rvs(built_model)
        assert all(isinstance(s, str) for s in result)

    def test_free_rv_names_present(self, built_model):
        """Happy: free RV names from the model appear in the output"""
        result = print_rvs(built_model)
        joined = " ".join(result)
        assert "mu" in joined
