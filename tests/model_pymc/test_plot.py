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

"""Tests for model_pymc.plot"""

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pandas as pd
import pytest
from matplotlib import figure

try:
    from oreum_core.model_pymc.plot import plot_coverage, plot_estimate, plot_rmse_range

    HAS_PYMC = True
except Exception:
    HAS_PYMC = False

pytestmark = pytest.mark.skipif(not HAS_PYMC, reason="pymc not loadable")


RNG = np.random.default_rng(seed=42)
N = 40


@pytest.fixture(scope="module")
def df_coverage() -> pd.DataFrame:
    """Coverage DataFrame as produced by calc_ppc_coverage."""
    crs = np.round(np.arange(0, 1.01, 0.1), 2)
    rows = [
        (method, cr, cr * 0.9) for method in ["pin_left", "middle_out"] for cr in crs
    ]
    return pd.DataFrame(rows, columns=["method", "cr", "coverage"])


class TestPlotCoverage:
    """Tests for plot_coverage()"""

    def test_returns_figure(self, df_coverage):
        """Happy: returns a Figure for a valid coverage DataFrame"""
        f = plot_coverage(df_coverage)
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_coverage(self, df_coverage):
        """Happy: suptitle mentions 'Coverage'"""
        f = plot_coverage(df_coverage)
        assert "Coverage" in f._suptitle.get_text()

    def test_txtadd_in_suptitle(self, df_coverage):
        """Happy: txtadd kwarg appears in suptitle"""
        f = plot_coverage(df_coverage, txtadd="my model")
        assert "my model" in f._suptitle.get_text()


@pytest.fixture(scope="module")
def rmse_data() -> tuple:
    """RMSE scalar and quantile Series as produced by calc_rmse(qs=True)."""
    qs = np.round(np.linspace(0, 1, 101), 2)
    rmse_qs = pd.Series(np.linspace(0.5, 0.1, 101), index=qs, name="rmse")
    rmse_qs.index.name = "q"
    return 0.3, rmse_qs


class TestPlotRmseRange:
    """Tests for plot_rmse_range()"""

    def test_returns_figure(self, rmse_data):
        """Happy: returns a Figure for valid rmse scalar and qs Series"""
        rmse, rmse_qs = rmse_data
        f = plot_rmse_range(rmse, rmse_qs)
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_rmse(self, rmse_data):
        """Happy: suptitle mentions 'RMSE'"""
        rmse, rmse_qs = rmse_data
        f = plot_rmse_range(rmse, rmse_qs)
        assert "RMSE" in f._suptitle.get_text()

    def test_txtadd_in_suptitle(self, rmse_data):
        """Happy: txtadd kwarg appears in suptitle"""
        rmse, rmse_qs = rmse_data
        f = plot_rmse_range(rmse, rmse_qs, txtadd="v1")
        assert "v1" in f._suptitle.get_text()


class TestPlotEstimate:
    """Tests for plot_estimate()"""

    def test_returns_figure_boxplot(self):
        """Happy: default boxplot mode returns a Figure"""
        yhat = RNG.standard_normal(500) + 5.0  # ensure positive mean for log10
        f = plot_estimate(yhat, nobs=N)
        assert isinstance(f, figure.Figure)

    def test_returns_figure_exceedance(self):
        """Happy: exceedance=True returns a Figure"""
        yhat = RNG.standard_normal(500) + 5.0
        f = plot_estimate(yhat, nobs=N, exceedance=True)
        assert isinstance(f, figure.Figure)

    def test_with_overplot_y(self):
        """Happy: passing y overlays observed values without error"""
        yhat = RNG.standard_normal(500) + 5.0
        y = RNG.standard_normal(N) + 5.0
        f = plot_estimate(yhat, nobs=N, y=y)
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_boxplot(self):
        """Happy: suptitle contains 'Boxplot' in default mode"""
        yhat = RNG.standard_normal(500) + 5.0
        f = plot_estimate(yhat, nobs=N)
        assert "Boxplot" in f._suptitle.get_text()
