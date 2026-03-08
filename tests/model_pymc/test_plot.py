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
    from oreum_core.model_pymc.plot import (
        plot_accuracy,
        plot_binary_performance,
        plot_coverage,
        plot_estimate,
        plot_f_measure,
        plot_rmse_range,
        plot_roc_precrec,
    )

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

    def test_returns_figure_with_two_axes(self, df_coverage):
        """Happy: returns a Figure with one axes per method column (2 methods in fixture)"""
        f = plot_coverage(df_coverage)
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 2

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

    def test_returns_figure_with_one_axis(self, rmse_data):
        """Happy: returns a Figure with exactly one axes"""
        rmse, rmse_qs = rmse_data
        f = plot_rmse_range(rmse, rmse_qs)
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 1

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

    def test_boxplot_mode(self):
        """Happy: default boxplot mode returns a Figure with correct suptitle"""
        yhat = RNG.standard_normal(500) + 5.0  # ensure positive mean for log10
        f = plot_estimate(yhat, nobs=N)
        assert isinstance(f, figure.Figure)
        assert "Boxplot" in f._suptitle.get_text()

    def test_exceedance_mode(self):
        """Happy: exceedance=True returns a Figure with correct suptitle"""
        yhat = RNG.standard_normal(500) + 5.0
        f = plot_estimate(yhat, nobs=N, exceedance=True)
        assert isinstance(f, figure.Figure)
        assert "Exceedance Curve" in f._suptitle.get_text()

    def test_overplot_y_and_force_xlim(self):
        """Happy: y overlay and force_xlim both return a Figure without error"""
        yhat = RNG.standard_normal(500) + 5.0
        y = RNG.standard_normal(N) + 5.0
        assert isinstance(plot_estimate(yhat, nobs=N, y=y), figure.Figure)
        assert isinstance(
            plot_estimate(yhat, nobs=N, force_xlim=[0, 15]), figure.Figure
        )

    def test_txtadd_in_suptitle(self):
        """Happy: txtadd kwarg appears in suptitle"""
        yhat = RNG.standard_normal(500) + 5.0
        f = plot_estimate(yhat, nobs=N, txtadd="v2")
        assert "v2" in f._suptitle.get_text()


@pytest.fixture(scope="module")
def df_roc() -> pd.DataFrame:
    """DataFrame with ROC/PrecRec curve columns"""
    thresholds = np.linspace(0, 1, 20)
    return pd.DataFrame(
        {
            "fpr": np.linspace(0, 1, 20),
            "tpr": np.clip(np.linspace(0, 1, 20) ** 0.5, 0, 1),
            "recall": np.linspace(0, 1, 20),
            "precision": np.linspace(1.0, 0.5, 20),
        },
        index=thresholds,
    )


@pytest.fixture(scope="module")
def df_perf() -> pd.DataFrame:
    """Performance metrics DataFrame with integer index named 'pct'.
    Integer index required so argmax() position == label for df.loc[] calls.
    Columns mirror those produced by calc_binary_performance_measures.
    """
    n = 20
    idx = pd.Index(range(n), name="pct")
    return pd.DataFrame(
        {
            "accuracy": np.linspace(0.5, 0.85, n),
            "f0.5": np.linspace(0.1, 0.75, n),
            "f1": np.linspace(0.1, 0.75, n),
            "f2": np.linspace(0.1, 0.75, n),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def df_binary_perf() -> pd.DataFrame:
    """DataFrame as produced by calc_binary_performance_measures.
    101-row float q-index (0.00..1.00) so np.round(argmax()/100, 2) gives a valid .loc[] label.
    """
    qs = np.round(np.linspace(0, 1, 101), 2)
    idx = pd.Index(qs, name="q")
    n = len(qs)
    return pd.DataFrame(
        {
            "fpr": np.linspace(1, 0, n),
            "tpr": np.linspace(0, 1, n),
            "precision": np.linspace(0.5, 1, n),
            "recall": np.linspace(1, 0, n),
            "f0.5": np.linspace(0.1, 0.8, n),
            "f1": np.linspace(0.1, 0.8, n),
            "f2": np.linspace(0.1, 0.8, n),
            "accuracy": np.linspace(0.5, 0.9, n),
        },
        index=idx,
    )


class TestPlotRocPrecrec:
    """Tests for plot_roc_precrec()"""

    def test_returns_figure_and_aucs(self, df_roc):
        """Happy: returns (Figure, float, float) with 2 axes for a valid metrics DataFrame"""
        result = plot_roc_precrec(df_roc)
        assert isinstance(result, tuple) and len(result) == 3
        f, roc_auc, pr_auc = result
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 2
        assert 0.0 <= roc_auc <= 1.0
        assert 0.0 <= pr_auc <= 1.0

    def test_suptitle_contains_roc(self, df_roc):
        """Happy: suptitle mentions 'ROC'"""
        f, _, _ = plot_roc_precrec(df_roc)
        assert "ROC" in f._suptitle.get_text()


class TestPlotFMeasure:
    """Tests for plot_f_measure()"""

    def test_returns_figure_with_one_axis(self, df_perf):
        """Happy: returns a Figure with exactly one axes"""
        f = plot_f_measure(df_perf)
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 1

    def test_suptitle_contains_f_scores(self, df_perf):
        """Happy: suptitle mentions 'F-scores'"""
        f = plot_f_measure(df_perf)
        assert "F-scores" in f._suptitle.get_text()


class TestPlotAccuracy:
    """Tests for plot_accuracy()"""

    def test_returns_figure_with_one_axis(self, df_perf):
        """Happy: returns a Figure with exactly one axes"""
        f = plot_accuracy(df_perf)
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 1

    def test_suptitle_contains_accuracy(self, df_perf):
        """Happy: suptitle mentions 'Accuracy'"""
        f = plot_accuracy(df_perf)
        assert "Accuracy" in f._suptitle.get_text()


class TestPlotBinaryPerformance:
    """Tests for plot_binary_performance()"""

    def test_returns_figure_with_four_axes(self, df_binary_perf):
        """Happy: returns a Figure with 4 axes (ROC, PrecRec, F-measure, Accuracy)"""
        f = plot_binary_performance(df_binary_perf, nobs=100)
        assert isinstance(f, figure.Figure)
        assert len(f.axes) == 4

    def test_suptitle_contains_binary(self, df_binary_perf):
        """Happy: suptitle mentions 'Binary'"""
        f = plot_binary_performance(df_binary_perf, nobs=100)
        assert "Binary" in f._suptitle.get_text()

    def test_txtadd_in_suptitle(self, df_binary_perf):
        """Happy: txtadd kwarg appears in suptitle"""
        f = plot_binary_performance(df_binary_perf, nobs=100, txtadd="run1")
        assert "run1" in f._suptitle.get_text()
