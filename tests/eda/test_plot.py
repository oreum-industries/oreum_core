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

"""Tests for eda.plot

Strategy: lightweight smoke tests using the Agg non-interactive backend.
Focused on catching pandas/seaborn API breaking changes, not pixel values.
Each test verifies:
  + Function does not raise on valid inputs
  + Return type is matplotlib.figure.Figure (or None for empty feature lists)
  + Basic figure structure (n axes, suptitle present)
  + Guard clauses return None when no feature columns match
  + Invalid inputs raise the correct exception type
"""

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib import figure

from oreum_core.eda.plot import (
    plot_bool_ct,
    plot_cat_ct,
    plot_cdf_ppc_vs_obs,
    plot_date_ct,
    plot_explained_variance,
    plot_float_dist,
    plot_grp_ct,
    plot_heatmap_corr,
    plot_int_dist,
    plot_joint_numeric,
    set_plot_theme,
)

RNG = np.random.default_rng(seed=42)
N = 40


@pytest.fixture(scope="module")
def df_cat() -> pd.DataFrame:
    """DataFrame with object and category columns"""
    return pd.DataFrame(
        {
            "colour": pd.Categorical(
                RNG.choice(["red", "blue", "green"], N),
                categories=["red", "blue", "green"],
            ),
            "shape": RNG.choice(["circle", "square", "triangle"], N).astype(object),
        }
    )


@pytest.fixture(scope="module")
def df_bool() -> pd.DataFrame:
    """DataFrame with two bool columns"""
    return pd.DataFrame(
        {"flag_a": RNG.choice([True, False], N), "flag_b": RNG.choice([True, False], N)}
    )


@pytest.fixture(scope="module")
def df_date() -> pd.DataFrame:
    """DataFrame with a single datetime column"""
    dates = pd.date_range("2023-01-01", periods=N, freq="W")
    return pd.DataFrame({"event_date": RNG.choice(dates, N)})


@pytest.fixture(scope="module")
def df_int() -> pd.DataFrame:
    """DataFrame with two integer count columns"""
    return pd.DataFrame(
        {"count_a": RNG.integers(0, 100, N), "count_b": RNG.integers(0, 50, N)}
    )


@pytest.fixture(scope="module")
def df_float() -> pd.DataFrame:
    """DataFrame with two float columns"""
    return pd.DataFrame(
        {"x": RNG.standard_normal(N), "y": RNG.standard_normal(N) * 2 + 1}
    )


@pytest.fixture(scope="module")
def df_numeric() -> pd.DataFrame:
    """40×5 DataFrame of standard-normal values"""
    return pd.DataFrame(RNG.standard_normal((N, 5)), columns=list("abcde"))


@pytest.fixture(scope="module")
def df_corr(df_numeric) -> pd.DataFrame:
    """Correlation matrix of df_numeric"""
    return df_numeric.corr()


@pytest.fixture(scope="module")
def df_grp() -> pd.DataFrame:
    """DataFrame with a single categorical group column"""
    return pd.DataFrame({"grp": pd.Categorical(RNG.choice(["A", "B", "C"], N))})


class TestSetPlotTheme:
    """Tests for set_plot_theme()"""

    def test_default_args_do_not_raise(self):
        """Happy: sns.set_theme called with default args without error"""
        set_plot_theme()

    def test_custom_style(self):
        """Happy: sns.set_theme called with custom style without error"""
        set_plot_theme(style="whitegrid", palette="deep", context="paper")

    def test_seaborn_rc_applied(self):
        """Happy: custom rc dict is forwarded to seaborn/matplotlib"""
        set_plot_theme(rc={"figure.dpi": 72, "figure.figsize": (8, 4)})
        import matplotlib as mpl

        assert mpl.rcParams["figure.figsize"] == [8, 4]


class TestPlotCatCt:
    """Tests for plot_cat_ct()"""

    def test_returns_figure(self, df_cat):
        """Happy: returns a Figure for valid cat columns"""
        f = plot_cat_ct(df_cat, fts=["colour", "shape"])
        assert isinstance(f, figure.Figure)

    def test_no_matching_fts_returns_none(self, df_cat):
        """Edge: returns None when no fts match df columns"""
        f = plot_cat_ct(df_cat, fts=["nonexistent"])
        assert f is None

    def test_single_ft(self, df_cat):
        """Happy: works with a single feature"""
        f = plot_cat_ct(df_cat, fts=["colour"])
        assert isinstance(f, figure.Figure)

    def test_suptitle_present(self, df_cat):
        """Happy: figure has a suptitle containing expected text"""
        f = plot_cat_ct(df_cat, fts=["colour"])
        assert f._suptitle is not None
        assert "Empirical distribution" in f._suptitle.get_text()

    def test_topn_limits_bars(self, df_cat):
        """Happy: topn parameter accepted without error"""
        f = plot_cat_ct(df_cat, fts=["colour"], topn=2)
        assert isinstance(f, figure.Figure)

    def test_cat_order_false(self, df_cat):
        """Happy: cat_order=False accepted without error"""
        f = plot_cat_ct(df_cat, fts=["colour"], cat_order=False)
        assert isinstance(f, figure.Figure)

    def test_txtadd_kwarg(self, df_cat):
        """Happy: txtadd kwarg appears in suptitle"""
        f = plot_cat_ct(df_cat, fts=["colour"], txtadd="my note")
        assert "my note" in f._suptitle.get_text()

    def test_partial_fts_match(self, df_cat):
        """Edge: only matching columns are plotted; unknown names ignored"""
        f = plot_cat_ct(df_cat, fts=["colour", "does_not_exist"])
        assert isinstance(f, figure.Figure)


class TestPlotBoolCt:
    """Tests for plot_bool_ct()"""

    def test_returns_figure(self, df_bool):
        """Happy: returns a Figure for valid bool columns"""
        f = plot_bool_ct(df_bool, fts=["flag_a", "flag_b"])
        assert isinstance(f, figure.Figure)

    def test_no_matching_fts_returns_none(self, df_bool):
        """Edge: returns None when no fts match df columns"""
        f = plot_bool_ct(df_bool, fts=["nonexistent"])
        assert f is None

    def test_single_ft(self, df_bool):
        """Happy: works with a single boolean feature"""
        f = plot_bool_ct(df_bool, fts=["flag_a"])
        assert isinstance(f, figure.Figure)

    def test_suptitle_present(self, df_bool):
        """Happy: figure suptitle contains 'bools'"""
        f = plot_bool_ct(df_bool, fts=["flag_a"])
        assert "bools" in f._suptitle.get_text()

    def test_groupby_dropna_compatibility(self, df_bool):
        """Happy: groupby(dropna=False) works with pandas current version.
        Uses nullable boolean dtype with pd.NA to avoid FutureWarning.
        This will break if pandas removes or renames the dropna parameter.
        """
        df_with_none = df_bool.copy()
        df_with_none["flag_a"] = df_with_none["flag_a"].astype("boolean")
        df_with_none.loc[0, "flag_a"] = pd.NA
        f = plot_bool_ct(df_with_none, fts=["flag_a"])
        assert isinstance(f, figure.Figure)


class TestPlotDateCt:
    """Tests for plot_date_ct()"""

    def test_returns_figure_single_ft(self, df_date):
        """Happy: returns a Figure for a single datetime column"""
        f = plot_date_ct(df_date, fts=["event_date"])
        assert isinstance(f, figure.Figure)

    def test_no_matching_fts_returns_none(self, df_date):
        """Edge: returns None when no fts match df columns"""
        f = plot_date_ct(df_date, fts=["nonexistent"])
        assert f is None

    def test_custom_fmt(self, df_date):
        """Happy: custom strftime format accepted without error"""
        f = plot_date_ct(df_date, fts=["event_date"], fmt="%Y")
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_dates(self, df_date):
        """Happy: suptitle contains 'dates'"""
        f = plot_date_ct(df_date, fts=["event_date"])
        assert "dates" in f._suptitle.get_text()

    def test_dt_strftime_api(self, df_date):
        """Happy: dt.strftime accessor works on datetime series.
        Will break if pandas removes or renames the dt accessor.
        """
        result = df_date["event_date"].dt.strftime("%Y-%m")
        assert isinstance(result, pd.Series)


class TestPlotIntDist:
    """Tests for plot_int_dist()"""

    def test_returns_figure(self, df_int):
        """Happy: returns a Figure for valid int columns"""
        f = plot_int_dist(df_int, fts=["count_a", "count_b"])
        assert isinstance(f, figure.Figure)

    def test_no_matching_fts_returns_none(self, df_int):
        """Edge: returns None when no fts match df columns"""
        f = plot_int_dist(df_int, fts=["nonexistent"])
        assert f is None

    def test_single_ft(self, df_int):
        """Happy: works with a single int column"""
        f = plot_int_dist(df_int, fts=["count_a"])
        assert isinstance(f, figure.Figure)

    def test_log_scale(self, df_int):
        """Happy: log=True accepted without error"""
        f = plot_int_dist(df_int, fts=["count_a"], log=True)
        assert isinstance(f, figure.Figure)

    def test_ecdf_mode(self, df_int):
        """Happy: ecdf=True uses cumulative histplot without error.
        Tests seaborn histplot(stat='proportion', cumulative=True) API.
        """
        f = plot_int_dist(df_int, fts=["count_a"], ecdf=True)
        assert isinstance(f, figure.Figure)

    def test_plot_zeros_false(self, df_int):
        """Happy: plot_zeros=False accepted without error"""
        f = plot_int_dist(df_int, fts=["count_a"], plot_zeros=False)
        assert isinstance(f, figure.Figure)

    def test_n_axes_matches_n_fts(self, df_int):
        """Happy: one subplot row per feature"""
        f = plot_int_dist(df_int, fts=["count_a", "count_b"])
        assert len(f.axes) == 2

    def test_suptitle_contains_ints(self, df_int):
        """Happy: suptitle contains 'ints'"""
        f = plot_int_dist(df_int, fts=["count_a"])
        assert "ints" in f._suptitle.get_text()


class TestPlotFloatDist:
    """Tests for plot_float_dist()"""

    def test_returns_figure(self, df_float):
        """Happy: returns a Figure for valid float columns.
        This exercises FacetGrid, violinplot(density_norm=), and
        pointplot(errorbar=) — all seaborn 0.13+ API.
        """
        f = plot_float_dist(df_float, fts=["x", "y"])
        assert isinstance(f, figure.Figure)

    def test_no_matching_fts_returns_none(self, df_float):
        """Edge: returns None when no fts match df columns"""
        f = plot_float_dist(df_float, fts=["nonexistent"])
        assert f is None

    def test_single_ft(self, df_float):
        """Happy: works with a single float column"""
        f = plot_float_dist(df_float, fts=["x"])
        assert isinstance(f, figure.Figure)

    def test_log_scale(self, df_float):
        """Happy: log=True accepted without error"""
        df_pos = pd.DataFrame({"x": np.abs(RNG.standard_normal(N)) + 0.1})
        f = plot_float_dist(df_pos, fts=["x"], log=True)
        assert isinstance(f, figure.Figure)

    def test_infs_filtered(self, df_float):
        """Happy: infinite values are dropped before plotting without error"""
        df_with_inf = df_float.copy()
        df_with_inf.loc[0, "x"] = np.inf
        f = plot_float_dist(df_with_inf, fts=["x"])
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_floats(self, df_float):
        """Happy: suptitle contains 'floats'"""
        f = plot_float_dist(df_float, fts=["x"])
        assert "floats" in f._suptitle.get_text()

    def test_sort_false(self, df_float):
        """Happy: sort=False accepted without error"""
        f = plot_float_dist(df_float, fts=["y", "x"], sort=False)
        assert isinstance(f, figure.Figure)

    def test_facetgrid_row_per_feature(self, df_float):
        """Happy: FacetGrid creates one row per feature.
        Tests that FacetGrid.map() API is compatible with current seaborn.
        """
        f = plot_float_dist(df_float, fts=["x", "y"])
        assert len(f.axes) == 2


class TestPlotJointNumeric:
    """Tests for plot_joint_numeric()"""

    def test_returns_figure_kde(self, df_float):
        """Happy: kind='kde' returns a Figure"""
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", kind="kde")
        assert isinstance(f, figure.Figure)

    def test_returns_figure_scatter(self, df_float):
        """Happy: kind='scatter' returns a Figure"""
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", kind="scatter")
        assert isinstance(f, figure.Figure)

    def test_returns_figure_kde_scatter(self, df_float):
        """Happy: kind='kde+scatter' returns a Figure"""
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", kind="kde+scatter")
        assert isinstance(f, figure.Figure)

    def test_returns_figure_reg(self, df_float):
        """Happy: kind='reg' returns a Figure"""
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", kind="reg")
        assert isinstance(f, figure.Figure)

    def test_invalid_kind_raises(self, df_float):
        """Sad: unknown kind raises ValueError"""
        with pytest.raises(ValueError, match="kind"):
            plot_joint_numeric(df_float, ft0="x", ft1="y", kind="hexbin")

    def test_with_hue_categorical(self, df_float):
        """Happy: hue column of object dtype accepted without error"""
        df = df_float.copy()
        df["grp"] = np.where(df["x"] > 0, "pos", "neg")
        f = plot_joint_numeric(df, ft0="x", ft1="y", hue="grp", kind="scatter")
        assert isinstance(f, figure.Figure)

    def test_linreg_annotation(self, df_float):
        """Happy: linreg=True adds text annotation to joint axis.
        Tests scipy.stats.linregress API compatibility.
        """
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", linreg=True, kind="scatter")
        assert isinstance(f, figure.Figure)

    def test_nsamp_subsample(self, df_float):
        """Happy: nsamp subsamples data without error"""
        f = plot_joint_numeric(df_float, ft0="x", ft1="y", nsamp=20)
        assert isinstance(f, figure.Figure)

    def test_log_x_scale(self, df_float):
        """Happy: log='x' sets x log scale without error"""
        df_pos = df_float.copy()
        df_pos["x"] = np.abs(df_pos["x"]) + 0.1
        f = plot_joint_numeric(df_pos, ft0="x", ft1="y", log="x", kind="scatter")
        assert isinstance(f, figure.Figure)


class TestPlotHeatmapCorr:
    """Tests for plot_heatmap_corr()"""

    def test_returns_figure(self, df_corr):
        """Happy: returns a Figure for a correlation matrix"""
        f = plot_heatmap_corr(df_corr)
        assert isinstance(f, figure.Figure)

    def test_single_axis(self, df_corr):
        """Happy: figure has exactly one axes"""
        f = plot_heatmap_corr(df_corr)
        assert len(f.axes) == 2  # main axes + colorbar

    def test_suptitle_contains_correlations(self, df_corr):
        """Happy: suptitle contains 'Feature correlations'"""
        f = plot_heatmap_corr(df_corr)
        assert "Feature correlations" in f._suptitle.get_text()

    def test_txtadd_in_suptitle(self, df_corr):
        """Happy: txtadd kwarg appears in suptitle"""
        f = plot_heatmap_corr(df_corr, txtadd="test run")
        assert "test run" in f._suptitle.get_text()

    def test_triu_mask_api(self, df_corr):
        """Happy: np.triu(np.ones_like(...), k=0) mask is compatible.
        Tests numpy API used inside the function.
        """
        mask = np.triu(np.ones_like(df_corr), k=0)
        assert mask.shape == df_corr.shape
        assert mask.dtype == np.float64


class TestPlotGrpCt:
    """Tests for plot_grp_ct()"""

    def test_returns_figure_categorical(self, df_grp):
        """Happy: returns a Figure for a categorical grp column"""
        f = plot_grp_ct(df_grp, grp="grp")
        assert isinstance(f, figure.Figure)

    def test_returns_figure_object(self):
        """Happy: returns a Figure for an object (string) grp column"""
        df = pd.DataFrame({"grp": RNG.choice(["X", "Y", "Z"], N).astype(object)})
        f = plot_grp_ct(df, grp="grp")
        assert isinstance(f, figure.Figure)

    def test_numeric_grp_raises_typeerror(self):
        """Sad: numeric grp column raises TypeError"""
        df = pd.DataFrame({"grp": RNG.integers(0, 5, N)})
        with pytest.raises(TypeError):
            plot_grp_ct(df, grp="grp")

    def test_orderby_ordinal(self, df_grp):
        """Happy: orderby='ordinal' accepted without error"""
        f = plot_grp_ct(df_grp, grp="grp", orderby="ordinal")
        assert isinstance(f, figure.Figure)

    def test_orderby_none(self, df_grp):
        """Happy: orderby=None accepted without error"""
        f = plot_grp_ct(df_grp, grp="grp", orderby=None)
        assert isinstance(f, figure.Figure)

    def test_topn_limits_groups(self, df_grp):
        """Happy: topn=2 restricts to top 2 groups without error"""
        f = plot_grp_ct(df_grp, grp="grp", topn=2)
        assert isinstance(f, figure.Figure)

    def test_countplot_api(self, df_grp):
        """Happy: sns.countplot with hue=, legend=False works.
        hue= required in seaborn 0.13+ to avoid deprecation; legend=False
        added in 0.13. Will break if seaborn removes these parameters.
        """
        f = plot_grp_ct(df_grp, grp="grp")
        assert isinstance(f, figure.Figure)

    def test_suptitle_contains_countplot(self, df_grp):
        """Happy: suptitle contains 'Countplot'"""
        f = plot_grp_ct(df_grp, grp="grp")
        assert "Countplot" in f._suptitle.get_text()


class TestPlotExplainedVariance:
    """Tests for plot_explained_variance()"""

    def test_returns_figure(self, df_numeric):
        """Happy: returns a Figure for a numeric DataFrame"""
        f = plot_explained_variance(df_numeric, k=3, topn=2)
        assert isinstance(f, figure.Figure)

    def test_single_axis(self, df_numeric):
        """Happy: figure has exactly one axes"""
        f = plot_explained_variance(df_numeric, k=3, topn=2)
        assert len(f.axes) == 1

    def test_suptitle_contains_explained_variance(self, df_numeric):
        """Happy: suptitle mentions 'Explained variance'"""
        f = plot_explained_variance(df_numeric, k=3, topn=2)
        assert "Explained variance" in f._suptitle.get_text()

    def test_pointplot_api(self, df_numeric):
        """Happy: sns.pointplot with x/y column names works.
        Tests seaborn pointplot API — x= and y= kwargs changed in 0.12.
        """
        f = plot_explained_variance(df_numeric, k=4, topn=3)
        assert isinstance(f, figure.Figure)


class TestPlotCdfPpcVsObs:
    """Tests for plot_cdf_ppc_vs_obs()"""

    def test_returns_figure(self):
        """Happy: returns a Figure for valid y and yhat arrays"""
        y = RNG.standard_normal(N)
        yhat = RNG.standard_normal((N, 50))
        f = plot_cdf_ppc_vs_obs(y, yhat)
        assert isinstance(f, figure.Figure)

    def test_intercept_only_model(self):
        """Edge: all-duplicate yhat rows (intercept-only model) uses rugplot path"""
        y = RNG.standard_normal(N)
        yhat = np.ones((N, 50)) * 2.5  # constant predictions
        f = plot_cdf_ppc_vs_obs(y, yhat)
        assert isinstance(f, figure.Figure)
