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

# eda.plot.py
"""EDA Plotting"""

import logging
from copy import copy
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure, gridspec, lines, ticker
from scipy import stats

from .calc import calc_svd
from .describe import get_fts_by_dtype

__all__ = [
    "set_plot_theme",
    "plot_cat_ct",
    "plot_bool_ct",
    "plot_date_ct",
    "plot_int_dist",
    "plot_float_dist",
    "plot_joint_numeric",
    "plot_mincovdet",
    "plot_bootstrap_lr",
    "plot_bootstrap_lr_grp",
    "plot_bootstrap_grp",
    "plot_bootstrap_delta_grp",
    "plot_smrystat",
    "plot_smrystat_grp",
    "plot_smrystat_grp_year",
    "plot_heatmap_corr",
    "plot_kj_summaries_for_linear_model",
    "plot_grp_ct",
    "plot_cdf_ppc_vs_obs",
    "plot_explained_variance",
]


_log = logging.getLogger(__name__)
RSD = 42
rng = np.random.default_rng(seed=RSD)


def set_plot_theme(
    style: str = "darkgrid",
    palette: str = "muted",
    context: str = "notebook",
    rc: dict = None,
):
    """Apply the oreum seaborn/matplotlib plot theme globally.
    Call once at the top of a notebook or script before plotting.
    Defaults match the oreum house style.
    """
    if rc is None:
        rc = {"figure.dpi": 72, "savefig.dpi": 144, "figure.figsize": (12, 4)}
    sns.set_theme(style=style, palette=palette, context=context, rc=rc)


def _get_kws_styling() -> dict:
    """Common styling kws for plots"""
    kws = dict(
        count_txt_kws=dict(
            color="#555555",
            fontsize=8,
            va="center",
            xycoords="data",
            textcoords="offset points",
        ),
        mn_pt_kws=dict(
            markerfacecolor="w", markeredgecolor="#333333", marker="d", markersize=12
        ),
        pest_mn_pt_kws=dict(
            markerfacecolor="#11c8bc",
            markeredgecolor="#eeeeee",
            marker="d",
            markersize=10,
        ),
        mn_txt_kws=dict(
            color="#555555",
            xycoords="data",
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.1,rounding_size=0.1", fc="w", ec="none"),
        ),
        pest_mn_txt_kws=dict(
            color="#555555",
            xycoords="data",
            xytext=(-6, -12),
            textcoords="offset points",
            fontsize=7,
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.1,rounding_size=0.1", fc="#c8fdf9", ec="none"
            ),
        ),
    )
    kws["count_txt_h_kws"] = dict(ha="left", xytext=(4, 0), **kws["count_txt_kws"])
    return kws


def plot_cat_ct(
    df: pd.DataFrame,
    fts: list,
    topn: int = 10,
    vsize: float = 1.8,
    cat_order: bool = True,
    m: int = 2,
    **kwargs,
) -> figure.Figure:
    """Conv fn: plot group counts for cats"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts) / m))
    f, ax2d = plt.subplots(vert, m, squeeze=False, figsize=(12, 0.5 + vert * vsize))

    for i, ft in enumerate(fts):
        counts_all = df.groupby(ft, observed=False).size().sort_values(ascending=True)
        if (df[ft].dtype == "category") and cat_order:
            counts_all = df.groupby(ft, observed=False).size()[::-1]  # need to invert

        if df[ft].dtype == bool:
            counts_all = counts_all.sort_index()  # sort so true plots on top

        counts = counts_all.iloc[-topn:]
        ax = counts.plot(
            kind="barh",
            ax=ax2d[i // m, i % m],
            title="{}: {} factor levels".format(ft, len(counts_all)),
            label="{} NaNs".format(pd.isnull(df[ft]).sum()),
        )
        _ = [
            ax.annotate(
                "{} ({:.0%})".format(c, c / counts_all.sum()),
                xy=(c, i),
                xycoords="data",
                xytext=(4, -2),
                textcoords="offset points",
                ha="left",
                fontsize=10,
                color="#666666",
            )
            for i, c in enumerate(counts)
        ]

        if df[ft].dtype != bool:
            ax.legend(loc="lower right")
        else:
            _ = ax.set(ylabel=None)

        _ = ax.set_yticklabels([lbl.get_text()[:30] for lbl in ax.get_yticklabels()])

    t = "Empirical distribution"
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle(" - ".join(filter(None, [t, "cats", txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_bool_ct(
    df: pd.DataFrame, fts: list, vsize: float = 1.4, **kwargs
) -> figure.Figure:
    """Conv fn: plot group counts for bools"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts) / 2))
    f, ax2d = plt.subplots(vert, 2, squeeze=False, figsize=(12, 0.5 + vert * vsize))

    for i, ft in enumerate(fts):
        counts = df.groupby(ft, dropna=False).size().sort_values(ascending=True)
        counts = counts.sort_index()  # sort so true plots on top
        ax = counts.plot(
            kind="barh",
            ax=ax2d[i // 2, i % 2],
            title="{}: {} boolean levels".format(ft, len(counts)),
        )
        _ = [
            ax.annotate(
                "{} ({:.0%})".format(c, c / counts.sum()),
                xy=(c, i),
                xycoords="data",
                xytext=(4, -2),
                textcoords="offset points",
                ha="left",
                fontsize=10,
                color="#666666",
            )
            for i, c in enumerate(counts)
        ]
        _ = ax.set(ylabel=None)
        _ = ax.set_yticklabels([lbl.get_text()[:30] for lbl in ax.get_yticklabels()])

    t = "Empirical distribution"
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle(" - ".join(filter(None, [t, "bools", txtadd])), y=1, fontsize=14)
    f.tight_layout(pad=0.9)
    return f


def plot_date_ct(
    df: pd.DataFrame, fts: list, fmt: str = "%Y-%m", vsize: float = 1.6, **kwargs
) -> figure.Figure:
    """Plot group sizes for dates by strftime format"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = len(fts)
    f, ax1d = plt.subplots(vert, 1, figsize=(12, 0.5 + vert * vsize), squeeze=True)

    if vert > 1:
        for i, ft in enumerate(fts):
            ax = (
                df[ft]
                .groupby(df[ft].dt.strftime(fmt))
                .size()
                .plot(
                    kind="bar",
                    ax=ax1d[i],
                    title=ft,
                    label="{} NaNs".format(pd.isnull(df[ft]).sum()),
                )
            )
            ax.legend(loc="upper right")
    else:
        ft = fts[0]
        ax = (
            df[ft]
            .groupby(df[ft].dt.strftime(fmt))
            .size()
            .plot(kind="bar", title=ft, label="{} NaNs".format(pd.isnull(df[ft]).sum()))
        )
        ax.legend(loc="upper right")
    t = "Empirical distribution"
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle(" - ".join(filter(None, [t, "dates", txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_int_dist(
    df: pd.DataFrame,
    fts: list,
    log: bool = False,
    vsize: float = 1.4,
    bins: int = None,
    plot_zeros: bool = True,
    ecdf: bool = False,
    **kwargs,
) -> figure.Figure:
    """Plot group counts as histogram (optional log)"""
    kws_hist = dict(stat="count")
    legpos = "upper right"
    t = "Empirical distribution"
    if ecdf:
        kws_hist = dict(stat="proportion", cumulative=True)
        legpos = "lower right"
        t += " ECDF"

    # handles under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None
    if bins is None:
        bins = "auto"

    vert = len(fts)
    f, ax1d = plt.subplots(len(fts), 1, figsize=(12, 0.5 + vert * vsize), squeeze=False)
    for i, ft in enumerate(fts):
        n_nans = pd.isnull(df[ft]).sum()
        mean = df[ft].mean()
        med = df[ft].median()
        n_zeros = (df[ft] == 0).sum()
        if not plot_zeros:
            df = df.loc[df[ft] != 0].copy()
        ax = sns.histplot(
            df.loc[df[ft].notnull(), ft],
            bins=bins,
            label=f"NaNs: {n_nans}, zeros: {n_zeros}, mean: {mean:.2f}, med: {med:.2f}",
            color=sns.color_palette()[i % 7],
            ax=ax1d[i][0],
            **kws_hist,
        )
        if log:
            _ = ax.set(yscale="log", title=ft, ylabel=f"log({kws_hist['stat']})")
        _ = ax.set(title=ft, ylabel=kws_hist["stat"], xlabel=None)
        _ = ax.legend(loc=legpos)
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle(" - ".join(filter(None, [t, "ints", txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_float_dist(
    df: pd.DataFrame,
    fts: list,
    log: bool = False,
    sharex: bool = False,
    sort: bool = True,
    **kwargs,
) -> figure.Figure:
    """
    Plot distributions for floats
    Annotate with count of nans, infs (+/-) and zeros
    """
    s = None
    t = "Empirical distribution"

    def _annotate_facets(data, **kwargs):
        """Func to be mapped to the dataframe (named `data` by seaborn)
        used per facet. Assume `data` is the simple result of a melt()
        and has two fts: variable, value
        """
        n_nans = pd.isnull(data["value"]).sum()
        n_zeros = (data["value"] == 0).sum()
        n_infs = kwargs.pop("n_infs", 0)
        mean = data["value"].mean()
        med = data["value"].median()
        ax = plt.gca()
        ax.text(
            0.993,
            0.93,
            (
                f"NaNs: {n_nans},  infs+/-: {n_infs},  zeros: {n_zeros},  "
                + f"mean: {mean:.2f},  med: {med:.2f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            backgroundcolor="w",
            fontsize=10,
        )

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    # hacky protect against massive datasets with a subsample
    ldf = len(df)
    if ldf > 1e6:
        df = df.sample(n=1000000, random_state=42)
        s = f"subsample 1e6 of {ldf:.0g} total obs ({1e6 / ldf:.1%})"

    if sort:
        dfm = df[sorted(fts)].melt(var_name="variable")
    else:
        dfm = df[fts].melt(var_name="variable")

    idx_inf = np.isinf(dfm["value"])
    dfm = dfm.loc[~idx_inf].copy()

    gd = sns.FacetGrid(
        data=dfm,
        row="variable",
        hue="variable",
        palette=sns.color_palette(),
        height=1.8,
        aspect=6,
        sharex=sharex,
    )
    _ = gd.map(sns.violinplot, "value", order=None, cut=0, density_norm="count")
    _ = gd.map(
        sns.pointplot,
        "value",
        order=None,
        color="C3",
        estimator=np.mean,
        errorbar=("ci", 94),
    )

    # https://stackoverflow.com/q/33486613/1165112
    # scatter_kws=(dict(edgecolor='k', edgewidth=100)))
    _ = gd.map_dataframe(_annotate_facets, n_infs=sum(idx_inf))

    if log:
        _ = gd.set(xscale="log")  # , title=ft, ylabel='log(count)')

    txtadd = kwargs.pop("txtadd", None)
    _ = gd.fig.suptitle(
        " - ".join(filter(None, [t, "floats", s, txtadd])), y=1, fontsize=14
    )
    _ = gd.fig.tight_layout(pad=0.9)
    return gd.fig


def plot_joint_numeric(
    data: pd.DataFrame,
    ft0: str,
    ft1: str,
    hue: str = None,
    kind: str = "kde",
    height: int = 6,
    kdefill: bool = True,
    log: Literal["x", "y", "both"] = None,
    colori: int = 0,
    nsamp: int = None,
    linreg: bool = True,
    legendpos: str = None,
    palette_type: Literal["q", "g"] = "g",
    palette: str = None,
    eq: int = 7,  # equal quantiles. Set higher in the case of extreme values
    overplot: dict = None,
    **kwargs,
) -> figure.Figure:
    """Jointplot of 2 numeric fts with optional: hue shading, linear regression
    Suitable for int or float
    NOTE this is not NaN friendly, and will fill nans with zeros"""

    dfp = data.copy()
    kws = dict(color=f"C{colori % 7}")  # color rotation max 7

    if nsamp is not None:
        dfp = dfp.sample(nsamp, random_state=RSD).copy()

    nobs = len(dfp)
    for ft in [ft0, ft1]:
        dfp[ft] = dfp[ft].fillna(0)

    if hue is not None:
        ngrps = len(dfp[hue].unique())
        nobs = len(dfp) // ngrps
        if palette is None:
            ftsd = get_fts_by_dtype(dfp)
            linreg = False
            if hue in ftsd["int"] + ftsd["float"]:  # bin into n equal quantiles
                dfp[hue] = pd.qcut(dfp[hue].values, q=eq, duplicates="drop")
                kws["palette"] = "viridis"
                nobs = len(dfp)
            else:
                if palette_type == "g":
                    kws["palette"] = sns.color_palette(
                        [f"C{i + colori % 7}" for i in range(ngrps)]
                    )
                else:  # palette_type == 'q':
                    kws["palette"] = sns.color_palette(
                        palette="RdYlBu_r", n_colors=ngrps
                    )
        elif isinstance(palette, str):
            kws["palette"] = sns.color_palette(palette=palette, n_colors=ngrps)
        else:  # pass a palette directly
            kws["palette"] = palette

    gd = sns.JointGrid(x=ft0, y=ft1, data=dfp, height=height, hue=hue)

    kde_kws = kws | dict(
        zorder=0, levels=7, cut=0, fill=kdefill, legend=True, warn_singular=False
    )
    scatter_kws = kws | dict(
        alpha=0.6, marker="o", linewidths=0.05, edgecolor="#dddddd", s=50
    )
    reg_kws = kws | dict(scatter_kws=scatter_kws)
    rug_kws = kws | dict(height=0.1, legend=False)

    if kind == "kde":
        _ = gd.plot_joint(sns.kdeplot, **kde_kws)
    elif kind == "scatter":
        _ = gd.plot_joint(sns.scatterplot, **scatter_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    elif kind == "kde+scatter":
        _ = gd.plot_joint(sns.kdeplot, **kde_kws)
        _ = gd.plot_joint(sns.scatterplot, **scatter_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    elif kind == "reg":
        _ = gd.plot_joint(sns.regplot, **reg_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    else:
        raise ValueError("kwarg `kind` must be in {kde, scatter, kde+scatter, reg}")

    if legendpos is not None:
        _ = sns.move_legend(gd.ax_joint, legendpos)

    _ = gd.plot_marginals(sns.histplot, kde=True, **kws)

    if overplot is not None:
        _ = sns.scatterplot(
            x=overplot["ft0"], y=overplot["ft1"], data=overplot["data"], ax=gd.ax_joint
        )

    if linreg:
        r = stats.linregress(x=dfp[ft0], y=dfp[ft1])
        _ = gd.ax_joint.text(
            0.98,
            0.98,
            f"y = {r.slope:.2f}x + {r.intercept:.2f}\nρ = {r.rvalue:.2f}",
            transform=gd.ax_joint.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    if log in ["x", "both"]:
        _ = gd.ax_joint.set_xscale("log")
        _ = gd.ax_marg_x.set_xscale("log")

    if log in ["y", "both"]:
        _ = gd.ax_joint.set_yscale("log")
        _ = gd.ax_marg_y.set_yscale("log")

    t = f"Joint & marginal dists: `{ft0}` vs `{ft1}`, {nobs} obs"
    txtadd = kwargs.pop("txtadd", None)
    _ = gd.fig.suptitle("\n".join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = gd.fig.tight_layout(pad=0.95)
    return gd.fig


def plot_mincovdet(df: pd.DataFrame, mcd, thresh: float = 0.99):
    """Interactive plot of MCD delta results"""

    dfp = df.copy()
    dfp["mcd_delta"] = mcd.dist_
    dfp = dfp.sort_values("mcd_delta")
    dfp["counter"] = np.arange(dfp.shape[0])

    cutoff = np.percentile(dfp["mcd_delta"], thresh * 100)
    dfp["mcd_outlier"] = dfp["mcd_delta"] > cutoff

    f = plt.figure(figsize=(12, 8))
    f.suptitle(
        "Distribution of outliers"
        + "\n(thresh @ {:.1%}, cutoff @ {:.1f}, identified {} outliers)".format(
            thresh, cutoff, dfp["mcd_outlier"].sum()
        ),
        fontsize=14,
    )

    grd = plt.GridSpec(nrows=1, ncols=2, wspace=0.05, width_ratios=[3, 1])

    # sorted MCD dist plot
    ax0 = plt.subplot(grd[0])
    _ = ax0.scatter(
        dfp["counter"],
        dfp["mcd_delta"],
        c=dfp["counter"] / 1,
        cmap="YlOrRd",
        alpha=0.8,
        marker="o",
        linewidths=0.05,
        edgecolor="#999999",
    )

    _ = ax0.axhline(y=cutoff, xmin=0, xmax=1, linestyle="--", color="#DD0011")
    _ = ax0.annotate(
        "Thresh @ {:.1%}, cutoff @ {:.1f}".format(thresh, cutoff),
        xy=(0, cutoff),
        xytext=(10, 10),
        textcoords="offset points",
        color="#DD0011",
        style="italic",
        weight="bold",
        size="large",
    )
    _ = ax0.set_xlim((-10, dfp.shape[0] + 10))
    _ = ax0.set_yscale("log")
    _ = ax0.set_ylabel("MCD delta (log)")
    _ = ax0.set_xlabel("Datapoints sorted by increasing MCD delta")

    # summary boxplot
    ax1 = plt.subplot(grd[1], sharey=ax0)
    bx = ax1.boxplot(
        dfp["mcd_delta"],
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markersize": 10,
            "markeredgecolor": "k",
            "markerfacecolor": "w",
            "markeredgewidth": 1,
        },
    )

    _ = ax1.axhline(y=cutoff, xmin=0, xmax=1, linestyle="--", color="#DD0011")
    _ = ax1.set_xlabel("Log MCD distance")
    #     _ = ax1.set_yticklabels([lbl.set_visible(False) for lbl in ax1.get_yticklabels()])
    _ = plt.setp(ax1.get_yticklabels(), visible=False)
    _ = plt.setp(bx["medians"], color="blue")

    return None


def plot_bootstrap_lr(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    prm: str = "premium",
    clm: str = "claim",
    clm_ct: str = None,
    obs_unit: str = "policies",
    ftname_year: str = "incept_year",
    pol_summary: bool = True,
    lr_summary: bool = True,
    force_xlim: list = None,
    color: str = None,
    pretty_plot: bool = False,
    **kwargs,
) -> figure.Figure:
    """Plot bootstrapped loss ratio, no grouping
    `pretty_plot`=True returns a low rezzed KDE suitable for presentations
    """

    sty = _get_kws_styling()
    mn = dfboot["lr"].mean()
    hdi = dfboot["lr"].quantile(q=[0.03, 0.25, 0.75, 0.97]).values

    clr = color if color is not None else sns.color_palette()[0]

    if not pretty_plot:  # default violin plots, more technical
        gd = sns.catplot(
            x="lr", data=dfboot, kind="violin", cut=0, color=clr, height=3, aspect=4
        )
        pest_mn = (
            np.nan_to_num(df[clm].to_numpy(copy=True), 0).sum() / df[prm].sum()
        )  # point est mean
        _ = gd.ax.plot(pest_mn, 0, **sty["pest_mn_pt_kws"])
        _ = gd.ax.annotate(f"{pest_mn:.1%}", xy=(pest_mn, 0), **sty["pest_mn_txt_kws"])
        elems = [
            lines.Line2D(
                [0], [0], label="population LR (bootstrap $\\mu$)", **sty["mn_pt_kws"]
            ),
            lines.Line2D(
                [0], [0], label="sample LR (point est. $\\mu$)", **sty["pest_mn_pt_kws"]
            ),
        ]
        gd.ax.legend(handles=elems, loc="upper right", fontsize=8)

    else:  # optional prettier plot
        gd = sns.displot(
            x="lr",
            data=dfboot,
            kind="kde",
            color=clr,
            height=4,
            aspect=3,
            fill=True,
            lw=0,
            alpha=0.7,
        )
        xlims = dfboot["lr"].quantile(q=[0.00, 0.99]).values  # cut the upper tail
        _ = gd.ax.set(xlim=xlims, ylim=(-0.05, None))
        _ = gd.ax.yaxis.grid(False)

    _ = gd.ax.plot(mn, 0, **sty["mn_pt_kws"])
    _ = gd.ax.annotate(f"{mn:.1%}", xy=(mn, 0), **sty["mn_txt_kws"])
    if force_xlim is not None:
        _ = gd.ax.set(xlim=force_xlim)

    _ = gd.ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    # hacky way to deal with year as int or datetime
    pmin = df[ftname_year].min()
    pmax = df[ftname_year].max()
    if np.issubdtype(df[ftname_year].dtype, np.datetime64):
        pmin = pmin.year
        pmax = pmax.year

    summary = ""
    if clm_ct is None:
        n_clm = ""
    else:
        n_clm = f"{df[clm_ct].sum():,.0f}"

    # get nicer units. There's almost certainly a better way of doing this
    exp = int(np.floor(np.log10(df[prm].sum())))
    a = np.arange(0, 12 + 1, 3)
    u = ["", "k", "M", "B", "T"]
    a_idx = np.argwhere(a < exp)[-1][0]
    unit_v = a[a_idx]
    unit_nm = u[a_idx]

    if pol_summary:
        summary += (
            f"Inception {str(pmin)} - {str(pmax)} inclusive, "
            + f"{len(df):,.0f} {obs_unit} with "
            + f"\\${df[prm].sum() / 10**unit_v:.1f}{unit_nm} premium, "
            + f"{n_clm}claims totalling "
            + f"\\${df[clm].sum() / 10**unit_v:.1f}{unit_nm}"
        )
    if lr_summary:
        summary += (
            f"\nPopulation LR: $\\mu$ = {mn:.1%}, "
            + f"$HDI_{{50}}$ = [{hdi[1]:.1%}, {hdi[2]:.1%}], "
            + f"$HDI_{{94}}$ = [{hdi[0]:.1%}, {hdi[3]:.1%}]"
        )
    txtadd = kwargs.pop("txtadd", None)
    t = "Population Loss Ratio"
    t = " - ".join(filter(None, [t, txtadd]))
    _ = gd.fig.suptitle("\n".join(filter(None, [t, summary])), y=1, fontsize=14)
    _ = gd.fig.tight_layout()
    return gd.fig


def plot_bootstrap_lr_grp(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    grp: str = "grp",
    prm: str = "premium",
    clm: str = "claim",
    clm_ct: str = None,
    obs_unit: str = "policies",
    ftname_year: str = "incept_year",
    pol_summary: bool = True,
    force_xlim: list = None,
    plot_grid: bool = True,
    annot_pest: bool = False,
    pal: str = "viridis",
    orderby: Literal["ordinal", "count", "lr"] = "ordinal",
    topn: int = None,
    **kwargs,
) -> figure.Figure:
    """Plot bootstrapped loss ratio, grouped by grp"""

    dfboot = dfboot.copy()
    df = df.copy()
    sty = _get_kws_styling()
    t = f"Bootstrapped Population Loss Ratio, grouped by {grp}"

    # hacky way to deal with year as int or datetime
    pmin = df[ftname_year].min()
    pmax = df[ftname_year].max()
    if np.issubdtype(df[ftname_year].dtype, np.datetime64):
        pmin = pmin.year
        pmax = pmax.year

    # hacky way to convert year as datetime to int
    # if np.issubdtype(df[grp].dtype, np.datetime64): error for categoricals
    if dfboot[grp].dtype == "<M8[ns]":
        dfboot[grp] = dfboot[grp].dt.year.astype(int)
        df[grp] = df[grp].dt.year.astype(int)

    # convert non object type to string
    if dfboot[grp].dtype not in ["object", "category"]:
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f"s{x}")
        df = df.copy()
        df[grp] = df[grp].map(lambda x: f"s{x}")

    ct = df.groupby(grp, observed=True).size()
    mn = dfboot.groupby(grp, observed=True)["lr"].mean()
    pest_mn = df.groupby(grp, observed=True).apply(
        lambda g: np.nan_to_num(g[clm].to_numpy(copy=True), 0).sum() / g[prm].sum(),
        include_groups=False,
    )

    # create order items / index
    if orderby == "count":
        ct = ct.sort_values()[::-1]
    elif orderby == "ordinal":
        pass  # ct == ct already
    elif orderby == "lr":
        ct = ct.reindex(mn.sort_values()[::-1].index)
    else:
        pass  # accept the default ordering as passed into func

    if topn is not None:
        ct = ct[:topn].copy()
        dfboot = dfboot.loc[dfboot[grp].isin(ct.index.values)].copy()
        t += f" (top {len(ct)} levels)"

    # reorder accordingly
    mn = mn.reindex(ct.index).values
    pest_mn = pest_mn.reindex(ct.index).values

    f = plt.figure(figsize=(16, 2 + (len(ct) * 0.3)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    ax0.set_title("Distribution of bootstrapped LR")
    ax1.set_title(f"Count ({len(ct)} lvls)")

    # common kws
    kws = dict(
        y=grp,
        order=ct.index.values,
        palette=pal,
        hue=grp,
        hue_order=ct.index.values,
        legend=False,
    )
    # add violinplot
    kws_vio = {**kws, **dict(cut=0, density_norm="count", width=0.6)}
    _ = sns.violinplot(**kws_vio, x="lr", data=dfboot, ax=ax0)

    _ = [ax0.plot(v, i % len(mn), **sty["mn_pt_kws"]) for i, v in enumerate(mn)]
    _ = [
        ax0.annotate(f"{v:.1%}", xy=(v, i % len(mn)), **sty["mn_txt_kws"])
        for i, v in enumerate(mn)
    ]
    _ = [
        ax0.plot(v, i % len(pest_mn), **sty["pest_mn_pt_kws"])
        for i, v in enumerate(pest_mn)
    ]
    if annot_pest:
        _ = [
            ax0.annotate(f"{v:.1%}", xy=(v, i % len(pest_mn)), **sty["pest_mn_txt_kws"])
            for i, v in enumerate(pest_mn)
        ]
    ax0.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    elems = [
        lines.Line2D([0], [0], label="population LR (bootstrap)", **sty["mn_pt_kws"]),
        lines.Line2D([0], [0], label="sample LR", **sty["pest_mn_pt_kws"]),
    ]
    _ = ax0.legend(handles=elems, loc="upper right", fontsize=8)
    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    # add countplot
    _ = sns.countplot(**kws, data=df, ax=ax1)
    _ = [
        ax1.annotate(f"{v}", xy=(v, i % len(ct)), **sty["count_txt_h_kws"])
        for i, v in enumerate(ct)
    ]
    # make easier to read labels for the count axis..
    # idea 1: shorten labels, doesnt work because shared with ax0
    # _ = ax1.set_yticks(ticks=ax1.get_yticks(), labels=[f"{lbl.get_text()[:8]}.." for lbl in ax1.get_yticklabels()])
    # idea 2 just make invisible, and add grid lines to aid the eye
    _ = ax1.get_yaxis().set_visible(False)
    if plot_grid:
        ax0.yaxis.grid(True)
        ax1.yaxis.grid(True)

    summary = ""
    if clm_ct is None:
        n_clm = ""
    else:
        n_clm = f"{df[clm_ct].sum():,.0f}"

    # get nicer units. There's almost certainly a better way of doing this
    exp = int(np.floor(np.log10(df[prm].sum())))
    a = np.arange(0, 12 + 1, 3)
    u = ["", "k", "M", "B", "T"]
    a_idx = np.argwhere(a < exp)[-1][0]
    unit_v = a[a_idx]
    unit_nm = u[a_idx]

    if pol_summary:
        summary += (
            f"Inception {str(pmin)} - {str(pmax)} inclusive, "
            + f"{len(df):,.0f} {obs_unit} with "
            + f"\\${df[prm].sum() / 10**unit_v:.1f}{unit_nm} premium, "
            + f"{n_clm}claims totalling "
            + f"\\${df[clm].sum() / 10**unit_v:.1f}{unit_nm}"
        )

    txtadd = kwargs.pop("txtadd", None)
    t = " - ".join(filter(None, [t, txtadd]))
    _ = f.suptitle("\n".join(filter(None, [t, summary])), y=1, fontsize=14)
    _ = f.tight_layout()
    return f


def plot_bootstrap_grp(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    grp: str = "grp",
    val: str = "y_eloss",
    title_add: str = "",
    force_xlim=None,
) -> figure.Figure:
    """Plot bootstrapped value, grouped by grp"""
    sty = _get_kws_styling()

    if not (
        dfboot[grp].dtype == "category"
        or pd.api.types.is_string_dtype(dfboot[grp].dtype)
    ):
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f"s{x}")

    mn = dfboot.groupby(grp, observed=True)[val].mean().tolist()
    pest_mn = df.groupby(grp, observed=True)[val].mean().values

    f = plt.figure(figsize=(12, 1.5 + (len(mn) * 0.25)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    _ = sns.violinplot(
        x=val,
        y=grp,
        hue=grp,
        data=dfboot,
        cut=0,
        density_norm="count",
        width=0.6,
        palette="cubehelix_r",
        legend=False,
        ax=ax0,
    )

    _ = [ax0.plot(v, i % len(mn), **sty["mn_pt_kws"]) for i, v in enumerate(mn)]
    _ = [
        ax0.annotate(f"{v:,.0f}", xy=(v, i % len(mn)), **sty["mn_txt_kws"])
        for i, v in enumerate(mn)
    ]
    _ = [
        ax0.plot(v, i % len(pest_mn), **sty["pest_mn_pt_kws"])
        for i, v in enumerate(pest_mn)
    ]
    _ = [
        ax0.annotate(f"{v:,.0f}", xy=(v, i % len(pest_mn)), **sty["pest_mn_txt_kws"])
        for i, v in enumerate(pest_mn)
    ]

    elems = [
        lines.Line2D([0], [0], label="population (bootstrap)", **sty["mn_pt_kws"]),
        lines.Line2D([0], [0], label="sample", **sty["pest_mn_pt_kws"]),
    ]
    _ = ax0.legend(handles=elems, loc="lower right", title="Mean Val")

    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    _ = sns.countplot(
        y=grp, hue=grp, data=df, ax=ax1, palette="cubehelix_r", legend=False
    )
    ct = df.groupby(grp, observed=True).size().tolist()
    _ = [
        ax1.annotate(f"{v}", xy=(v, i % len(ct)), **sty["count_txt_h_kws"])
        for i, v in enumerate(ct)
    ]

    if title_add != "":
        title_add = f"\n{title_add}"
    title = (
        "Grouped Mean Value (Population Estimates via Bootstrapping)"
        + f" - grouped by {grp}"
    )
    _ = f.suptitle(f"{title}{title_add}")
    _ = f.tight_layout()
    return f


def plot_bootstrap_delta_grp(dfboot, df, grp, force_xlim=None, title_add=""):
    """Plot delta between bootstrap results, grouped"""

    sty = _get_kws_styling()
    if not (
        dfboot[grp].dtype == "category"
        or pd.api.types.is_string_dtype(dfboot[grp].dtype)
    ):
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f"s{x}")

    mn = dfboot.groupby(grp, observed=True).size()

    f = plt.figure(figsize=(12, 2 + (len(mn) * 0.2)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    _ = sns.boxplot(
        x="lr_delta",
        y=grp,
        hue=grp,
        data=dfboot,
        palette="cubehelix_r",
        showfliers=False,
        whis=[3, 97],
        showmeans=True,
        notch=True,
        legend=False,
        ax=ax0,
    )
    _ = ax0.axvline(0, ls="--", lw=2, c="#555555", zorder=-1)

    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    _ = sns.countplot(
        y=grp, hue=grp, data=df, ax=ax1, palette="cubehelix_r", legend=False
    )
    ct = df.groupby(grp, observed=True).size().tolist()
    _ = [
        ax1.annotate(f"{v}", xy=(v, i % len(ct)), **sty["count_txt_h_kws"])
        for i, v in enumerate(ct)
    ]

    if title_add != "":
        title_add = f"\n{title_add}"

    title = f"2-sample bootstrap test - grouped by {grp}"
    _ = f.suptitle(f"{title}{title_add}")

    _ = f.tight_layout()
    return gs


def plot_smrystat(
    df: pd.DataFrame,
    val: str = "y_eloss",
    smry: Literal["sum", "mean"] = "sum",
    plot_outliers: bool = False,
    pal: str | list | None = None,
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist) of numeric value `val`"""
    sty = _get_kws_styling()
    idx = df[val].notnull()
    dfp = df.loc[idx].copy()

    f = plt.figure(figsize=(12, 2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1])

    ax0.set_title(f"Distribution of bootstrapped {smry}")
    ax1.set_title("Distribution of indiv. values")

    if pal is None:
        pal = "viridis"
    clr = sns.color_palette(pal, 1)[0]

    estimator = np.sum if smry == "sum" else np.mean
    _ = sns.pointplot(
        x=val,
        data=dfp,
        color=clr,
        estimator=estimator,
        errorbar=("ci", 94),
        ax=ax0,
        legend=False,
    )

    _ = sns.boxplot(
        x=val,
        data=dfp,
        color=clr,
        showfliers=plot_outliers,
        whis=[3, 97],
        showmeans=True,
        meanprops=sty["mn_pt_kws"],
        ax=ax1,
    )

    txtadd = kwargs.pop("txtadd", None)
    t = f"Diagnostic 1D plots of `{val}`"
    _ = f.suptitle("\n".join(filter(None, [t, txtadd])), y=1, fontsize=14)

    if idx.any():
        t = (
            f"Note: {sum(~idx):,.0f} NaNs found in value,"
            f"\nplotted non-NaN dataset of {sum(idx):,.0f}"
        )
        _ = ax1.annotate(
            t, xy=(0.94, 0.94), xycoords="figure fraction", ha="right", fontsize=8
        )

    ax0.xaxis.label.set_visible(False)
    ax1.xaxis.label.set_visible(False)
    _ = f.tight_layout()
    return f


def plot_smrystat_grp(
    df: pd.DataFrame,
    grp: str = "grp",
    grpkind: str = None,
    val: str = "y_eloss",
    smry: Literal["sum", "mean"] = "sum",
    plot_outliers: bool = False,
    plot_compact: bool = True,
    plot_grid: bool = True,
    pal: str | list | None = None,
    orderby: Literal["ordinal", "count", "smrystat", None] = "ordinal",
    topn: int = 10,
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist, count) of numeric value `val`
    grouped by categorical value `grp`, with group, ordered by count desc
    """
    sty = _get_kws_styling()
    est = np.sum if smry == "sum" else np.mean
    idx = df[val].notnull()
    dfp = df.loc[idx].copy()
    t = f"Diagnostic 1D plots of `{val}` grouped by `{grp}`"

    if grpkind == "year":
        dfp[grp] = dfp[grp].dt.year

    if not (
        dfp[grp].dtype == "category" or pd.api.types.is_string_dtype(dfp[grp].dtype)
    ):
        dfp[grp] = dfp[grp].map(lambda x: f"s{x}")

    ct = dfp.groupby(grp, observed=True).size()
    smrystat = dfp.groupby(grp, observed=True)[val].apply(est)

    # create order items / index
    if orderby == "count":
        ct = ct.sort_values()[::-1]
    elif orderby == "ordinal":
        pass  # ct == ct already
    elif orderby == "smrystat":
        ct = ct.reindex(smrystat.sort_values()[::-1].index)
    else:
        pass  # accept the default ordering as passed into func

    if topn is not None:
        orig_lvls = len(ct)
        ct = ct[:topn].copy()
        dfp = dfp.loc[dfp[grp].isin(ct.index.values)].copy()
        dfp[grp] = dfp[grp].cat.remove_unused_categories()
        t += f" (top {len(ct)} of {orig_lvls} levels)"

    f = plt.figure(figsize=(16, 2 + (len(ct) * 0.25)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)
    ax2 = f.add_subplot(gs[2], sharey=ax0)

    if plot_compact:
        ax1.yaxis.label.set_visible(False)
        ax2.yaxis.label.set_visible(False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

    ax0.set_title(f"Distribution of bootstrapped {smry}")
    ax1.set_title("Distribution of indiv. values")
    ax2.set_title(f"Count ({len(ct)} lvls)")

    if pal is None:
        pal = "viridis"

    kws = dict(
        y=grp,
        order=ct.index.values,
        data=dfp,
        palette=pal,
        hue=grp,
        hue_order=ct.index.values,
        legend=False,
    )
    kws_point = {**kws, **dict(estimator=est, errorbar=("ci", 94))}
    kws_box = {
        **kws,
        **dict(
            showfliers=plot_outliers,
            whis=[3, 97],
            meanprops=sty["mn_pt_kws"],
            showmeans=True,
        ),
    }

    _ = sns.pointplot(**kws_point, x=val, ax=ax0)
    _ = sns.boxplot(**kws_box, x=val, native_scale=False, ax=ax1)
    _ = sns.countplot(**kws, native_scale=False, ax=ax2)
    _ = [
        ax2.annotate(
            f"{c} ({c / ct.sum():.0%})", xy=(c, i % len(ct)), **sty["count_txt_h_kws"]
        )
        for i, c in enumerate(ct)
    ]

    if plot_grid:
        ax0.yaxis.grid(True)
        ax1.yaxis.grid(True)
        ax2.yaxis.grid(True)

    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle("\n".join(filter(None, [t, txtadd])), y=1, fontsize=14)

    if idx.any():
        t = (
            f"Note: {sum(~idx):,.0f} NaNs found in value,"
            f"\nplotted non-NaN dataset of {sum(idx):,.0f}"
        )
        _ = ax2.annotate(
            t, xy=(0.96, 0.96), xycoords="figure fraction", ha="right", fontsize=8
        )

    _ = f.tight_layout()
    return f


def plot_smrystat_grp_year(
    df: pd.DataFrame,
    grp: str = "grp",
    val: str = "y_eloss",
    year: str = "uw_year",
    smry: Literal["sum", "mean"] = "sum",
    plot_outliers: bool = True,
    plot_compact: bool = True,
    plot_grid: bool = True,
    pal: str | list | None = None,
    orderby: Literal["ordinal", "count", "smrystat", None] = "ordinal",
    topn: int = 10,
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist, count) of numeric value `val`
    grouped by categorical value `grp`, further grouped by `year`
    """

    sty = _get_kws_styling()
    est = np.sum if smry == "sum" else np.mean
    if pal is None:
        pal = "viridis"
    lvls = df.groupby(grp, observed=True).size().index.tolist()
    yrs = df.groupby(year, observed=True).size().index.tolist()
    t = f"Diagnostic 1D plots of `{val}` grouped by `{grp}` split by {year}"

    vert = min(len(lvls), topn)
    f = plt.figure(figsize=(16, len(yrs) * 2 + (vert * 0.25)))
    gs = gridspec.GridSpec(len(yrs), 3, width_ratios=[5, 5, 1], figure=f)
    ax0d, ax1d, ax2d = {}, {}, {}

    for i, yr in enumerate(yrs):  # ugly loop over years
        dfs = df.loc[df[year] == yr].copy()

        if dfs[grp].dtype not in ["object", "category", "string"]:
            dfs[grp] = dfs[grp].map(lambda x: f"s{x}")

        ct = dfs.groupby(grp, observed=True).size()
        smrystat = dfs.groupby(grp, observed=True)[val].apply(est)

        # create order items / index
        if orderby == "count":
            ct = ct.sort_values()[::-1]
        elif orderby == "ordinal":
            pass  # ct == ct already
        elif orderby == "smrystat":
            ct = ct.reindex(smrystat.sort_values()[::-1].index)
        else:
            pass  # accept the default ordering as passed into func

        if topn is not None:
            ct = ct[:topn].copy()
            dfs = dfs.loc[dfs[grp].isin(ct.index.values)].copy()
            dfs[grp] = dfs[grp].cat.remove_unused_categories()

        if i == 0:
            ax0d[i] = f.add_subplot(gs[i, 0])
            ax1d[i] = f.add_subplot(gs[i, 1], sharey=ax0d[i])
            ax2d[i] = f.add_subplot(gs[i, 2], sharey=ax0d[i])
        else:
            ax0d[i] = f.add_subplot(gs[i, 0], sharey=ax0d[0], sharex=ax0d[0])
            ax1d[i] = f.add_subplot(gs[i, 1], sharey=ax0d[i], sharex=ax1d[0])
            ax2d[i] = f.add_subplot(gs[i, 2], sharey=ax0d[i], sharex=ax2d[0])

        if plot_compact:
            ax1d[i].yaxis.label.set_visible(False)
            ax2d[i].yaxis.label.set_visible(False)
            plt.setp(ax1d[i].get_yticklabels(), visible=False)
            plt.setp(ax2d[i].get_yticklabels(), visible=False)

        ax0d[i].set_title(f'Distribution of bootstrapped {smry} [{yr:"%Y"}]')
        ax1d[i].set_title(f'Distribution of indiv. values [{yr:"%Y"}]')
        ax2d[i].set_title(f'Count [{yr:"%Y"}] ({len(ct)} lvls)')

        kws = dict(
            y=grp,
            data=dfs,
            order=ct.index.values,
            palette=pal,
            hue=grp,
            hue_order=ct.index.values,
            legend=False,
        )
        kws_point = {**kws, **dict(estimator=est, errorbar=("ci", 94))}
        kws_box = {
            **kws,
            **dict(
                showfliers=plot_outliers,
                whis=[3, 97],
                meanprops=sty["mn_pt_kws"],
                showmeans=True,
            ),
        }

        _ = sns.pointplot(**kws_point, x=val, linestyles="-", ax=ax0d[i])
        _ = sns.boxplot(**kws_box, x=val, native_scale=False, ax=ax1d[i])
        _ = sns.countplot(**kws, native_scale=False, ax=ax2d[i])
        _ = [
            ax2d[i].annotate(f"{v}", xy=(v, j % len(ct)), **sty["count_txt_h_kws"])
            for j, v in enumerate(ct)
        ]

        if plot_grid:
            ax0d[i].yaxis.grid(True)
            ax1d[i].yaxis.grid(True)
            ax2d[i].yaxis.grid(True)

    if topn is not None:
        t += f" (top {len(ct)} levels)"
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle("\n".join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout()
    return f


def plot_heatmap_corr(dfx_corr: pd.DataFrame, **kwargs) -> figure.Figure:
    """Convenience plot correlation as heatmap"""
    f, axs = plt.subplots(
        1, 1, figsize=(6 + 0.25 * len(dfx_corr), 4 + 0.25 * len(dfx_corr))
    )
    _ = sns.heatmap(
        dfx_corr,
        mask=np.triu(np.ones_like(dfx_corr), k=0),
        cmap="RdBu_r",
        square=True,
        ax=axs,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        center=0,
    )
    _ = axs.set_xticklabels(axs.get_xticklabels(), rotation=40, ha="right")
    txtadd = kwargs.pop("txtadd", None)
    t = "Feature correlations"
    _ = f.suptitle(" - ".join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout()
    return f


def plot_kj_summaries_for_linear_model(dfp, policy_id, title_add="psi"):
    """Convenience: plot summary of kj components for a linear model coeffs
    Highly coupled to summarise_kj_components_for_linear_model
    """

    idx = ~dfp["ft_mapped"].isnull()
    gd = sns.FacetGrid(
        hue="component", data=dfp.loc[idx], palette="vlag", height=4, aspect=1.5
    )
    _ = gd.map(
        sns.barplot,
        "component",
        "ft_mapped",
        order=dfp.loc[idx, "ft_mapped"],
        lw=3,
        zorder=1,
    )
    _ = gd.axes.flat[0].axvline(0, color="#dddddd", lw=3, zorder=2)
    _ = gd.axes.flat[0].set(xlabel=None, ylabel=None)  # , xticklabels=[])
    _ = gd.fig.suptitle(
        f"Components of linear submodel predictions: {title_add}\nfor policy {policy_id}",
        y=1.08,
    )

    rhs_lbls = dfp.loc[idx, "input_val_as_label"].values[::-1]

    axr = gd.axes.flat[0].twinx()
    _ = axr.plot(np.zeros(len(rhs_lbls)), np.arange(len(rhs_lbls)) + 0.5, lw=0)
    # _ = axr.set_ylim((-1,len(rhs_lbls)))
    _ = axr.set_yticks([x for x in np.arange(len(rhs_lbls)) + 0.5])
    _ = axr.set_yticklabels(rhs_lbls)
    _ = axr.yaxis.grid(False)
    _ = axr.xaxis.grid(False)
    # _ = axr.spines['top'].set_visible(False)
    # _ = axr.spines['right'].set_visible(False)
    return gd


def plot_grp_ct(
    df: pd.DataFrame,
    grp: str = "grp",
    orderby: Literal["ordinal", "count", None] = "count",
    topn: int = None,
    **kwargs,
) -> figure.Figure:
    """Simple countplot for factors in grp, label with percentages
    Works nicely with categorical too
    """

    if not (df[grp].dtype == "category" or pd.api.types.is_string_dtype(df[grp].dtype)):
        raise TypeError("grp must be Object (string) or Categorical")

    t = f"Countplot: {len(df)} obs, grouped by `{grp}`"
    sty = _get_kws_styling()
    ct = df.groupby(grp, observed=True).size()

    # create order items / index
    if orderby == "count":
        ct = ct.sort_values()[::-1]
    elif orderby == "ordinal":
        pass  # ct == ct already
    else:
        pass  # accept the default ordering as passed into func

    if topn is not None:
        ct = ct[:topn].copy()
        df = df.loc[df[grp].isin(ct.index.values)].copy()
        t += f" (top {len(ct)} levels)"

    f, axs = plt.subplots(1, 1, figsize=(12, 2 + (len(ct) * 0.25)))
    _ = sns.countplot(
        data=df,
        y=grp,
        order=ct.index,
        hue=grp,
        hue_order=ct.index,
        legend=False,
        ax=axs,
        palette="viridis",
    )
    _ = [
        axs.annotate(
            f"{v:.0f} ({v / len(df):.0%})",
            xy=(v, i % len(ct)),
            **sty["count_txt_h_kws"],
        )
        for i, v in enumerate(ct)
    ]

    _ = axs.set(ylabel=None)
    txtadd = kwargs.pop("txtadd", None)
    _ = f.suptitle(" - ".join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)

    return f


def plot_cdf_ppc_vs_obs(
    y: np.ndarray, yhat: np.ndarray, xlim_max_override=None
) -> figure.Figure:
    """Plot (quantile summaries of) yhat_ppc vs y
    NOTE:
    y shape: (nobs,)
    yhat shape: (nobs, nsamples)
    """
    ps = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]
    df_yhat_qs = pd.DataFrame(
        np.percentile(yhat, ps, axis=1).T, columns=[f"q{p / 100}" for p in ps]
    )

    f, axs = plt.subplots(1, 1, figsize=(12, 5), sharey=True, sharex=True)
    _ = sns.kdeplot(
        y, cumulative=True, lw=2, c="g", ax=axs, common_norm=False, common_grid=True
    )

    if df_yhat_qs.duplicated().sum() == len(df_yhat_qs) - 1:
        # all dupes: model was intercept only
        dfm = df_yhat_qs.iloc[:1].melt(var_name="ppc_q")
        _ = sns.rugplot(
            x="value",
            hue="ppc_q",
            data=dfm,
            palette="coolwarm",
            lw=2,
            ls="-",
            height=1,
            ax=axs,
            zorder=-1,
        )
    else:
        dfm = df_yhat_qs.melt(var_name="ppc_q")
        _ = sns.kdeplot(
            x="value",
            hue="ppc_q",
            data=dfm,
            cumulative=True,
            palette="coolwarm",
            lw=2,
            ls="-",
            ax=axs,
            zorder=-1,
            common_norm=False,
            common_grid=True,
        )

    if xlim_max_override is not None:
        _ = axs.set(xlim=(0, xlim_max_override), ylim=(0, 1))
    else:
        _ = axs.set(xlim=(0, np.ceil(y.max())), ylim=(0, 1))

    _ = f.suptitle("Cumulative density plot of the posterior predictive vs actual")

    return f


def plot_explained_variance(
    df: pd.DataFrame, k: int = 10, topn: int = 3
) -> figure.Figure:
    """Calculate Truncated SVD and plot explained variance curve, with optional
    vline for the topn components Related to eda.calc.get_svd"""

    _, svd_fit = calc_svd(df, k)
    evr = pd.Series(
        svd_fit.explained_variance_ratio_.cumsum(), name="explained_variance_csum"
    )
    evr.index = np.arange(1, len(evr) + 1)
    evr.index.name = "component"

    f, axs = plt.subplots(1, 1, figsize=(12, 5))
    _ = sns.pointplot(
        x="component", y="explained_variance_csum", data=evr.reset_index(), ax=axs
    )
    _ = axs.vlines(topn - 1, 0, 1, "orange", "-.")
    _ = axs.annotate(
        "{:.1%}".format(evr[topn]),
        xy=(topn - 1, evr[topn]),
        xycoords="data",
        xytext=(-10, 10),
        textcoords="offset points",
        color="orange",
        ha="right",
        fontsize=12,
    )

    _ = axs.set_ylim(0, 1.001)
    _ = f.suptitle(
        f"Explained variance @ top {topn} components ~ {evr[topn]:.1%}", fontsize=14
    )
    _ = f.tight_layout()
    return f
