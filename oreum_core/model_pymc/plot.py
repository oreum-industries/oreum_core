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

# model.plot.py
"""Model Plotting"""

from copy import copy, deepcopy
from enum import Enum

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import figure, gridspec, lines, ticker
from scipy import integrate

from ..model_pymc import BasePYMCModel

az.rcParams["plot.max_subplots"] = 200

__all__ = [
    "plot_trace",
    "plot_energy",
    "facetplot_kruschke",
    "pairplot_corr",
    "forestplot_single",
    "forestplot_multiple",
    "plot_ppc",
    "plot_loo_pit",
    "plot_compare",
    "plot_lkjcc_corr",
    "plot_coverage",
    "plot_rmse_range",
    "plot_estimate",
    "plot_roc_precrec",
    "plot_f_measure",
    "plot_accuracy",
    "plot_binary_performance",
]


class IDataGroupName(str, Enum):
    prior = "prior"
    posterior = "posterior"


def plot_trace(mdl: BasePYMCModel, rvs: list, **kwargs) -> figure.Figure:
    """Create traceplot for passed mdl NOTE a useful kwarg is `kind` e.g.
    'trace', the default is `kind = 'rank_vlines'`"""
    kind = kwargs.pop("kind", "rank_vlines")
    txtadd = kwargs.pop("txtadd", None)
    _ = az.plot_trace(
        mdl.idata, var_names=rvs, kind=kind, figsize=(12, 0.8 + 1.8 * len(rvs))
    )
    f = plt.gcf()
    _ = f.suptitle(
        " - ".join(filter(None, [f"Posterior traces of {rvs}", txtadd]))
        + f"\n{mdl.mdl_id}"
    )
    _ = f.tight_layout()
    return f


def plot_energy(mdl: BasePYMCModel) -> figure.Figure:
    """Simple wrapper around energy plot to provide a simpler interface"""
    _ = az.plot_energy(
        mdl.idata, fill_alpha=(0.8, 0.6), fill_color=("C0", "C8"), figsize=(12, 1.8)
    )
    f = plt.gcf()
    _ = f.suptitle(
        "NUTS Energy (Marginal vs Transitional, and E-BFMI)" + f" - `{mdl.mdl_id}`"
    )
    _ = f.tight_layout()
    return f


def facetplot_kruschke(
    mdl: BasePYMCModel,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    ref_vals: dict = None,
    **kwargs,
) -> figure.Figure:
    """Create Kruschke-style plots using Arviz, univariate RVs, control faceting
    NOTE
    + ref_vals should look like a dict of list of dict with a key "ref_val"
        e.g. ref_vals = { 'beta_sigma' : [ {'ref_val':2} ] }
    + Optional Pass kwargs like hdi_prob = 0.5, coords = {'oid', oids}
    """
    txtadd = kwargs.pop("txtadd", None)
    transform = kwargs.pop("transform", None)

    _, flt = az.sel_utils.xarray_to_ndarray(mdl.idata.get(group), var_names=rvs)
    nvars = flt.shape[0]
    m = min(nvars, 4)
    n = (nvars + m - 1) // m
    f, axs = plt.subplots(n, m, figsize=(3 * m, 0.8 + 1.5 * n))
    _ = az.plot_posterior(
        mdl.idata,
        group=group,
        ax=axs,
        var_names=rvs,
        ref_val=ref_vals,
        transform=transform,
        **kwargs,
    )
    _ = f.suptitle(
        " - ".join(filter(None, [f"Distribution of {rvs}", group, txtadd]))
        + f"\n{mdl.mdl_id}"
    )
    p = f.subplotpars
    _ = f.subplots_adjust(hspace=max(0.1, p.hspace), wspace=max(0.1, p.wspace))
    _ = f.tight_layout()
    return f


def forestplot_single(
    mdl: BasePYMCModel,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    kind: str = "forestplot",
    d_rebase_coords: dict = None,
    **kwargs,
) -> figure.Figure:
    """Plot forestplot or ridgeplot for list of rvs (optional factor sublevels)
    Pass d_rebase_coords with a dict (single pair only) of coords to rebase upon
    e.g. {"oid": "t"} to replace "oid" with "t" (in a copied version of idata)
    and thus allow plotting e.g. a deterministic rv vs t (rather than oid)
    NOTE the dict value e.g. "t" must be present in mdl.idata.constant_data
    """
    txtadd = kwargs.pop("txtadd", None)
    dp = kwargs.pop("dp", 2)
    plot_mn = kwargs.pop("plot_mn", True)
    transform = kwargs.pop("transform", None)
    desc = None
    kws = dict(
        colors=sns.color_palette("tab20c", n_colors=16).as_hex()[
            kwargs.pop("clr_offset", 0) :
        ][0],
        ess=False,
        combined=kwargs.pop("combined", True),
        ridgeplot_overlap=4,
        ridgeplot_alpha=0.8,
        coords=kwargs.pop("coords", None),
    )

    if d_rebase_coords is None:
        idata0 = mdl.idata
    else:
        if len(d_rebase_coords) > 1:
            raise NotImplementedError("Only accepting one dict pair for now")
        xa_dataset = deepcopy(mdl.idata[group])
        old_coord_nm = list(d_rebase_coords.keys())[0]
        new_coord_nm = list(d_rebase_coords.values())[0]
        new_coord_vals = mdl.idata.constant_data[new_coord_nm].values
        xa_dataset0 = xa_dataset.assign_coords(**{old_coord_nm: new_coord_vals})
        xa_dataset1 = xa_dataset0.sortby(old_coord_nm)
        idata0 = az.InferenceData(**{group: xa_dataset1})

    # get overall stats
    df = az.extract(idata0, group=group, var_names=rvs).to_dataframe()
    if transform is not None:
        df = df.apply(transform)
    if len(rvs) == 1:
        mn = df[rvs[0]].mean(axis=0)
        qs = df[rvs[0]].quantile(q=[0.03, 0.97]).values
        desc = (
            f"Overall: $Mean =$ {mn:.{dp}f}"
            + ", $HDI_{94}$ = ["
            + ", ".join([f"{qs[v]:.{dp}f}" for v in range(2)])
            + "]"
        )
    nms = [nm for nm in df.index.names if nm not in ["chain", "draw"]]
    n = sum([len(df.index.get_level_values(nm).unique()) for nm in nms])

    f = plt.figure(figsize=(12, 1.5 + 0.2 * n))
    ax0 = f.add_subplot()
    _ = az.plot_forest(
        idata0[group], var_names=rvs, transform=transform, kind=kind, ax=ax0, **kws
    )
    _ = ax0.set_title("")

    if plot_mn and len(rvs) == 1:
        _ = ax0.axvline(mn, color="#ADD8E6", ls="--", lw=3, zorder=-1)
    else:
        _ = ax0.axvline(0, color="#ADD8E6", ls="--", lw=3, zorder=-1)
    _ = f.suptitle(
        "\n".join(
            filter(
                None,
                [
                    " - ".join(
                        filter(None, [f"{kind.title()} of {rvs}", group, txtadd])
                    ),
                    mdl.mdl_id,
                    desc,
                ],
            )
        )
    )
    _ = f.tight_layout()
    return f


def forestplot_multiple(
    mdl: BasePYMCModel,
    datasets: dict[str, xr.core.dataarray.DataArray],
    group: IDataGroupName = IDataGroupName.posterior.value,
    **kwargs,
) -> figure.Figure:
    """Plot set of forestplots for related datasets RVs
    Useful for a linear model of RVs, where each RV can have sublevel factors
    TODO This makes a few too many assumptions, will improve in future
    """
    txtadd = kwargs.pop("txtadd", None)
    clr_offset = kwargs.pop("clr_offset", 0)
    dp = kwargs.pop("dp", 1)
    plot_med = kwargs.pop("plot_med", True)
    plot_combined = kwargs.pop("plot_combined", False)
    vsize = kwargs.pop("vsize", 10)
    desc = None

    hs = [0.22 * (np.prod(data.shape[2:])) for data in datasets.values()]
    f = plt.figure(figsize=(12, vsize + (0.5 * len(datasets)) + (0.2 * sum(hs))))
    gs = gridspec.GridSpec(len(hs), 1, height_ratios=hs, figure=f)

    for i, (txt, data) in enumerate(datasets.items()):
        ax = f.add_subplot(gs[i])
        _ = az.plot_forest(
            data,
            ax=ax,
            colors=sns.color_palette("tab20c", n_colors=16).as_hex()[clr_offset:][i],
            ess=False,
            combined=plot_combined,
        )

        _ = ax.set_title(txt)
        if plot_med:
            if i == 0:
                qs = np.quantile(data, q=[0.03, 0.25, 0.5, 0.75, 0.97])
                _ = ax.axvline(qs[2], color="#ADD8E6", ls="--", lw=3, zorder=-1)
                desc = (
                    f"med {qs[2]:.{dp}f}, HDI50 ["
                    + ", ".join([f"{qs[v]:.{dp}f}" for v in [1, 3]])
                    + "], HDI94 ["
                    + ", ".join([f"{qs[v]:.{dp}f}" for v in [0, 4]])
                    + "]"
                )
            else:
                _ = ax.axvline(1, color="#ADD8E6", ls="--", lw=3, zorder=-1)

    _ = f.suptitle(
        "\n".join(
            filter(
                None,
                [
                    " - ".join(filter(None, ["Forestplot levels", group, txtadd])),
                    mdl.mdl_id,
                    desc,
                ],
            )
        )
    )
    _ = f.tight_layout()
    return f


def pairplot_corr(
    mdl: BasePYMCModel,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    ref_vals: dict = None,
    **kwargs,
) -> figure.Figure:
    """Create posterior pair / correlation plots using Arviz, correlated RVs,
    Pass-through kwargs to az.plot_pair, e.g. ref_vals
    Default to posterior, allow for override to prior
    """
    txtadd = kwargs.pop("txtadd", None)
    kind = kwargs.pop("kind", "kde")
    coords = kwargs.pop("coords", None)
    # ignore hsize, wsize
    kwargs.pop("hsize", None)
    kwargs.pop("wsize", None)

    data = mdl.idata.get(group)
    if coords:
        data = data.sel(**coords)
    _, flt = az.sel_utils.xarray_to_ndarray(data, var_names=rvs)
    nvars = flt.shape[0]

    pair_kws = dict(
        group=group,
        var_names=rvs,
        reference_values=ref_vals,  # deal with inconsistent naming
        divergences=True,
        marginals=True,
        kind=kind,
        kde_kwargs=dict(
            contourf_kwargs=dict(alpha=0.5, cmap="Blues"),
            contour_kwargs=dict(colors=None, cmap="Blues"),
            hdi_probs=[0.5, 0.94, 0.99],
        ),
        figsize=(2 + 1.4 * nvars, 2 + 1.4 * nvars),
    )
    if coords is not None:
        pair_kws["coords"] = coords
    pair_kws.update(kwargs)
    axs = az.plot_pair(mdl.idata, **pair_kws)
    corr = pd.DataFrame(az.sel_utils.xarray_to_ndarray(data, var_names=rvs)[1].T).corr()
    n_corr = len(corr)
    n_ax = axs.shape[0] if hasattr(axs, "shape") else int(np.sqrt(axs.size))
    i, j = np.tril_indices(n=min(n_corr, n_ax), k=-1)
    for ij in zip(i, j, strict=False):
        axs[ij].set_title(f"rho: {corr.iloc[ij]:.2f}", fontsize=6, loc="right", pad=0)
    vh_y = dict(rotation=20, va="center", ha="right", fontsize=7)
    vh_x = dict(rotation=20, va="top", ha="right", fontsize=7)
    for ax in axs.flat:
        ax.tick_params(axis="both", labelsize=6)
        ax.set_ylabel(ax.get_ylabel(), **vh_y)
        ax.set_xlabel(ax.get_xlabel(), **vh_x)

    f = plt.gcf()
    _ = f.suptitle(
        " - ".join(filter(None, ["Pairplot", mdl.name, group, "selected RVs", txtadd]))
    )
    p = f.subplotpars
    _ = f.subplots_adjust(hspace=max(0.1, p.hspace), wspace=max(0.1, p.wspace))
    _ = f.tight_layout()
    return f


def plot_ppc(
    mdl: BasePYMCModel,
    var_names: list,
    idata: az.InferenceData = None,
    group: str = "posterior",
    ecdf: bool = True,
    flatten: list = None,
    observed_rug: bool = True,
    logx: bool = False,
    **kwargs,
) -> figure.Figure:
    """Plot In- or Out-of-Sample Prior or Posterior Retrodictive. Does not
    require log-likelihood. Does require `observed_data`, which is not made by
    Potentials
    NOTE:
    + use var_names to only plot e.g. yhat
    + pass through kwargs, possibly of particular use is:
        `data_pairs` = {key (in observed_data): value (in {group}_predictive)}
        although we remind that the constant_data has the real name, but once
        it's observed in a log-likelihood the idata.observed_data will get the
        same name as the {group}_predictive, so data_pairs is not often needed
    """
    txtadd = kwargs.pop("txtadd", None)
    sharex = kwargs.pop("sharex", True)
    xlim = kwargs.pop("xlim", None)
    kind = "kde"
    kindnm = kind.upper()
    ynm = "density"
    loc = "upper right"
    n = len(var_names)
    if ecdf:
        kind = "cumulative"
        kindnm = "ECDF"
        ynm = "prop"
        loc = "lower right"

    if idata is None:
        _idata = mdl.idata
        insamp = True
    else:
        _idata = idata
        insamp = False

    if flatten is not None:
        n = 1
        for k in var_names:
            n *= _idata["observed_data"][k].shape[-1]
    # wild hack to get the size of observed
    i = list(dict(_idata.observed_data.sizes).values())[0]
    num_pp_samples = None if i < 500 else 200
    f, axs = plt.subplots(n, 1, figsize=(12, 3 + 2 * n), sharex=sharex, squeeze=False)
    _ = az.plot_ppc(
        _idata,
        group=group,
        kind=kind,
        ax=axs,
        var_names=var_names,
        flatten=flatten,
        observed_rug=observed_rug,
        random_seed=42,
        num_pp_samples=num_pp_samples,
        **kwargs,
    )
    _ = [ax.legend(fontsize=8, loc=loc) for ax in axs.flat]  # fix legend
    ls = None
    if logx:
        _ = [ax.set_xscale("log") for ax in axs.flat]
        ls = "(logscale)"
    if xlim is not None:
        _ = [ax.set_xlim(xlim) for ax in axs.flat]
    _ = [
        ax.set(title=t, ylabel=ynm) for ax, t in zip(axs.flat, var_names, strict=False)
    ]
    t = f"{'In' if insamp else 'Out-of'}-sample {group.title()} Retrodictive {kindnm}"
    _ = f.suptitle(" --- ".join(filter(None, [t, txtadd, ls])) + f"\n{mdl.mdl_id}")
    _ = f.tight_layout()
    return f


def plot_loo_pit(
    mdl: BasePYMCModel, data_pairs: dict = None, **kwargs
) -> figure.Figure:
    """Calc and plot LOO-PIT after run `mdl.sample_posterior_predictive()`
    ref: https://oriolabrilpla.cat/en/blog/posts/2019/loo-pit-tutorial.html
    NOTE:
    mdl.idata needs: observed_data AND log_likelihood AND posterior_predictive
    data_pairs {key (in observed AND log_likelihood): value (in posterior_predictive)}

    """
    txtadd = kwargs.pop("txtadd", None)
    f, axs = plt.subplots(
        len(data_pairs), 2, figsize=(12, 3 * len(data_pairs)), squeeze=False
    )
    for i, (y, yhat) in enumerate(data_pairs.items()):
        kws = dict(y=y, y_hat=yhat)
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][0], **kwargs)
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][1], ecdf=True, **kwargs)

        _ = axs[i][0].set(
            title=f"Predicted {yhat} LOO-PIT", xlabel="PIT", ylabel="ECDF"
        )
        _ = axs[i][1].set(
            title=f"Predicted {yhat} LOO-PIT cumulative",
            xlabel="PIT",
            ylabel="\u0394 ECDF",
        )

    _ = f.suptitle(
        " - ".join(filter(None, ["In-sample LOO-PIT", txtadd])) + f"\n{mdl.mdl_id}"
    )
    _ = f.tight_layout()
    return f


def plot_compare(
    mdl_dict: dict[str, BasePYMCModel], yhats: list[str], **kwargs
) -> tuple[figure.Figure, dict[str, pd.DataFrame]]:
    """Calc and plot model comparison in-sample via expected log pointwise
    predictive density (ELPD) using LOO
    NOTE:
    idata needs: observed_data AND log_likelihood
    hats should be the key for observed_data AND log_likelihood
    """
    txtadd = kwargs.pop("txtadd", None)
    sharex = kwargs.pop("sharex", False)
    f, axs = plt.subplots(
        len(yhats),
        1,
        figsize=(12, 2.6 * len(yhats) + 0.2 * len(mdl_dict)),
        squeeze=False,
        sharex=sharex,
    )
    # mdlnms = ' vs '.join(idata_dict.keys())
    idata_dict = {f"{k}\n{v.mdl_id_fn}": v.idata for k, v in mdl_dict.items()}
    dcomp = {}
    for i, y in enumerate(yhats):
        dfcomp = az.compare(
            idata_dict, var_name=y, ic="loo", method="stacking", scale="log"
        )
        dcomp[y] = dfcomp
        ax = az.plot_compare(
            dfcomp, ax=axs[i][0], title=False, textsize=10, legend=False
        )
        _ = ax.set_title(y)
    t = (
        "Model Performance Comparison: ELPD via In-Sample LOO-PIT:\n`"
        + "` vs `".join(list(mdl_dict.keys()))
        + "`\n(higher & narrower is better)"
    )
    _ = f.suptitle(" - ".join(filter(None, [t, txtadd])))
    _ = f.tight_layout()

    return f, dcomp


def plot_lkjcc_corr(mdl: BasePYMCModel, **kwargs) -> figure.Figure:
    """Plot lkjcc_corr model RVs
    Drop diagonals, assume coord is called lkjcc_corr
    Also see https://python.arviz.org/en/stable/user_guide/label_guide.html#custom-labellers
    """
    coords = {
        "lkjcc_corr_dim_0": xr.DataArray([0, 1], dims=["asdf"]),
        "lkjcc_corr_dim_1": xr.DataArray([1, 0], dims=["asdf"]),
    }

    return facetplot_kruschke(
        mdl=mdl,
        txtadd="lkjcc_corr, diagonals only",
        rvs=["lkjcc_corr"],
        coords=coords,
        **kwargs,
    )


def plot_yhat_vs_y(
    mdl: BasePYMCModel,
    dfhat: pd.DataFrame,
    yhat: str = "yhat",
    y: str = "y",
    oid: str = "oid",
    insamp: bool = False,
    **kwargs,
) -> figure.Figure:
    """Boxplot forecast yhat with overplotted y"""
    txtadd = kwargs.pop("txtadd", None)
    kws_mn = dict(
        markerfacecolor="w", markeredgecolor="#333333", marker="d", markersize=12
    )
    kws_box = dict(
        kind="box", showfliers=False, showmeans=True, whis=(3, 97), meanprops=kws_mn
    )
    kws_sctr = dict(s=80, color="#32CD32")

    g = sns.catplot(
        x=yhat, y=oid, data=dfhat.reset_index(), **kws_box, height=4, aspect=3
    )
    _ = g.map(sns.scatterplot, y, oid, **kws_sctr, zorder=100)
    t_io = (
        f"{'In' if insamp else 'Out-of'}-sample: boxplots of posterior `{yhat}`"
        + f" with overplotted actual `{y}` values per observation"
        + f" `{oid}` (green dots) - `{mdl.name}`"
    )
    _ = g.fig.suptitle(" - ".join(filter(None, [t_io, txtadd])) + f"\n{mdl.mdl_id}")

    _ = g.tight_layout()
    return g.fig


def plot_coverage(df: pd.DataFrame, **kwargs) -> figure.Figure:
    """Convenience plot PPC coverage from model_pymc.calc.calc_ppc_coverage"""

    txt_kws = dict(
        color="#555555",
        xycoords="data",
        xytext=(2, -4),
        textcoords="offset points",
        fontsize=11,
        backgroundcolor="w",
    )

    g = sns.lmplot(
        x="cr",
        y="coverage",
        col="method",
        hue="method",
        data=df,
        fit_reg=False,
        height=4,
        scatter_kws={"s": 70},
    )
    txtadd = kwargs.get("txtadd", None)
    for i, method in enumerate(df["method"].unique()):
        idx = df["method"] == method
        y = df.loc[idx, "coverage"].values
        x = df.loc[idx, "cr"].values
        ae = np.abs(y - x)
        auc = integrate.trapezoid(ae, x)

        g.axes[0][i].plot((0, 1), (0, 1), ls="--", color="#aaaaaa", zorder=-1)
        g.axes[0][i].fill_between(x, y, x, color="#bbbbbb", alpha=0.8, zorder=-1)
        g.axes[0][i].annotate(f"AUC={auc:.3f}", xy=(0, 1), **txt_kws)

    t = "PPC Coverage vs CR"
    _ = g.fig.suptitle(" - ".join(filter(None, [t, txtadd])), y=1.05, fontsize=14)

    return g.fig


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


def plot_rmse_range(
    rmse: float, rmse_qs: pd.Series, yhat: str = "yhat", y: str = "y", **kwargs
) -> figure.Figure:
    """Convenience to plot RMSE mean and qs from model_pymc.calc.calc_rmse
    Requires that `rmse_qs` has an entry at the median 0.5"""
    txtadd = kwargs.get("txtadd", None)
    dfp = rmse_qs.to_frame()
    min_rmse_qs = rmse_qs.min()
    min_rmse_qs_q = rmse_qs.idxmin()
    j = max(-int(np.ceil(np.log10(rmse))) + 2, 1)

    f, axs = plt.subplots(1, 1, figsize=(12, 4))
    ax = sns.lineplot(x="q", y="rmse", data=dfp, lw=2, ax=axs)
    _ = ax.axhline(rmse, c="r", ls="-.", label=f"rmse @ mean {rmse:,.{j}f}")
    _ = ax.axhline(
        rmse_qs[0.5], c="b", ls="--", label=f"rmse @ med (q0.5) {rmse_qs[0.5]:,.{j}f}"
    )
    _ = ax.axhline(
        min_rmse_qs,
        c="g",
        ls="--",
        label=f"rmse @ min (q{min_rmse_qs_q:,.{j}f}) {min_rmse_qs:,.{j}f}",
    )
    _ = ax.legend()
    t = f"RMSE range for `{yhat}` vs `{y}`"
    tq = f"qs in [{rmse_qs.index.min()}, {rmse_qs.index.max()}]"
    _ = f.suptitle(" - ".join(filter(None, [t, tq, txtadd])))
    _ = f.tight_layout()
    return f


def plot_estimate(
    yhat: np.ndarray,
    nobs: int,
    yhat_nm: str = "yhat",
    force_xlim: list = None,
    color: str = None,
    exceedance: bool = False,
    y: np.ndarray = None,
    y_nm: str = "y",
    **kwargs,
) -> figure.Figure:
    """Plot distribution of univariate estimates in 1D array yhat:
        typically used for PPC output pre-summarised across observations or for
        bootstrap summarised observed data. The sampling / bootstrapping and
        summary stat should have already been applied to yhat
    Default to boxplot, allow exceedance curve.
    Optionally overplot true values y 1D array: this will be shown as a modified
        sns.pointplot (which has internal bootstrapping to show the mean of y)
        NOTE we slightly abuse the pointplot by (1) set n_boot = 1, and (2)
        set errorbar to show y CR94(min, max), rather than the default which is
        some ci of internal bootstrapping. This makes the pointplot comparable
        to the boxplot, but usefully different in drawing style
    Refactored this to operate on simple arrays
    """

    def _tuple_errorbar(a) -> tuple[float]:
        """Convenient hack for pointplot errorbar to return CR94"""
        cr = np.quantile(a, q=[0.03, 0.97])
        return (cr[0], cr[1])

    txtadd = kwargs.pop("txtadd", None)
    t = f"{yhat_nm} for {nobs} obs"
    sty = _get_kws_styling()
    clr = color if color is not None else sns.color_palette()[0]
    kws = {"color": clr}
    kws_box = kws | {
        "showfliers": False,
        "orient": "h",
        "showmeans": True,
        "whis": (3, 97),
        "meanprops": sty["mn_pt_kws"],
    }
    kws_exc = kws | {"complementary": True, "lw": 3, "legend": None}
    kws_pt = {
        "estimator": np.mean,
        "n_boot": 1,
        "errorbar": _tuple_errorbar,
        "linestyle": "none",
        "orient": "h",
        "color": "C1",
    }
    mn_pt_kws = copy(sty["mn_pt_kws"])
    mn_pt_kws.update(
        markerfacecolor="C1", markeredgecolor="C1", c="C1", marker="o", markersize=8
    )
    mn_txt_kws = copy(sty["mn_txt_kws"])
    mn_txt_kws["backgroundcolor"] = "C1"
    mn = yhat.mean()
    sigfigs_min = 3
    j = max(-int(np.ceil(np.log10(mn))), sigfigs_min)
    f, axs = plt.subplots(1, 1, figsize=(12, 3 + 2 * exceedance))

    if not exceedance:  # default to boxplot, nice and simple
        ax = sns.boxplot(x=yhat, y=0, ax=axs, **kws_box)
        _ = ax.annotate(f"{mn:,.{j}g}", xy=(mn, 0), **sty["mn_txt_kws"])
        elems = [
            lines.Line2D(
                [0], [0], label=f"{yhat_nm} (sample $\\mu$)", **sty["mn_pt_kws"]
            )
        ]
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            mn_y = y.mean()
            j_y = max(-int(np.ceil(np.log10(mn_y))), sigfigs_min)
            _ax = sns.pointplot(data=y, ax=axs, **kws_pt)
            _ = _ax.annotate(f"{mn_y:,.{j_y}g}", xy=(mn_y, 0), **mn_txt_kws)
            elems.append(
                lines.Line2D([0], [0], label=f"{y_nm} (sample $\\mu$)", **mn_pt_kws)
            )
            txtadd = ". ".join(filter(None, [txtadd, f"Overplotted {y_nm}"]))

        _ = ax.legend(handles=elems, loc="lower right", fontsize=8)
        _ = ax.set(yticklabels="", xlabel=yhat_nm)

        if force_xlim is not None:
            _ = ax.set(xlim=force_xlim)

        hdi = np.quantile(a=yhat, q=[0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97])
        smry_stats = (
            f"$\\mu = {mn:,.{j}g}$; "
            + f"$q_{{50}} = {hdi[3]:,.{j}g}$; "
            + f"$HDI_{{50}} = [{hdi[2]:,.{j}g}, {hdi[4]:,.{j}g}]$; "
            + f"$HDI_{{94}} = [{hdi[0]:,.{j}g}, {hdi[6]:,.{j}g}]$"
        )
        t = " ".join(filter(None, ["Boxplot", t]))

    else:  # do exceedance, slightly less intuitive for beginner clients
        ax = sns.ecdfplot(x=yhat, ax=axs, **kws_exc)
        _ = ax.set(ylabel=f"P({yhat_nm} ≥ x)", xlabel="x")
        qs = kwargs.pop("qs", np.array([0.5, 0.9, 0.95, 0.99]))
        qvals = np.quantile(a=yhat, q=qs)
        clrs = sns.color_palette("Blues", len(qs))
        for i, (q, qv) in enumerate(zip(qs, qvals, strict=True)):
            _ = ax.vlines(x=qv, ymin=0, ymax=1 - q, lw=2, zorder=-1, colors=clrs[i])
        ax1 = sns.scatterplot(
            x=qvals,
            y=1 - qs,
            hue=qs,
            palette=clrs,
            style=qs,
            markers=["s", "o", "^", "d"],
            edgecolor="#999",
            ax=axs,
            s=120,
            zorder=10,
        )
        hdls, _lbls = ax1.get_legend_handles_labels()
        lbls = [str(round(1 - float(lbl), 2)) for lbl in _lbls]  # HACK floating pt
        _ = ax1.legend(hdls, lbls, loc="upper right", title=f"P({yhat_nm}) ≥ x")
        smry_stats = ", ".join(
            [
                f"$P_{{@{{{q:.2f}}}}} \\geq {{{qv:.{j}g}}}$"
                for q, qv in zip(1 - qs, qvals, strict=True)
            ]
        )
        t = " ".join(filter(None, ["Exceedance Curve", t]))
        _ = ax.set(xlabel=yhat_nm)

    if kwargs.get("ispercent", False):
        _ = ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    _ = f.suptitle(", ".join(filter(None, [t, txtadd])) + f"\nSummary: {smry_stats}")
    _ = f.tight_layout()
    return f


def plot_roc_precrec(df: pd.DataFrame) -> tuple[figure.Figure, float, float]:
    """Plot ROC and PrecRec, also calc and return AUC
    Pass perf df from calc.calc_binary_performance_measures
    """

    roc_auc = integrate.trapezoid(y=df["tpr"], x=df["fpr"])
    prec_rec_auc = integrate.trapezoid(y=df["precision"], x=df["recall"])

    f, axs = plt.subplots(1, 2, figsize=(11.5, 6), sharex=True, sharey=True)
    _ = f.suptitle("ROC and Precision Recall Curves", y=1.0)

    _ = axs[0].plot(
        df["fpr"],
        df["tpr"],
        lw=2,
        marker="d",
        alpha=0.8,
        label=f"ROC (auc={roc_auc:.2f})",
    )
    _ = axs[0].plot((0, 1), (0, 1), "--", c="#cccccc", label="line of equiv")
    _ = axs[0].legend(loc="upper left")
    _ = axs[0].set(title="ROC curve", xlabel="FPR", ylabel="TPR")

    _ = axs[1].plot(
        df["recall"],
        df["precision"],
        lw=2,
        marker="o",
        alpha=0.8,
        color="C3",
        label=f"PrecRec (auc={prec_rec_auc:.2f})",
    )
    _ = axs[1].legend(loc="upper right")
    _ = axs[1].set(title="Precision Recall curve", xlabel="Recall", ylabel="Precision")

    f.tight_layout()

    return f, roc_auc, prec_rec_auc


def plot_f_measure(df: pd.DataFrame) -> figure.Figure:
    """Plot F-measures (F0.5, F1, F2) at different percentiles"""

    f1_at = df["f1"].argmax()
    dfm = df.reset_index()[["pct", "f0.5", "f1", "f2"]].melt(
        id_vars="pct", var_name="f-measure", value_name="f-score"
    )
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(
        x="pct", y="f-score", hue="f-measure", data=dfm, palette="Greens", lw=2, ax=axs
    )
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle(
        "F-scores across the percentage range of PPC"
        + f"\nBest F1 = {df.loc[f1_at, 'f1']:.3f} @ {f1_at} pct",
        y=1.03,
    )
    return f


def plot_accuracy(df: pd.DataFrame) -> figure.Figure:
    """Plot accuracy at different percentiles"""

    acc_at = df["accuracy"].argmax()
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(x="pct", y="accuracy", color="C1", data=df, lw=2, ax=axs)
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle(
        "Accuracy across the percentage range of PPC"
        + f"\nBest = {df.loc[acc_at, 'accuracy']:.1%} @ {acc_at} pct",
        y=1.03,
    )
    return f


def plot_binary_performance(
    dfperf: pd.DataFrame, nobs: int = 1, **kwargs
) -> figure.Figure:
    """Plot ROC, PrecRec, F-score, Accuracy sweeping across PPC sample quantiles
    Created for perf df from model_pymc.calc.calc_binary_performance_measures
    Return summary stats
    """
    f, axs = plt.subplots(1, 4, figsize=(18, 5), sharex=False, sharey=False)

    # ROC -------------
    roc_auc = integrate.trapezoid(y=dfperf["tpr"], x=dfperf["fpr"])
    r_at = np.argmin(np.sqrt(dfperf["fpr"] ** 2 + (1 - dfperf["tpr"]) ** 2))
    r_at = np.round(r_at / 100, 2)

    _ = axs[0].plot(
        dfperf["fpr"],
        dfperf["tpr"],
        lw=2,
        marker="d",
        alpha=0.8,
        label=f"ROC (auc={roc_auc:.2f})",
    )
    _ = axs[0].plot((0, 1), (0, 1), "--", c="#cccccc", label="line of equiv")
    _ = axs[0].plot(
        dfperf.loc[r_at, "fpr"],
        dfperf.loc[r_at, "tpr"],
        lw=2,
        marker="D",
        color="w",
        markeredgewidth=1,
        markeredgecolor="b",
        markersize=9,
        label=f"Optimum ROC @ q{r_at}",
    )
    _ = axs[0].legend(loc="lower right")
    _ = axs[0].set(title="ROC curve", xlabel="FPR", ylabel="TPR", ylim=(0, 1))

    # Precision-Recall -------------
    prec_rec_auc = integrate.trapezoid(y=dfperf["precision"], x=dfperf["recall"])
    _ = axs[1].plot(
        dfperf["recall"],
        dfperf["precision"],
        lw=2,
        marker="o",
        alpha=0.8,
        color="C3",
        label=f"PrecRec (auc={prec_rec_auc:.2f})",
    )
    _ = axs[1].legend(loc="upper right")
    _ = axs[1].set(
        title="Precision Recall curve", ylim=(0, 1), xlabel="Recall", ylabel="Precision"
    )

    # F-measure -------------
    f1_at = np.round(dfperf["f1"].argmax() / 100, 2)
    dfm = dfperf.reset_index()[["q", "f0.5", "f1", "f2"]].melt(
        id_vars="q", var_name="f-measure", value_name="f-score"
    )

    _ = sns.lineplot(
        x="q", y="f-score", hue="f-measure", data=dfm, palette="Greens", lw=2, ax=axs[2]
    )
    _ = axs[2].plot(
        f1_at,
        dfperf.loc[f1_at, "f1"],
        lw=2,
        marker="D",
        color="w",
        markeredgewidth=1,
        markeredgecolor="b",
        markersize=9,
        label=f"Optimum F1 @ q{f1_at}",
    )
    _ = axs[2].legend(loc="upper left")
    _ = axs[2].set(
        title="F-measures across the PPC qs"
        + f"\nBest F1 = {dfperf.loc[f1_at, 'f1']:.3f} @ q{f1_at}",
        xlabel="q",
        ylabel="F-Score",
        ylim=(0, 1),
    )

    # Accuracy -------------
    acc_at = np.round(dfperf["accuracy"].argmax() / 100, 2)
    _ = sns.lineplot(x="q", y="accuracy", color="C1", data=dfperf, lw=2, ax=axs[3])
    _ = axs[3].text(
        x=0.04,
        y=0.04,
        s=(
            "Class imbalance:"
            + f"\n0: {dfperf['accuracy'].values[0]:.1%}"
            + f"\n1: {dfperf['accuracy'].values[-1]:.1%}"
        ),
        transform=axs[3].transAxes,
        ha="left",
        va="bottom",
        backgroundcolor="w",
        fontsize=10,
    )
    _ = axs[3].set(
        title="Accuracy across the PPC pcts"
        + f"\nBest = {dfperf.loc[acc_at, 'accuracy']:.1%} @ q{acc_at}",
        xlabel="q",
        ylabel="Accuracy",
        ylim=(0, 1),
    )

    t = (
        "Evaluations of Binary Predictions made by sweeping across PPC "
        + f"sample quantiles (more reliable if nobs>100: here nobs={nobs})"
    )
    txtadd = kwargs.get("txtadd", None)
    _ = f.suptitle("\n".join(filter(None, [t, txtadd])), y=1.0)
    _ = f.tight_layout()
    return f
