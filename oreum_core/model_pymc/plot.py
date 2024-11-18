# Copyright 2024 Oreum Industries
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
from enum import Enum

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import figure, gridspec

from ..model_pymc import BasePYMCModel

__all__ = [
    'plot_trace',
    'plot_energy',
    'facetplot_krushke',
    'pairplot_corr',
    'forestplot_single',
    'forestplot_multiple',
    'plot_ppc',
    'plot_loo_pit',
    'plot_compare',
    'plot_lkjcc_corr',
]

sns.set_theme(
    style='darkgrid',
    palette='muted',
    context='notebook',
    rc={'figure.dpi': 72, 'savefig.dpi': 144, 'figure.figsize': (12, 4)},
)


class IDataGroupName(str, Enum):
    prior = 'prior'
    posterior = 'posterior'


def plot_trace(mdl: BasePYMCModel, rvs: list, **kwargs) -> figure.Figure:
    """Create traceplot for passed mdl NOTE a useful kwarg is `kind` e.g.
    'trace', the default is `kind = 'rank_vlines'`"""
    kind = kwargs.pop('kind', 'rank_vlines')
    txtadd = kwargs.pop('txtadd', None)
    _ = az.plot_trace(
        mdl.idata, var_names=rvs, kind=kind, figsize=(12, 0.8 + 1.5 * len(rvs))
    )
    f = plt.gcf()
    _ = f.suptitle(
        ' - '.join(filter(None, [f'Posterior traces of {rvs}', txtadd]))
        + f'\n{mdl.mdl_id}'
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
        'NUTS Energy (Marginal vs Transitional, and E-BFMI)' + f' - `{mdl.mdl_id}`'
    )
    _ = f.tight_layout()
    return f


def facetplot_krushke(
    mdl: BasePYMCModel,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    m: int = 1,
    rvs_hack: int = 0,
    ref_vals: dict = None,
    **kwargs,
) -> figure.Figure:
    """Create Krushke-style plots using Arviz, univariate RVs, control faceting
    NOTE
    + ref_vals should look like a dict of list of dict with a key "ref_val"
        e.g. ref_vals = { 'beta_sigma' : [ {'ref_val':2} ] }
    + Optional Pass kwargs like hdi_prob = 0.5, coords = {'oid', oids}
    """
    # TODO unpack the compressed rvs from the idata
    txtadd = kwargs.pop('txtadd', None)
    transform = kwargs.pop('transform', None)
    n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    f, axs = plt.subplots(n, m, figsize=(2.6 * m, 0.8 + 1.5 * n))
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
        ' - '.join(filter(None, [f'Distribution of {rvs}', group, txtadd]))
        + f'\n{mdl.mdl_id}'
    )
    _ = f.tight_layout()
    return f


def forestplot_single(
    mdl: BasePYMCModel,
    var_names: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    **kwargs,
) -> figure.Figure:
    """Plot forestplot for list of var_names RV (optionally with factor sublevels)"""
    txtadd = kwargs.pop('txtadd', None)
    dp = kwargs.pop('dp', 2)
    plot_mn = kwargs.pop('plot_mn', True)
    transform = kwargs.pop('transform', None)
    desc = None
    kws = dict(
        colors=sns.color_palette('tab20c', n_colors=16).as_hex()[
            kwargs.pop('clr_offset', 0) :
        ][0],
        ess=False,
        combined=kwargs.pop('combined', True),
    )

    # get overall stats
    df = az.extract(mdl.idata, group=group, var_names=var_names).to_dataframe()
    if transform is not None:
        df = df.apply(transform)
    if len(var_names) == 1:
        mn = df[var_names[0]].mean(axis=0)
        qs = df[var_names[0]].quantile(q=[0.03, 0.97]).values
        desc = (
            f'Overall: $Mean =$ {mn:.{dp}f}'
            + ', $HDI_{94}$ = ['
            + ', '.join([f'{qs[v]:.{dp}f}' for v in range(2)])
            + ']'
        )
    nms = [nm for nm in df.index.names if nm not in ['chain', 'draw']]
    n = sum([len(df.index.get_level_values(nm).unique()) for nm in nms])

    f = plt.figure(figsize=(12, 1.2 + 0.3 * n))
    ax0 = f.add_subplot()
    _ = az.plot_forest(
        mdl.idata[group], var_names=var_names, **kws, transform=transform, ax=ax0
    )
    _ = ax0.set_title('')

    if plot_mn & (len(var_names) == 1):
        _ = ax0.axvline(mn, color='#ADD8E6', ls='--', lw=3, zorder=-1)
    else:
        _ = ax0.axvline(0, color='#ADD8E6', ls='--', lw=3, zorder=-1)
    _ = f.suptitle(
        '\n'.join(
            filter(
                None,
                [
                    ' - '.join(
                        filter(None, [f'Forestplot of {var_names}', group, txtadd])
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
    txtadd = kwargs.pop('txtadd', None)
    clr_offset = kwargs.pop('clr_offset', 0)
    dp = kwargs.pop('dp', 1)
    plot_med = kwargs.pop('plot_med', True)
    plot_combined = kwargs.pop('plot_combined', False)
    desc = None

    hs = [0.22 * (np.prod(data.shape[2:])) for data in datasets.values()]
    f = plt.figure(figsize=(12, 1.4 + 0.2 * sum(hs)))
    gs = gridspec.GridSpec(len(hs), 1, height_ratios=hs, figure=f)

    for i, (txt, data) in enumerate(datasets.items()):
        ax = f.add_subplot(gs[i])
        _ = az.plot_forest(
            data,
            ax=ax,
            colors=sns.color_palette('tab20c', n_colors=16).as_hex()[clr_offset:][i],
            ess=False,
            combined=plot_combined,
        )

        _ = ax.set_title(txt)
        if plot_med:
            if i == 0:
                qs = np.quantile(data, q=[0.03, 0.25, 0.5, 0.75, 0.97])
                _ = ax.axvline(qs[2], color='#ADD8E6', ls='--', lw=3, zorder=-1)
                desc = (
                    f'med {qs[2]:.{dp}f}, HDI50 ['
                    + ', '.join([f'{qs[v]:.{dp}f}' for v in [1, 3]])
                    + '], HDI94 ['
                    + ', '.join([f'{qs[v]:.{dp}f}' for v in [0, 4]])
                    + ']'
                )
            else:
                _ = ax.axvline(1, color='#ADD8E6', ls='--', lw=3, zorder=-1)

    _ = f.suptitle(
        '\n'.join(
            filter(
                None,
                [
                    ' - '.join(filter(None, ['Forestplot levels', group, txtadd])),
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
    """Create posterior pair / correlation plots using Arviz, corrrlated RVs,
    Pass-through kwargs to az.plot_pair, e.g. ref_vals
    Default to posterior, allow for override to prior
    """
    txtadd = kwargs.pop('txtadd', None)
    kind = kwargs.pop('kind', 'kde')

    pair_kws = dict(
        group=group,
        var_names=rvs,
        reference_values=ref_vals,  # deal with inconsistent naming
        divergences=True,
        marginals=True,
        kind=kind,
        kde_kwargs=dict(
            contourf_kwargs=dict(alpha=0.5, cmap='Blues'),
            contour_kwargs=dict(colors=None, cmap='Blues'),
            hdi_probs=[0.5, 0.94, 0.99],
        ),
        figsize=(2 + 1.8 * len(rvs), 2 + 1.8 * len(rvs)),
    )
    # idata[group][rvs].stack(dims=('chain', 'draw')).values.T,
    axs = az.plot_pair(mdl.idata, **pair_kws)
    corr = pd.DataFrame(
        az.sel_utils.xarray_to_ndarray(mdl.idata.get(group), var_names=rvs)[1].T
    ).corr()
    i, j = np.tril_indices(n=len(corr), k=-1)
    for ij in zip(i, j):
        axs[ij].set_title(f'rho: {corr.iloc[ij]:.2f}', fontsize=8, loc='right', pad=2)
    vh_y = dict(rotation=0, va='center', ha='right')
    vh_x = dict(rotation=40, va='top', ha='right')
    _ = [a.set_ylabel(a.get_ylabel(), **vh_y) for ax in axs for a in ax]
    _ = [a.set_xlabel(a.get_xlabel(), **vh_x) for ax in axs for a in ax]

    f = plt.gcf()
    _ = f.suptitle(
        ' - '.join(filter(None, ['Pairplot', mdl.name, group, 'selected RVs', txtadd]))
    )
    _ = f.tight_layout()
    return f


def plot_ppc(
    mdl: BasePYMCModel,
    var_names: list,
    idata: az.InferenceData = None,
    group: str = 'posterior',
    insamp: bool = True,
    ecdf: bool = True,
    flatten: list = None,
    observed_rug: bool = True,
    logx: bool = False,
    **kwargs,
) -> figure.Figure:
    """Plot In- or Out-of-Sample Prior or Posterior Retrodictive, does not
    require log-likelihood.
    NOTE:
    + use var_names to only plot e.g. yhat
    + pass through kwargs, possibly of particular use is:
        `data_pairs` = {key (in observed_data): value (in {group}_predictive)}
        although we remind that the constant_data has the real name, but once
        it's observed in a log-likelihoood the idata.observed_data will get the
        same name as the {group}_predictive, so data_pairs is not often needed
    """
    txtadd = kwargs.pop('txtadd', None)
    kind = 'kde'
    kindnm = kind.upper()
    ynm = 'density'
    loc = 'upper right'
    if ecdf:
        kind = 'cumulative'
        kindnm = 'ECDF'
        ynm = 'prop'
        loc = 'lower right'
    _idata = mdl.idata if idata is None else idata
    n = len(var_names)
    if flatten is not None:
        n = 1
        for k in var_names:
            n *= _idata['observed_data'][k].shape[-1]
    # wild hack to get the size of observed
    i = list(dict(_idata.observed_data.sizes).values())[0]
    num_pp_samples = None if i < 500 else 200
    f, axs = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, squeeze=False)
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
    _ = [ax.legend(fontsize=8, loc=loc) for ax in axs.flatten()]  # fix legend
    ls = None
    if logx:
        _ = [ax.set_xscale('log') for ax in axs.flatten()]
        ls = '(logscale)'
    _ = [ax.set(title=t, ylabel=ynm) for ax, t in zip(axs.flatten(), var_names)]
    t = f'{"In" if insamp else "Out-of"}-sample {group.title()} Retrodictive {kindnm}'
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd, ls])) + f'\n{mdl.mdl_id}')
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
    txtadd = kwargs.pop('txtadd', None)
    f, axs = plt.subplots(
        len(data_pairs), 2, figsize=(12, 3 * len(data_pairs)), squeeze=False
    )
    for i, (y, yhat) in enumerate(data_pairs.items()):
        kws = dict(y=y, y_hat=yhat)
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][0], **kwargs)
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][1], ecdf=True, **kwargs)

        _ = axs[i][0].set_title(f'Predicted {yhat} LOO-PIT')
        _ = axs[i][1].set_title(f'Predicted {yhat} LOO-PIT cumulative')

    _ = f.suptitle(
        ' - '.join(filter(None, ['In-sample LOO-PIT', txtadd])) + f'\n{mdl.mdl_id}'
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
    txtadd = kwargs.pop('txtadd', None)
    sharex = kwargs.pop('sharex', False)
    f, axs = plt.subplots(
        len(yhats),
        1,
        figsize=(12, 2.6 * len(yhats) + 0.2 * len(mdl_dict)),
        squeeze=False,
        sharex=sharex,
    )
    # mdlnms = ' vs '.join(idata_dict.keys())
    idata_dict = {f'{k}\n{v.mdl_id_fn}': v.idata for k, v in mdl_dict.items()}
    dcomp = {}
    for i, y in enumerate(yhats):
        dfcomp = az.compare(
            idata_dict, var_name=y, ic='loo', method='stacking', scale='log'
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
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])))
    _ = f.tight_layout()

    return f, dcomp


def plot_lkjcc_corr(mdl: BasePYMCModel, **kwargs) -> figure.Figure:
    """Plot lkjcc_corr model RVs
    Drop diagonals, assume coord is called lkjcc_corr
    Also see https://python.arviz.org/en/stable/user_guide/label_guide.html#custom-labellers
    """
    coords = {
        'lkjcc_corr_dim_0': xr.DataArray([0, 1], dims=['asdf']),
        'lkjcc_corr_dim_1': xr.DataArray([1, 0], dims=['asdf']),
    }

    return facetplot_krushke(
        mdl=mdl,
        txtadd='lkjcc_corr, diagonals only',
        rvs=['lkjcc_corr'],
        coords=coords,
        m=2,
        rvs_hack=0,
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
    txtadd = kwargs.pop('txtadd', None)
    kws_mn = dict(
        markerfacecolor="w", markeredgecolor="#333333", marker="d", markersize=12
    )
    kws_box = dict(kind="box", sym='', showmeans=True, whis=(3, 97), meanprops=kws_mn)
    kws_sctr = dict(s=80, color="#32CD32")

    g = sns.catplot(
        x=yhat, y=oid, data=dfhat.reset_index(), **kws_box, height=4, aspect=3
    )
    _ = g.map(sns.scatterplot, y, oid, **kws_sctr, zorder=100)
    t_io = (
        f'{"In" if insamp else "Out-of"}-sample: boxplots of posterior `{yhat}`'
        + f' with overplotted actual `{y}` values per observation'
        + f' `{oid}` (green dots) - `{mdl.name}`'
    )
    _ = g.fig.suptitle(' - '.join(filter(None, [t_io, txtadd])) + f'\n{mdl.mdl_id}')

    _ = g.tight_layout()
    return g.fig
