# Copyright 2023 Oreum Industries
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
import xarray
from matplotlib import figure, gridspec

__all__ = [
    'plot_trace',
    'facetplot_krushke',
    'pairplot_corr',
    'forestplot_single',
    'forestplot_multiple',
    'plot_dist_fns_over_x',
    'plot_dist_fns_over_x_manual_only',
    'plot_ppc_loopit',
    'plot_energy',
]

sns.set(
    style='darkgrid',
    palette='muted',
    context='notebook',
    rc={'savefig.dpi': 300, 'figure.figsize': (12, 6)},
)


class IDataGroupName(str, Enum):
    prior = 'prior'
    posterior = 'posterior'


def plot_trace(
    idata: az.data.inference_data.InferenceData, rvs: list, **kwargs
) -> figure.Figure:
    """Create traceplot for passed idata"""
    kind = kwargs.pop('kind', 'rank_vlines')
    mdlname = kwargs.pop('mdlname', None)
    txtadd = kwargs.pop('txtadd', None)
    _ = az.plot_trace(idata, var_names=rvs, kind=kind, figsize=(12, 1.8 * len(rvs)))
    f = plt.gcf()
    _ = f.suptitle(
        ' - '.join(
            filter(None, ['Traceplot', mdlname, 'posterior', ', '.join(rvs), txtadd])
        )
    )
    _ = f.tight_layout()
    return f


def facetplot_krushke(
    idata: az.data.inference_data.InferenceData,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    m: int = 1,
    rvs_hack: int = 0,
    ref_vals: dict = None,
    **kwargs,
) -> figure.Figure:
    """Create Krushke-style plots using Arviz, univariate RVs, control faceting"""
    # TODO unpack the compressed rvs from the idata
    mdlname = kwargs.pop('mdlname', None)
    txtadd = kwargs.pop('txtadd', None)
    n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    f, axs = plt.subplots(n, m, figsize=(4 + m * 2.4, 2 * n))
    _ = az.plot_posterior(
        idata, group=group, ax=axs, var_names=rvs, ref_val=ref_vals, **kwargs
    )
    s = 's' if len(rvs) > 1 else ''
    _ = f.suptitle(
        ' - '.join(filter(None, [f'Distribution plot{s}', mdlname, group, txtadd]))
    )
    _ = f.tight_layout()
    return f


def forestplot_single(
    data: xarray.core.dataarray.DataArray,
    group: IDataGroupName = IDataGroupName.posterior.value,
    **kwargs,
) -> figure.Figure:
    """Plot forestplot for a single RV (optionally with factor sublevels)"""
    mdlname = kwargs.pop('mdlname', None)
    txtadd = kwargs.pop('txtadd', None)
    clr_offset = kwargs.pop('clr_offset', 0)
    dp = kwargs.pop('dp', 1)
    plot_med = kwargs.pop('plot_med', True)
    plot_combined = kwargs.pop('plot_combined', False)
    kws = dict(
        colors=sns.color_palette('tab20c', n_colors=16).as_hex()[clr_offset:][0],
        ess=False,
        combined=plot_combined,
    )

    qs = np.quantile(data, q=[0.03, 0.25, 0.5, 0.75, 0.97])
    desc = (
        f'med {qs[2]:.{dp}f}, HDI50 ['
        + ', '.join([f'{qs[v]:.{dp}f}' for v in [1, 3]])
        + '], HDI94 ['
        + ', '.join([f'{qs[v]:.{dp}f}' for v in [0, 4]])
        + ']'
    )

    f = plt.figure(figsize=(12, 2 + 0.15 * (np.prod(data.shape[2:]))))
    ax0 = f.add_subplot()
    _ = az.plot_forest(data, ax=ax0, **kws)
    if plot_med:
        _ = ax0.axvline(qs[2], color='#ADD8E6', ls='--', lw=3, zorder=-1)
    _ = f.suptitle(
        ' - '.join(filter(None, ['Forestplot levels', mdlname, group, txtadd, desc]))
    )
    _ = f.tight_layout()
    return f


def forestplot_multiple(
    datasets: dict[str, xarray.core.dataarray.DataArray],
    group: IDataGroupName = IDataGroupName.posterior.value,
    **kwargs,
) -> figure.Figure:
    """Plot set of forestplots for related datasets RVs
    Useful for a linear model of RVs, where each RV can have sublevel factors
    TODO This makes a few too many assumptions, will improve in future
    """
    mdlname = kwargs.pop('mdlname', None)
    txtadd = kwargs.pop('txtadd', None)
    clr_offset = kwargs.pop('clr_offset', 0)
    dp = kwargs.pop('dp', 1)
    plot_med = kwargs.pop('plot_med', True)
    plot_combined = kwargs.pop('plot_combined', False)
    desc = ''

    hs = [0.22 * (np.prod(data.shape[2:])) for data in datasets.values()]
    f = plt.figure(figsize=(12, 2 + sum(hs)))
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
        ' - '.join(filter(None, ['Forestplot levels', mdlname, group, txtadd]))
        + f'\n{desc}'
    )
    _ = f.tight_layout()
    return f


def pairplot_corr(
    idata: az.data.inference_data.InferenceData,
    rvs: list[str],
    group: IDataGroupName = IDataGroupName.posterior.value,
    colnames=list[str],
    ref_vals: dict = None,
    **kwargs,
) -> figure.Figure:
    """Create posterior pair / correlation plots using Arviz, corrrlated RVs,
    Pass-through kwargs to az.plot_pair, e.g. ref_vals
    """
    mdlname = kwargs.pop('mdlname', None)
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
    axs = az.plot_pair(idata, **pair_kws)
    corr = pd.DataFrame(
        az.sel_utils.xarray_to_ndarray(idata.posterior, var_names=rvs)[1].T
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
        ' - '.join(filter(None, ['Pairplot', mdlname, group, 'selected RVs', txtadd]))
    )
    _ = f.tight_layout()
    return f


def plot_dist_fns_over_x(
    dfpdf: pd.DataFrame, dfcdf: pd.DataFrame, dfinvcdf: pd.DataFrame, **kwargs
) -> figure.Figure:
    """Convenience to plot results of calc_dist_fns_over_x()"""

    name = kwargs.get('name', 'unknown_dist')
    islog = kwargs.get('log', False)
    lg = 'log ' if islog else ''
    f, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=False, sharey=False)
    f.suptitle(f'Comparisons manual vs scipy for {lg}{name}', y=1.02)
    n = len(dfpdf)
    is_close = {
        k: np.sum(np.isclose(v['manual'], v['scipy'], equal_nan=True))
        for k, v in zip(['p', 'c', 'i'], [dfpdf, dfcdf, dfinvcdf])
    }

    dfm = dfpdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax0 = sns.lineplot(
        x='x', y='density', hue='method', style='method', data=dfm, ax=axs[0]
    )
    _ = ax0.set_title(f"{lg}PDF: match {is_close['p'] / n :.1%}")

    dfm = dfcdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax1 = sns.lineplot(
        x='x', y='density', hue='method', style='method', data=dfm, ax=axs[1]
    )
    _ = ax1.set_title(f"{lg}CDF: match {is_close['c'] / n :.1%}")
    if not islog:
        ylimmin = ax1.get_ylim()[0]
        _ = ax1.set(ylim=(min(0, ylimmin), None))

    dfm = dfinvcdf.reset_index().melt(id_vars='u', value_name='x', var_name='method')
    ax2 = sns.lineplot(x='u', y='x', hue='method', style='method', data=dfm, ax=axs[2])
    _ = ax2.set_title(f"{lg}InvCDF: match {is_close['i'] / n :.1%}")
    # f.tight_layout()
    return f


def plot_dist_fns_over_x_manual_only(
    dfpdf: pd.DataFrame, dfcdf: pd.DataFrame, dfinvcdf: pd.DataFrame, **kwargs
) -> figure.Figure:
    """Convenience to plot results of calc_dist_fns_over_x_manual_only()"""

    name = kwargs.get('name', 'unknown_dist')
    islog = kwargs.get('log', False)
    lg = 'log ' if islog else ''
    f, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=False, sharey=False)
    f.suptitle(f'Display manual calcs for {lg}{name}', y=1.02)
    # n = len(dfpdf)

    dfm = dfpdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax0 = sns.lineplot(
        x='x', y='density', hue='method', style='method', data=dfm, ax=axs[0]
    )
    _ = ax0.set_title(f"{lg}PDF")

    dfm = dfcdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax1 = sns.lineplot(
        x='x', y='density', hue='method', style='method', data=dfm, ax=axs[1]
    )
    _ = ax1.set_title(f"{lg}CDF")
    if not islog:
        _ = ax1.set(ylim=(0, None))

    dfm = dfinvcdf.reset_index().melt(id_vars='u', value_name='x', var_name='method')
    ax2 = sns.lineplot(x='u', y='x', hue='method', style='method', data=dfm, ax=axs[2])
    _ = ax2.set_title(f"{lg}InvCDF")
    return f


def plot_ppc_loopit(
    idata: az.data.inference_data.InferenceData,
    kind: str = 'kde',
    tgts: dict = {'y': 'yhat'},
    **kwargs,
) -> figure.Figure:
    """Calc and Plot PPC & LOO-PIT after run `mdl.sample_posterior_predictive()`
    also see
    https://oriolabrilpla.cat/python/arviz/pymc/2019/07/31/loo-pit-tutorial.html
    """
    mdlname = kwargs.pop('mdlname', None)
    txtadd = kwargs.pop('txtadd', None)
    if len(tgts) > 1:
        raise AttributeError(
            'NOTE: live issue in Arviz, if more than one tgt '
            + 'it will plot them all its own way'
        )

    f = plt.figure(figsize=(12, 6 * len(tgts)))
    gs = gridspec.GridSpec(
        2 * len(tgts),
        2,
        height_ratios=[1.5, 1] * len(tgts),
        width_ratios=[1, 1],
        figure=f,
    )
    var_names = kwargs.pop('var_names', None)

    # TODO: live issue this selection doesnt work in Arviz,
    # it just plots every tgt. so this loop is a placeholder that does work
    # if there's only a single tgt
    for i, (tgt, tgt_hat) in enumerate(tgts.items()):
        ax0 = f.add_subplot(gs[0 + 4 * i, :])
        ax1 = f.add_subplot(gs[2 + 4 * i])
        ax2 = f.add_subplot(gs[3 + 4 * i], sharex=ax1)
        _ = az.plot_ppc(
            idata,
            kind=kind,
            flatten=None,
            ax=ax0,
            group='posterior',
            data_pairs={tgt: tgt_hat},
            var_names=var_names,
            **kwargs,
        )
        # using y=tgt_hat below. seems wrong, possibly a bug in arviz
        _ = az.plot_loo_pit(idata, y=tgt_hat, ax=ax1, **kwargs)
        _ = az.plot_loo_pit(idata, y=tgt_hat, ecdf=True, ax=ax2, **kwargs)

        _ = ax0.set_title(f'PPC Predicted {tgt_hat} vs Observed {tgt}')
        _ = ax1.set_title(f'Predicted {tgt_hat} LOO-PIT')
        _ = ax2.set_title(f'Predicted {tgt_hat} LOO-PIT cumulative')

    _ = f.suptitle(' - '.join(filter(None, ['In-sample PPC', mdlname, txtadd])))
    _ = f.tight_layout()
    return f


def plot_energy(idata: az.data.inference_data.InferenceData, **kwargs) -> figure.Figure:
    """Simple wrapper around energy plot to provide a simpler interface"""
    _ = az.plot_energy(idata, figsize=(12, 2))
    f = plt.gcf()
    _ = f.tight_layout()
    return f


def facetplot_df_dist(
    df: pd.DataFrame, rvs: list, m: int = 3, rvs_hack: int = 0, **kwargs
) -> figure.Figure:
    """Control facet positioning of Arviz Krushke style plots, data in df
    Pass-through kwargs to az.plot_posterior, e.g. ref_val
    """
    raise DeprecationWarning('No longer needed, will deprecate in future versions')
    # n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    # sharex = kwargs.get('sharex', False)
    # f, axs = plt.subplots(n, m, figsize=(4 + m * 2.4, 2 * n), sharex=sharex)
    # ref_val = kwargs.get('ref_val', [None for i in range(len(df))])

    # for i, ft in enumerate(df.columns):
    #     axarr = az.plot_posterior(
    #         df[ft].values, ax=axs.flatten()[i], ref_val=ref_val[i]
    #     )
    #     axarr.set_title(ft)
    # title = kwargs.get('title', '')
    # _ = f.suptitle(f'{title} {rvs}', y=0.96 + n * 0.005)
    # _ = f.tight_layout()
    # return f
