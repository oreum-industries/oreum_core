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

from oreum_core.model import BasePYMCModel

__all__ = [
    'plot_trace',
    'plot_energy',
    'facetplot_krushke',
    'pairplot_corr',
    'forestplot_single',
    'forestplot_multiple',
    'plot_ppc',
    'plot_loopit',
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


def plot_trace(mdl: BasePYMCModel, rvs: list, **kwargs) -> figure.Figure:
    """Create traceplot for passed mdl"""
    kind = kwargs.pop('kind', 'rank_vlines')
    txtadd = kwargs.pop('txtadd', None)
    _ = az.plot_trace(mdl.idata, var_names=rvs, kind=kind, figsize=(12, 1.8 * len(rvs)))
    f = plt.gcf()
    _ = f.suptitle(
        ' - '.join(
            filter(None, ['Traceplot', mdl.name, 'posterior', ', '.join(rvs), txtadd])
        )
    )
    _ = f.tight_layout()
    return f


def plot_energy(mdl: BasePYMCModel) -> figure.Figure:
    """Simple wrapper around energy plot to provide a simpler interface"""
    _ = az.plot_energy(mdl.idata, figsize=(12, 2))
    f = plt.gcf()
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
    NOTE can pass kwargs like hdi_prob = 0.5
    """
    # TODO unpack the compressed rvs from the idata
    txtadd = kwargs.pop('txtadd', None)
    n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    f, axs = plt.subplots(n, m, figsize=(4 + m * 2.4, 2 * n))
    _ = az.plot_posterior(
        mdl.idata, group=group, ax=axs, var_names=rvs, ref_val=ref_vals, **kwargs
    )
    s = 's' if len(rvs) > 1 else ''
    _ = f.suptitle(
        ' - '.join(filter(None, [f'Distribution plot{s}', mdl.name, group, txtadd]))
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
    idata: az.InferenceData = None,
    group: str = 'posterior',
    insamp: bool = True,
    ecdf: bool = True,
    data_pairs: dict = None,
    **kwargs,
) -> figure.Figure:
    """Plot In- or Out-of-Sample Prior or Posterior predictive ECDF,
    does not require log-likelihood.
    """
    txtadd = kwargs.pop('txtadd', None)
    kind = 'cumulative' if ecdf else 'kde'
    _idata = mdl.idata if idata is None else idata
    f, axs = plt.subplots(len(data_pairs), 1, figsize=(12, 3 * len(data_pairs)))
    _ = az.plot_ppc(_idata, group=group, kind=kind, ax=axs, data_pairs=data_pairs)
    t = f'{"In" if insamp else "Out-of"}-sample {group.title()} Predictive'
    _ = f.suptitle(' - '.join(filter(None, [t, mdl.name, txtadd])))
    _ = f.tight_layout()
    return f


def plot_loopit(mdl: BasePYMCModel, data_pairs: dict = None, **kwargs) -> figure.Figure:
    """Calc and plot LOO-PIT after run `mdl.sample_posterior_predictive()`"""
    txtadd = kwargs.pop('txtadd', None)
    f, axs = plt.subplots(len(data_pairs), 2, figsize=(12, 3 * len(data_pairs)))
    for i, (_, hat) in enumerate(data_pairs.items()):
        kws = dict(y=hat, y_hat=hat)  # NOTE named badly
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][0], **kwargs)
        _ = az.plot_loo_pit(mdl.idata, **kws, ax=axs[i][1], ecdf=True, **kwargs)

        _ = axs[i][0].set_title(f'Predicted {hat} LOO-PIT')
        _ = axs[i][1].set_title(f'Predicted {hat} LOO-PIT cumulative')

    _ = f.suptitle(' - '.join(filter(None, ['In-sample LOO-PIT', mdl.name, txtadd])))
    _ = f.tight_layout()
    return f
