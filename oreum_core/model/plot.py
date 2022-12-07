# model.plot.py
# copyright 2022 Oreum Industries
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure, gridspec

__all__ = [
    'plot_ppc_loopit',
    'facetplot_idata_dist',
    'facetplot_df_dist',
    'plot_dist_fns_over_x',
    'plot_dist_fns_over_x_manual_only',
]


def plot_ppc_loopit(
    idata: az.data.inference_data.InferenceData,
    kind: str = 'kde',
    tgts: dict = {'y': 'yhat'},
    **kwargs,
) -> figure.Figure:
    """Calc and Plot PPC & LOO-PIT after run `mdl.sample_posterior_predictive()`
    also see
    https://oriolabrilpla.cat/python/arviz/pymc3/2019/07/31/loo-pit-tutorial.html
    """
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

    _ = f.suptitle('In-sample PPC Evaluation')
    _ = f.tight_layout()
    return f


def facetplot_idata_dist(
    idata: az.data.inference_data.InferenceData,
    rvs: list,
    group: str = 'posterior',
    m: int = 3,
    rvs_hack: int = 0,
    **kwargs,
) -> figure.Figure:
    """Control facet positioning of Arviz Krushke style plots, data in idata
    Pass-through kwargs to az.plot_posterior, e.g. ref_val
    """
    # TODO unpack the compressed rvs from the idata
    n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    f, axs = plt.subplots(n, m, figsize=(4 + m * 2.4, 2 * n))
    _ = az.plot_posterior(idata, group=group, ax=axs, var_names=rvs, **kwargs)
    _ = f.suptitle(f'{group} {rvs}', y=0.96 + n * 0.005)
    _ = f.tight_layout()
    return f


def facetplot_df_dist(
    df: pd.DataFrame, rvs: list, m: int = 3, rvs_hack: int = 0, **kwargs
) -> figure.Figure:
    """Control facet positioning of Arviz Krushke style plots, data in df
    Pass-through kwargs to az.plot_posterior, e.g. ref_val
    """
    n = 1 + ((len(rvs) + rvs_hack - m) // m) + ((len(rvs) + rvs_hack - m) % m)
    sharex = kwargs.get('sharex', False)
    f, axs = plt.subplots(n, m, figsize=(4 + m * 2.4, 2 * n), sharex=sharex)
    ref_val = kwargs.get('ref_val', [None for i in range(len(df))])

    for i, ft in enumerate(df.columns):
        axarr = az.plot_posterior(
            df[ft].values, ax=axs.flatten()[i], ref_val=ref_val[i]
        )
        axarr.set_title(ft)
    title = kwargs.get('title', '')
    _ = f.suptitle(f'{title} {rvs}', y=0.96 + n * 0.005)
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
