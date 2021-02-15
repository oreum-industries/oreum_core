# model.plot.py
# copyright 2021 Oreum OÜ
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns


def facetplot_azid_dist(azid, rvs, rvs_hack_extra=0, group='posterior', **kwargs):
    """Control facet positioning of Arviz Krushke style plots, data in azid
        Pass-through kwargs to az.plot_posterior, e.g. ref_val
    """
    # TODO unpack the compressed rvs from the azid
    
    m, n = 2, ((len(rvs)+rvs_hack_extra) // 2) + ((len(rvs)+rvs_hack_extra) % 2)
    f, axs = plt.subplots(n, m, figsize=(m*6, 2.2*n))
    _ = az.plot_posterior(azid, group=group, ax=axs, var_names=rvs, **kwargs)
    f.suptitle(group, y=0.9 + n*0.005)
    f.tight_layout()


def facetplot_df_dist(df, rvs, rvs_hack_extra=0, **kwargs):
    """Control facet positioning of Arviz Krushke style plots, data in df
        Pass-through kwargs to az.plot_posterior, e.g. ref_val
    """  
    m, n = 2, ((len(rvs)+rvs_hack_extra) // 2) + ((len(rvs)+rvs_hack_extra) % 2)
    f, axs = plt.subplots(n, m, figsize=(m*6, 2.2*n))
    ref_val = kwargs.get('ref_val', None)
    for i, ft in enumerate(df.columns):
        axarr = az.plot_posterior(df[ft].values, ax=axs.flatten()[i], 
                                ref_val=ref_val[i])
        axarr.set_title(ft) 
    f.suptitle(rvs, y=0.9 + n*0.005)
    f.tight_layout()


def plot_dist_fns_over_x(dfpdf, dfcdf, dfinvcdf, **kwargs):
    """Convenience to plot results of calc_dist_fns_over_x() """

    name = kwargs.get('name', 'unknown_dist')
    islog = kwargs.get('log', False)
    l = 'log ' if islog else ''
    f, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=False, sharey=False)
    f.suptitle(f'Comparisons manual vs scipy for {l}{name}', y=1.02)
    n = len(dfpdf)
    is_close = {k: np.sum(np.isclose(v['manual'], v['scipy'], equal_nan=True))
        for k, v in zip(['p','c','i'], [dfpdf, dfcdf, dfinvcdf])}

    dfm = dfpdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax0 = sns.lineplot(x='x', y='density', hue='method', style='method',data=dfm, ax=axs[0])
    _ = ax0.set_title(f"{l}PDF: match {is_close['p'] / n :.1%}")

    dfm = dfcdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax1 = sns.lineplot(x='x', y='density', hue='method', style='method', data=dfm, ax=axs[1])
    _ = ax1.set_title(f"{l}CDF: match {is_close['c'] / n :.1%}")

    dfm = dfinvcdf.reset_index().melt(id_vars='u', value_name='density', var_name='method')
    ax2 = sns.lineplot(x='u', y='density', hue='method', style='method', data=dfm, ax=axs[2])
    _ = ax2.set_title(f"{l}InvCDF: match {is_close['i'] / n :.1%}")