# eda.calc.py
# copyright 2021 Oreum OÜ
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

# TODO see issue #2
def fit_and_plot_fn(obs, tail_kind='right', title_insert=None):
    """
    Fit dists to 1d array of `obs`, report RMSE and plot the fits
    # see https://stackoverflow.com/a/37616966 
    """

    import warnings
    # warnings.filterwarnings("error") # handle RuntimeWarning as error so can catch
    # warnings.simplefilter(action='ignore', category='RuntimeWarning')

    dists_discrete = {'poisson': stats.poisson}

    dists_cont_right_tail = {'expon': stats.expon,
                            'invgamma':stats.invgamma, 
                            'gamma':stats.gamma, 
                            'lognorm': stats.lognorm,
                            'invgauss': stats.invgauss,
                            'halfnorm': stats.halfnorm,
                            'halfcauchy': stats.halfcauchy,
                            'gumbel': stats.gumbel_r,
                            'invweibull': stats.invweibull}
    # NOTE: not quite true since gumbel and invweibull can go neg

    dists_cont_centered = {'norm': stats.norm,
                            'cauchy': stats.cauchy}
   
    if tail_kind not in set(['right', 'both']):
        raise ValueError("tail_kind must be in {'right', 'both'}")

    obs_is_discrete = sum(obs == (obs // 1)) == len(obs)
    nbins = 100
    dist_kind = 'Continuous'
    params = {}
    f, ax1d = plt.subplots(1, 1, figsize=(16, 5))
    
    if obs_is_discrete:
        dist_kind = 'Discrete'
        nbins = np.int(obs.max())
        obs_count, bin_edges = np.histogram(obs, bins=nbins, density=False)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        bin_centers_int = np.round(bin_centers)
        dists = dists_discrete

        ax0 = sns.histplot(x=obs, bins=nbins, stat='count', kde=False, label='data', ax=ax1d)

        # TODO fix this hack: only works for poisson right now
        for i, (d, dist) in enumerate(dists.items()):
            mle_pois = obs.mean()
            params[d] = dict(shape=mle_pois, loc=0, scale=None)
            pmf = stats.poisson.pmf(k=bin_centers_int, mu=mle_pois) * len(obs)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_count - pmf, 2.0)) / len(obs))
            ax1 = sns.lineplot(x=bin_centers_int, y=pmf, label=f'{d}: {rmse:.2g}', lw=1, ax=ax1d)

    else:
        obs_density, bin_edges = np.histogram(obs, bins=nbins, density=True)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        
        if tail_kind == 'both':
            dists = dists_cont_centered
        elif tail_kind == 'right':
            dists = dists_cont_right_tail
     
        ax0 = sns.histplot(x=obs, bins=nbins, stat='density', kde=False, label='data', ax=ax1d)

        for i, (d, dist) in enumerate(dists.items()):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                ps = dist.fit(obs, floc=0)   # can throw RuntimeWarnings which we will ignore
            shape, loc, scale = ps[:-2], ps[-2], ps[-1]
            params[d] = dict(shape=shape, loc=loc, scale=scale)
            pdf = dist.pdf(bin_centers, loc=loc, scale=scale, *shape)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_density - pdf, 2.0)) / len(obs))
            ax1 = sns.lineplot(x=bin_centers, y=pdf, label=f'{d}: {rmse:.2g}', lw=1, ax=ax1d)

    title = (f'{dist_kind} function approximations to `{title_insert}`')
    _ = f.suptitle(title, y=1)
    _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
    return f, params


def get_gini(r, n):
    """ For array r, return estimate of gini co-efficient over n
        g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))
