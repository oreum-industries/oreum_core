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
def fit_fn(obs, dist_names=['invgamma'], title_insert=None, plot=True):
    """
    Fit `dists` to 1d array of `observations`, report MSE and plot the fits
    # see https://stackoverflow.com/a/37616966 
    """

    import warnings
    # warnings.filterwarnings("error") # handle RuntimeWarning as error so can catch
    # warnings.simplefilter(action='ignore', category='RuntimeWarning')

    dists_options = {'invgamma':stats.invgamma, 
                     'gamma':stats.gamma, 
                     'lognorm': stats.lognorm,
                     'gumbel': stats.gumbel_r,
                     'invgauss': stats.invgauss,
                     'invweibull': stats.invweibull,
                     'expon': stats.expon,
                     'norm': stats.norm,
                     'halfnorm': stats.halfnorm,
                     'cauchy': stats.cauchy,
                     'halfcauchy': stats.halfcauchy}
                     #TODO add other scipy stats fns
                    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html#scipy.stats.invgauss
    dists = {d: dists_options.get(d) for d in dist_names}
   
    nbins = 100
    density, bin_edges = np.histogram(obs, bins=nbins, density=True)
    bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
    params = {}
    if plot:
        f, ax1d = plt.subplots(1, 1, figsize=(16, 5))
        ax = sns.histplot(x=obs, bins=nbins, stat='density', kde=False, label='data', ax=ax1d)

    for i, (d, dist) in enumerate(dists.items()):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            ps = dist.fit(obs, floc=0)   # can throw RuntimeWarnings which we will ignore
        shape, loc, scale = ps[:-2], ps[-2], ps[-1]
        params[d] = dict(shape=shape, loc=loc, scale=scale)

        pdf = dist.pdf(bin_edges, loc=loc, scale=scale, *shape)
        mse = np.sum(np.power(density - pdf, 2.0)) / len(density)
        rmse = np.sqrt(mse)
        
        if plot:
            ax = sns.lineplot(x=bin_edges, y=pdf, label=f'{d}: {rmse:.2g}', lw=1, ax=ax1d)
    if plot:
        title = (f'Function approximations to `{title_insert}`')
        _ = f.suptitle(title, y=1)
        _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
        return f, params
    else:
        return params

def get_gini(r, n):
    """ For array r, return estimate of gini co-efficient over n
        g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))
