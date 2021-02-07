# model_tools.py
# copyright 2021 Oreum OÜ
import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

def model_desc(fml):
    """ Convenience: return patsy modeldesc
        NOTE minor issue: `describe()` doesn't return the `1 +` (intercept)
        term correctly.
    """
    t = pt.ModelDesc.from_formula(fml).describe()[2:]
    return f'model desc: \n1 + {t}\n'


def read_azid(dir_traces=[], fn='azid'):
    """Convenience: read azid from file"""
    return az.from_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))       


def create_azid(model, save=False, dir_traces=[], fn='azid',
                prior=None, trace=None, ppc=None):
    """Convenience: create azid structure"""
    
    azid = az.from_pymc3(model=model, prior=prior, trace=trace,
                        posterior_predictive=ppc)
    if save:
        azid.to_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))
        del azid
        azid = az.from_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))       
    return azid


def facetplot_azid_dist(azid, rvs, rvs_hack_extra=0, group='posterior', ref_vals=None):
    """Convenience: plot Krushke style in facets """
    # TODO unpack the compressed rvs from the azid
    
    # m, n = 2, (len(rvs) // 2) + (len(rvs) % 2)
    m, n = 2, ((len(rvs)+rvs_hack_extra) // 2) + ((len(rvs)+rvs_hack_extra) % 2)
    f, ax1d = plt.subplots(n, m, figsize=(m*6, 2.2*n))
    kw = {}
    if ref_vals is not None:
        kw['ref_vals'] = ref_vals
    _ = az.plot_posterior(azid, group=group, ax=ax1d, var_names=rvs, **kw)
    f.suptitle(group, y=0.9 + n*0.005)
    f.tight_layout()

    
def extract_yobs_yhat(azid, obs='y', pred='yhat'):
    """Convenience: extract y_obs, y_hat from azid
        get yhat in the shape (nsamples, nobs)
    """
    nsamp = np.product(azid.posterior_predictive[pred].shape[:-1])    
    yobs = azid.constant_data[obs].values                            # (nobs,)
    yhat = azid.posterior_predictive[pred].values.reshape(nsamp, -1) # (nsamp, nobs)
    
    return yobs, yhat

    
def compute_mse(y, yhat):
    r""" Convenience: Calculate MSE using all samples
        shape (nsamples, nobservations)
   
    Mean-Squared Error of prediction vs observed 
    $$\frac{1}{n}\sum_{i=1}^{i=n}(\hat{y}_{i}-y_{i})^{2}$$

    \begin{align}
    \text{Method A, compress to mean of samples: } \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{i}) - y_{i})^{2} \\
    \text{Method B, use all samples then mean: } \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{ij} - y_{i})^{2})
    \end{align}

    WARNING: 
        the 'samples' approach squares outliers pushing farther from the mean
        # se_samples = np.power(yhat - yobs, 2)      # (nsamp, nobs)
        # mse_samples = np.mean(se_samples, axis=1)  # (nsamp,)
        i.e.
        take mean across samples then square the differences is usually smaller than 
        square the differences each sample and preserve samples
        https://en.wikipedia.org/wiki/Generalized_mean
        
        I can only think to calc summary stats and then calc MSE for them
    """  
    # collapse samples to mean then calc error
    se = np.power(yhat.mean(axis=0) - y, 2) # (nobs)
    mse = np.mean(se, axis=0)                  # 1

    # collapse samples to a range of summary stats then calc error
    smry = np.arange(0, 101, 2)
    se_pct = np.power(np.percentile(yhat, smry, axis=0) - y, 2) # (len(smry), nobs)
    mse_pct = np.mean(se_pct, axis=1)                              # len(smry)
    
    s_mse_pct = pd.Series(mse_pct, index=smry, name='mse')
    s_mse_pct.index.rename ('pct', inplace=True) 
    return mse, s_mse_pct


def compute_rmse(y, yhat):
    """ Convenience: Calculate RMSE """
    mse, s_mse_pct = compute_mse(y, yhat)
    s_rmse_pct = s_mse_pct.map(np.sqrt)
    s_rmse_pct._set_name('rmse', inplace=True)
       
    return np.sqrt(mse), s_rmse_pct


def compute_r2(y, yhat):
    """ Calculate R2, 
        return mean r2 and via summary stats of yhat
        NOTE: shape (nsamples, nobservations)
        $$R^{2} = 1 - \frac{\sum e_{model}^{2}}{\sum e_{mean}^{2}}$$
    """
    sse_mean = np.sum((y - y.mean(axis=0))**2)

    # Collapse samples to mean then calc error
    sse_model_mean = np.sum((y - yhat.mean(axis=0))**2)
    r2_mean = 1 - (sse_model_mean / sse_mean)
    
    # calc summary stats of yhat
    smry = np.arange(0, 101, 5)
    sse_model =  np.sum((y - np.percentile(yhat, smry, axis=0))**2, axis=1) # (len(smry), nobs)       
    r2_pct = pd.Series(1 - (sse_model / sse_mean), index=smry, name='r2')
    r2_pct.index.rename ('pct', inplace=True) 
    
    return r2_mean, r2_pct 

    
def ppc_coverage(y, yhat, crs=np.arange(0, 1.01, .1)):
    """ Calc the proportion of coverage from full yhat ppc 
        shape (nsamples, nobservations)
    """
    lower_bounds = np.percentile(yhat, 50 - (50 * crs), axis=0)
    upper_bounds = np.percentile(yhat, 50 + (50 * crs), axis=0)
   
    coverage = []
    for i, cr in enumerate(crs):
        coverage.append((cr, np.sum((y >= lower_bounds[i]) * (y <= upper_bounds[i])) / len(y)))

    return pd.DataFrame(coverage, columns=['cr', 'coverage'])
