# model.calc.py
# copyright 2021 Oreum OÜ
import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

    
def calc_mse(y, yhat):
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


def calc_rmse(y, yhat):
    """ Convenience: Calculate RMSE """
    mse, s_mse_pct = compute_mse(y, yhat)
    s_rmse_pct = s_mse_pct.map(np.sqrt)
    s_rmse_pct._set_name('rmse', inplace=True)
       
    return np.sqrt(mse), s_rmse_pct


def calc_r2(y, yhat):
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

    
def calc_ppc_coverage(y, yhat, crs=np.arange(0, 1.01, .1)):
    """ Calc the proportion of coverage from full yhat ppc 
        shape (nsamples, nobservations)
    """
    lower_bounds = np.percentile(yhat, 50 - (50 * crs), axis=0)
    upper_bounds = np.percentile(yhat, 50 + (50 * crs), axis=0)
   
    coverage = []
    for i, cr in enumerate(crs):
        coverage.append((cr, np.sum((y >= lower_bounds[i]) * (y <= upper_bounds[i])) / len(y)))

    return pd.DataFrame(coverage, columns=['cr', 'coverage'])


# TODO fix this at source
# Minor edit to a math fn to prevent annoying deprecation warnings
# Jon Sedar 2020-03-31
# Users/jon/anaconda/envs/instechex/lib/python3.6/site-packages/theano/tensor/subtensor.py:2339: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
#   out[0][inputs[2:]] = inputs[1]

def expand_packed_triangular(n, packed, lower=True, diagonal_only=False):
    R"""Convert a packed triangular matrix into a two dimensional array.
    Triangular matrices can be stored with better space efficiancy by
    storing the non-zero values in a one-dimensional array. We number
    the elements by row like this (for lower or upper triangular matrices):
        [[0 - - -]     [[0 1 2 3]
         [1 2 - -]      [- 4 5 6]
         [3 4 5 -]      [- - 7 8]
         [6 7 8 9]]     [- - - 9]
    Parameters
    ----------
    n: int
        The number of rows of the triangular matrix.
    packed: theano.vector
        The matrix in packed format.
    lower: bool, default=True
        If true, assume that the matrix is lower triangular.
    diagonal_only: bool
        If true, return only the diagonal of the matrix.
    """
    if packed.ndim != 1:
        raise ValueError('Packed triagular is not one dimensional.')
    if not isinstance(n, int):
        raise TypeError('n must be an integer')

    if diagonal_only and lower:
        diag_idxs = np.arange(1, n + 1).cumsum() - 1
        return packed[diag_idxs]
    elif diagonal_only and not lower:
        diag_idxs = np.arange(2, n + 2)[::-1].cumsum() - n - 1
        return packed[diag_idxs]
    elif lower:
        out = tt.zeros((n, n), dtype=theano.config.floatX)
        idxs = np.tril_indices(n)
        return tt.set_subtensor(out[idxs], packed)
    elif not lower:
        out = tt.zeros((n, n), dtype=theano.config.floatX)
        idxs = np.triu_indices(n)
        return tt.set_subtensor(out[idxs], packed)


def calc_dist_fns_over_x(fd_scipy, d_manual, params, **kwargs):
    """ Test my manual model PDF, CDF, InvCDF vs 
        a scipy fixed dist over range x 
    """
    logdist = kwargs.get('logdist', False)
    upper = kwargs.get('upper', 1)
    lower = kwargs.get('lower', 0)
    nsteps = kwargs.get('nsteps', 200)
    x = np.linspace(lower, upper, nsteps)
    u = np.linspace(0, 1, nsteps)

    if logdist:
        dfpdf = pd.DataFrame({'manual': d_manual.logpdf(x, **params),
                              'scipy': fd_scipy.logpdf(x), 
                              'x': x}).set_index('x')
        dfcdf = pd.DataFrame({'manual': d_manual.logcdf(x, **params),
                              'scipy': fd_scipy.logcdf(x), 
                              'x': x}).set_index('x')
        dfinvcdf = pd.DataFrame({'manual': d_manual.loginvcdf(u, **params),
                                 'scipy': np.log(fd_scipy.ppf(u)), 
                                 'u': u}).set_index('u')
    else:
        dfpdf = pd.DataFrame({'manual': d_manual.pdf(x, **params),
                             'scipy': fd_scipy.pdf(x), 
                             'x': x}).set_index('x')
        dfcdf = pd.DataFrame({'manual': d_manual.cdf(x, **params),
                              'scipy': fd_scipy.cdf(x), 
                              'x': x}).set_index('x')
        dfinvcdf = pd.DataFrame({'manual': d_manual.invcdf(u, **params),
                                 'scipy': fd_scipy.ppf(u), 
                                 'u': u}).set_index('u')
            
    return dfpdf, dfcdf, dfinvcdf


def calc_dist_fns_over_x_manual_only(d_manual, params, **kwargs):
    """ Test my manual model PDF, CDF, InvCDF over range x 
    """
    logdist = kwargs.get('logdist', False)
    upper = kwargs.get('upper', 1)
    lower = kwargs.get('lower', 0)
    nsteps = kwargs.get('nsteps', 200)
    x = np.linspace(lower, upper, nsteps)
    u = np.linspace(0, 1, nsteps)

    if logdist:
        dfpdf = pd.DataFrame({'manual': d_manual.logpdf(x, **params),
                              'x': x}).set_index('x')
        dfcdf = pd.DataFrame({'manual': d_manual.logcdf(x, **params),
                              'x': x}).set_index('x')
        dfinvcdf = pd.DataFrame({'manual': d_manual.loginvcdf(u, **params),
                                 'u': u}).set_index('u')
    else:
        dfpdf = pd.DataFrame({'manual': d_manual.pdf(x, **params),
                              'x': x}).set_index('x')
        dfcdf = pd.DataFrame({'manual': d_manual.cdf(x, **params),
                              'x': x}).set_index('x')
        dfinvcdf = pd.DataFrame({'manual': d_manual.invcdf(u, **params),
                                 'u': u}).set_index('u')
            
    return dfpdf, dfcdf, dfinvcdf


def log_jacobian_det(f_inv_x, x):
    """ Calc log of Jacobian determinant 
        used to aid log-likelihood maximisation of copula marginals
        see JPL: https://github.com/junpenglao/advance-bayesian-modelling-with-PyMC3/blob/master/Advance_topics/Box-Cox%20transformation.ipynb
    """
    grad = tt.reshape(pm.theanof.gradient(tt.sum(f_inv_x), [x]), x.shape)
    return tt.log(tt.abs_(grad))