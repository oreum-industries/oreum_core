# model.describe.py
# copyright 2021 Oreum OÜ
import arviz as az
import numpy as np
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

   
def extract_yobs_yhat(azid, obs='y', pred='yhat'):
    """Convenience: extract y_obs, y_hat from azid
        get yhat in the shape (nsamples, nobs)
    """
    nsamp = np.product(azid.posterior_predictive[pred].shape[:-1])    
    yobs = azid.constant_data[obs].values                            # (nobs,)
    yhat = azid.posterior_predictive[pred].values.reshape(nsamp, -1) # (nsamp, nobs)
    
    return yobs, yhat


def describe_dist(mdl, log=False, inc_summary=False):
    """ Convenience: get distribution descriptions from distributions.DistNumpy 
        and return for printing or Markdown
    """
    title = f'{mdl.name}: Natural Distributions'
    dist = mdl.dist_natural 
    if log:
        title = f'{mdlname}: Logged Distributions'
        dist = mdl.dist_log

    if inc_summary:
        return title, {**mdl.notation, **dist, **mdl.conditions, **mdl.summary_stats}     

    return title, {**dist}