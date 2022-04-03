# model.describe.py
# copyright 2022 Oreum Industries
import numpy as np
import patsy as pt
import re

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

def model_desc(fml):
    """ Convenience: return patsy modeldesc
        NOTE: `.describe()` doesn't return the `1 +` (intercept) term in the 
            case that it's present. check and add if needed
    """
    fmls = fml.split(' ~ ')
    add_intercept = False if re.match(r'1 \+', fml) is None else True
    r = pt.ModelDesc.from_formula(fml).describe()
    if len(fmls) == 2:
        rs = r.split(' ~ ')
        if add_intercept:
            r = f'{rs[0]} ~ 1 + {rs[1]}'
    elif len(fmls) == 1:
        if add_intercept:
            r = f'1 + {r[2:]}'
    else:
        raise ValueError('fml must have only a single tilde `~`')
    return f'patsy linear model desc:\n{r}\n'

   
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
        title = f'{mdl.name}: Logged Distributions'
        dist = mdl.dist_log

    if inc_summary:
        return title, {**mdl.notation, **dist, **mdl.conditions, **mdl.summary_stats}     
    return title, {**dist}