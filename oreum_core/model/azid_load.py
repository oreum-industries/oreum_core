# model.azid_load.py
# copyright 2021 Oreum OÜ
import os
import arviz as az

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
