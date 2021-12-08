# model.file_io.py
# copyright 2021 Oreum OÜ
import errno
import os
# import warnings
import arviz as az

def read_azid(dir_traces=[], fn='azid'):
    """ Convenience: read arviz.InferenceData object from file """
    return az.from_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))       


def write_azid(mdl, dir_traces=[]):
    """ Accept a BasePYMC3Model object mdl, and write the 
        mdl.idata (an arviz.InferenceData object) to file using mdl.name 
    """
    d = os.path.join(*dir_traces)
    fqn = os.path.join(*dir_traces, f'{mdl.name}.netcdf')
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    mdl.idata.to_netcdf(fqn)
    return f'Wrote: {fqn}'
