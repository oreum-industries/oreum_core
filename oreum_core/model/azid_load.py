# model.azid_load.py
# copyright 2021 Oreum OÜ
import errno
import os
# import warnings
import arviz as az

def read_azid(dir_traces=[], fn='azid'):
    """ Convenience: read arviz.InferenceData object from file """
    return az.from_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))       


def write_azid(azid, dir_traces=[], fn='azid'):
    """ Convenience: write arviz.InferenceData to file """

    d = os.path.join(*dir_traces)
    fqn = os.path.join(*dir_traces, f'{fn}.netcdf')
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    azid.to_netcdf(fqn)
    return f'Wrote: {fqn}'
