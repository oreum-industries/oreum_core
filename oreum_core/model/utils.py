# model.utils.py
# copyright 2022 Oreum Industries
import errno
import os

import arviz as az
import pymc3 as pm


def read_azid(dir_traces: list = [], fn: str = 'azid'):
    """Convenience: read arviz.InferenceData object from file"""
    # with az.rc_context(rc={'data.load': 'eager'}):   # alternative: 'lazy'
    azid = az.from_netcdf(os.path.join(*dir_traces, f'{fn}.netcdf'))
    return azid


def write_azid(mdl, dir_traces: list = []):
    """Accept a BasePYMC3Model object mdl, and write the
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


def save_graph(mdl, fp: list = [], fn_extra: str = '', format: str = 'png'):
    """Accept a BasePYMC3Model object mdl, get the graphviz representation,
    write to file and return the fqn
    """
    gv = pm.model_graph.model_to_graphviz(mdl.model, formatting='plain')
    if fn_extra != '':
        fn_extra = f'_{fn_extra}'
    fqn = os.path.join(*fp, f'{mdl.name}{fn_extra}')

    if format == 'png':
        gv.attr(dpi='300')
    elif format == 'svg':
        pass
    else:
        raise AttributeError

    # auto adds the file extension
    gv.render(filename=fqn, format=format, cleanup=True)

    return fqn + f'.{format}'
