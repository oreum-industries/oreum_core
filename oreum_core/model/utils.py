# model.utils.py
# copyright 2022 Oreum Industries
from pathlib import Path

import pymc3 as pm


def save_graph(mdl, fp: list = [], fn_extra: str = '', format: str = 'png'):
    """Accept a BasePYMC3Model object mdl, get the graphviz representation,
    write to file and return the fqn
    """
    gv = pm.model_graph.model_to_graphviz(mdl.model, formatting='plain')
    if fn_extra != '':
        fn_extra = f'_{fn_extra}'
    fqn = Path.join(*fp, f'{mdl.name}{fn_extra}')

    if format == 'png':
        gv.attr(dpi='300')
    elif format == 'svg':
        pass
    else:
        raise AttributeError

    # auto adds the file extension
    gv.render(filename=str(fqn), format=format, cleanup=True)

    return f'{fqn}.{format}'
