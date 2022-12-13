# model.model_io.py
# copyright 2022 Oreum Industries
import logging
from pathlib import Path

import arviz as az
from pymc3.model_graph import model_to_graphviz

from oreum_core.file_io import BaseFileIO
from oreum_core.model import BasePYMC3Model

_log = logging.getLogger(__name__)


class ModelIO(BaseFileIO):
    """Helper class to read/write NetCDF files for Arviz inference data.
    Can also write model graphs to file
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_idata(self, fqn: str) -> az.InferenceData:
        """Read arviz.InferenceData object from fqn e.g. `model/mdl.netcdf`"""
        path = self.get_path_read(fqn)
        return az.from_netcdf(str(path))

    def write_idata(self, mdl: BasePYMC3Model, fqn: str = '') -> Path:
        """Accept a BasePYMC3Model object mdl, and fqn e.g. `model/mdl.netcdf`
        write to fqn
        """
        path = self.get_path_write(fqn)
        if fqn == '':
            path = path.joinpath(Path(f'{mdl.name}.netcdf'))
        mdl.idata.to_netcdf(str(path))
        _log.info(f'Written to {str(path)}')
        return path

    def write_graph(
        self, mdl: BasePYMC3Model, fqn: str = '', format: str = 'png'
    ) -> Path:
        """Accept a BasePYMC3Model object mdl, get the graphviz representation
        Write to file and return the fqn to allow use within eda.display_image_file()
        """
        path = self.get_path_write(fqn)
        if fqn == '':
            path = path.joinpath(Path(f'{mdl.name}.{format}'))
        gv = model_to_graphviz(mdl.model, formatting='plain')
        if format == 'png':
            gv.attr(dpi='300')
        elif format == 'svg':
            pass
        else:
            raise ValueError('format must be in {"png", "svg"}')

        # gv auto adds the file extension, so pre-remove if present
        gv.render(filename=str(path.with_suffix('')), format=format, cleanup=True)
        _log.info(f'Written to {str(path)}')
        return path
