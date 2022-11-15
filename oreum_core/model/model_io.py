# model.model_io.py
# copyright 2022 Oreum Industries
from pathlib import Path

import arviz as az

from oreum_core.file_io import BaseFileIO
from oreum_core.model import BasePYMC3Model


class ModelIO(BaseFileIO):
    """Helper class to read/write NetCDF files for Arviz inference data
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, fqn: str) -> az.InferenceData:
        """Read arviz.InferenceData object from fqn e.g. `model/mdl.netcdf`"""
        path = self.get_path_read(fqn)
        return az.from_netcdf(str(path))

    def write(self, mdl: BasePYMC3Model, fqn: str, use_model_name: bool = False) -> str:
        """Accept a BasePYMC3Model object mdl, and fqn e.g. `model/mdl.netcdf`
        write to fqn, optionally set `use_model_name` True to overwrite the
        filename with mdl.name
        """
        path = self.get_path_write(fqn)
        if use_model_name:
            path = Path(*path.parts[:-1], f'{mdl.name}.netcdf')
        mdl.idata.to_netcdf(str(path))
        return f'Written to {str(path)}'
