# model.io.py
# copyright 2022 Oreum Industries
from pathlib import Path

import arviz as az
from model import BasePYMC3Model


class ModelIO:
    """Helper class to read/write NetCDF files for Arviz inference data
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def read(fqn: str) -> az.InferenceData:
        """Read arviz.InferenceData object from fqn e.g. `model/mdl.netcdf`"""
        path = Path(fqn)
        if not path.exists():
            raise FileNotFoundError(f'Required file does not exist {str(path)}')
        # with az.rc_context(rc={'data.load': 'eager'}):   # alternative: 'lazy'
        idata = az.from_netcdf(str(path))
        return idata

    def write(mdl: BasePYMC3Model, fqn: str, use_model_name: bool = False) -> str:
        """Accept a BasePYMC3Model object mdl, and fqn e.g. `model/mdl.netcdf`
        write to fqn, optionally set `use_model_name` True to overwrite the
        filename with mdl.name
        """
        path = Path(fqn)
        dr = Path(*path.parts[:-1])
        if not dr.is_dir():
            raise FileNotFoundError(f'Required dir does not exist {str(dr)}')
        if use_model_name:
            path = Path(*path.parts[:-1], f'{mdl.name}.netcdf')

        try:
            mdl.idata.to_netcdf(str(path))
        except Exception as e:
            raise e
        return f'Written to {str(path)}'
