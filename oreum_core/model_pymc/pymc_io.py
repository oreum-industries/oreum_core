# Copyright 2024 Oreum Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# model_pymc.pymc_io.py
"""Handling of Model Posterior Samples"""
import logging
from pathlib import Path

import arviz as az
import graphviz
from pymc.model_graph import model_to_graphviz

from ..utils.file_io import BaseFileIO
from . import BasePYMCModel

__all__ = ['PYMCIO']

_log = logging.getLogger(__name__)


class PYMCIO(BaseFileIO):
    """Helper class to read/write NetCDF files for Arviz inference data.
    Can also write model graphs to file
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def __init__(self, *args, **kwargs):
        """Inherit super"""
        super().__init__(*args, **kwargs)

    def read_idata(
        self, mdl: BasePYMCModel = None, fn: str = '', **kwargs
    ) -> az.InferenceData:
        """Read InferenceData using mdl.mdl_id_fn + txtadd, or from fn"""
        txtadd = kwargs.pop('txtadd', None)
        if mdl is not None:
            fn = '_'.join(filter(None, ['idata', mdl.mdl_id_fn, txtadd]))
        fqn = self.get_path_read(Path(self.snl.clean(fn)).with_suffix('.netcdf'))
        idata = az.from_netcdf(str(fqn.resolve()))
        _log.info(f'Read model idata from {str(fqn.resolve())}')
        return idata

    def write_idata(self, mdl: BasePYMCModel, fn: str = '', **kwargs) -> Path:
        """Accept BasePYMCModel object write to InferenceData using
        mdl.mdl_id_fn + txtadd"""
        txtadd = kwargs.pop('txtadd', None)
        if fn == '':
            fn = '_'.join(filter(None, ['idata', mdl.mdl_id_fn, txtadd]))
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.netcdf'))
        mdl.idata.to_netcdf(str(fqn.resolve()))
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn

    def write_graph(
        self,
        mdl: BasePYMCModel,
        fn: str = '',
        fmt: str = 'png',
        write: bool = True,
        **kwargs,
    ) -> Path | graphviz.graphs.Digraph:
        """Accept a BasePYMCModel object mdl, get the graphviz representation
        Write to file and return the fqn to allow use within
        eda_io.FigureIO.read()
        Optionally set `write = False` and receive the graphviz directly
        """
        txtadd = kwargs.pop('txtadd', None)
        if fn == '':
            fn = '_'.join(filter(None, ['graph', mdl.mdl_id_fn, txtadd]))
        fqn = self.get_path_write(f'{fn}.{fmt}')
        gv = model_to_graphviz(mdl.model, formatting='plain')
        if write == False:
            return gv
        if fmt == 'png':
            gv.attr(dpi='300')
        elif fmt == 'svg':
            pass
        else:
            raise ValueError('format must be in {"png", "svg"}')

        # gv auto-adds the file extension, so pre-remove if present
        gv.render(filename=str(fqn.with_suffix('')), format=fmt, cleanup=True)
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn
