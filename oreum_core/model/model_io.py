# Copyright 2023 Oreum Industries
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

# model.model_io.py
"""Handling of Model Posterior Samples"""
import logging
from pathlib import Path

import arviz as az
from pymc.model_graph import model_to_graphviz

from oreum_core.file_io import BaseFileIO
from oreum_core.model import BasePYMCModel

__all__ = ['ModelIO']

_log = logging.getLogger(__name__)


class ModelIO(BaseFileIO):
    """Helper class to read/write NetCDF files for Arviz inference data.
    Can also write model graphs to file
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def __init__(self, *args, **kwargs):
        """Inherit super"""
        super().__init__(*args, **kwargs)

    def read_idata(self, fn: str) -> az.InferenceData:
        """Read arviz.InferenceData object from fn e.g. `mdl.netcdf`"""
        fqn = self.get_path_read(fn)
        _log.info(f'Read idata from {str(fqn.resolve())}')
        return az.from_netcdf(str(fqn.resolve()))

    def write_idata(self, mdl: BasePYMCModel, fn: str = '') -> Path:
        """Accept BasePYMCModel object and fn e.g. `mdl.netcdf`, write to file"""
        fn = f'idata_{mdl.name}.netcdf' if fn == '' else fn
        fqn = self.get_path_write(fn)
        mdl.idata.to_netcdf(str(fqn.resolve()))
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn

    def write_graph(
        self, mdl: BasePYMCModel, fn: str = '', fmt: str = 'png', **kwargs
    ) -> Path:
        """Accept a BasePYMCModel object mdl, get the graphviz representation
        Write to file and return the fqn to allow use within eda.display_image_file()
        """
        t = kwargs.pop('txtadd', None)
        fn = f"{'_'.join(filter(None, [mdl.name, t]))}.{fmt}" if fn == '' else fn
        fqn = self.get_path_write(fn)
        gv = model_to_graphviz(mdl.model, formatting='plain')
        if fmt == 'png':
            gv.attr(dpi='300')
        elif fmt == 'svg':
            pass
        else:
            raise ValueError('format must be in {"png", "svg"}')

        # gv auto adds the file extension, so pre-remove if present
        gv.render(filename=str(fqn.with_suffix('')), format=fmt, cleanup=True)
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn
