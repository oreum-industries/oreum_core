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
        super().__init__(*args, **kwargs)

    def read_idata(self, fqn: str) -> az.InferenceData:
        """Read arviz.InferenceData object from fqn e.g. `model/mdl.netcdf`"""
        path = self.get_path_read(fqn, use_rootdir=False)
        _log.info(f'Read idata from {str(path.resolve())}')
        return az.from_netcdf(str(path.resolve()))

    def write_idata(self, mdl: BasePYMCModel, fqn: str = '') -> Path:
        """Accept a BasePYMCModel object mdl, and fqn e.g. `model/mdl.netcdf`
        write to fqn
        """
        path = self.get_path_write(fqn, use_rootdir=False)
        if fqn == '':
            path = path.joinpath(Path(f'idata_{mdl.name}.netcdf'))
        mdl.idata.to_netcdf(str(path.resolve()))
        _log.info(f'Written to {str(path.resolve())}')
        return path

    def write_graph(
        self, mdl: BasePYMCModel, fqn: str = '', txtadd: str = None, fmt: str = 'png'
    ) -> Path:
        """Accept a BasePYMCModel object mdl, get the graphviz representation
        Write to file and return the fqn to allow use within eda.display_image_file()
        """
        path = self.get_path_write(fqn)
        if fqn == '':
            path = path.joinpath(
                Path(f"{'_'.join(filter(None, [mdl.name, txtadd]))}.{fmt}")
            )
        gv = model_to_graphviz(mdl.model, formatting='plain')
        if fmt == 'png':
            gv.attr(dpi='300')
        elif fmt == 'svg':
            pass
        else:
            raise ValueError('format must be in {"png", "svg"}')

        # gv auto adds the file extension, so pre-remove if present
        gv.render(filename=str(path.with_suffix('')), format=fmt, cleanup=True)
        _log.info(f'Written to {str(path.resolve())}')
        return path
