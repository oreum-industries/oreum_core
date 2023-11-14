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

# model_xgb.xgb_io.py
"""Handling of Model Posterior Samples"""
import logging
from pathlib import Path

from xgboost.core import Booster

from ..utils.file_io import BaseFileIO

__all__ = ['XGBIO']

_log = logging.getLogger(__name__)


class XGBIO(BaseFileIO):
    """Helper class to read/write XGBoost JSON files for fitted model data.
    Note similar behaviour to curate.data_io.SimpleStringIO
    """

    def __init__(self, *args, **kwargs):
        """Inherit super"""
        super().__init__(*args, **kwargs)

    def read_data(self, fn: str) -> Booster:
        """Read XGB.core.Booster object from fn e.g. `bst.json`"""
        fqn = self.get_path_read(fn)
        _log.info(f'Read model data from {str(fqn.resolve())}')
        return None  # az.from_netcdf(str(fqn.resolve()))

    def write_data(self, bst: Booster, fn: str = '') -> Path:
        """Accept XGB.core.Booster object and fn e.g. `bst.json`, write to file"""
        fn = 'bst.json' if fn == '' else fn
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.json'))
        bst.save_model(str(fqn.resolve()))
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn
