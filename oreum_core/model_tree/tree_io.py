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

# model_tree.tree_io.py
"""Handling of Fitted Model Parameters"""
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

    def read(self, fn: str) -> Booster:
        """Read XGB.core.Booster object from fn e.g. `bst.json`"""
        fqn = self.get_path_read(Path(self.snl.clean(fn)).with_suffix('.json'))
        bst = Booster()
        bst.load_model(fname=str(fqn.resolve()))
        _log.info(f'Read Booster model data from {str(fqn.resolve())}')
        return bst

    def write(self, bst: Booster, fn: str = '') -> Path:
        """Accept XGB.core.Booster object and fn e.g. `bst.json`, write to file"""
        fn = 'bst.json' if fn == '' else fn
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.json'))
        bst.save_model(str(fqn.resolve()))
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn

    def get_sqlite_uri_for_optuna_study(self, fn: str = '') -> str:
        """Get URI of local SQLite DB to pass to optuna.create_study(storage=)"""
        fn = 'optuna_study.sqlite' if fn == '' else fn
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.sqlite'))
        return f'sqlite:////{str(fqn.resolve())}'
