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

# eda.eda_io.py
"""EDA File Handling"""
import logging
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import figure

from ..utils.file_io import BaseFileIO
from .describe import describe, get_fts_by_dtype

__all__ = ['FigureIO', 'display_image_file', 'output_data_dict']

_log = logging.getLogger(__name__)


class FigureIO(BaseFileIO):
    """Helper class to save matplotlib.figure.Figure objects to image file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write(self, f: figure.Figure, fn: str, *args, **kwargs) -> Path:
        """Accept figure.Figure & fqn e.g. `plots/plot.png`, write to fqn"""
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.png'))
        f.savefig(
            fname=fqn, format='png', bbox_inches='tight', dpi=300, *args, **kwargs
        )
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn


def display_image_file(
    fqn: str, title: str = None, figsize: tuple = (12, 6)
) -> figure.Figure:
    """Hacky way to display pre-created image file in a Notebook
    such that nbconvert can see it and render to PDF
    Force to max width 16 inches, for fullwidth render in live Notebook and PDF

    NOTE:
    Alternatives are bad
        1. This one is entirely missed by nbconvert at render to PDF
        # <img src="img.jpg" style="float:center; width:900px" />

        2. This one causes following markdown to render monospace in PDF
        # from IPython.display import Image
        # Image("./assets/img/oreum_eloss_blueprint3.jpg", retina=True)
    """
    img = mpimg.imread(fqn)
    f, axs = plt.subplots(1, 1, figsize=figsize)
    _ = axs.imshow(img)
    ax = plt.gca()
    _ = ax.grid(False)
    _ = ax.set_frame_on(False)
    _ = plt.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    if title is not None:
        _ = f.suptitle(f'{title}', y=1.0)
    _ = f.tight_layout()
    return f


def output_data_dict(
    df: pd.DataFrame, dd_notes: dict[str, str], fqp: Path, fn: str = ''
):
    """Convenience fn: output data dict"""

    # flag if is index
    idx_names = list(df.index.names)
    dfi = pd.DataFrame({'ft': idx_names, 'is_index': [True] * len(idx_names)})

    # get desc overview
    nrows = 3
    dfd = describe(df, nrows=nrows, return_df=True)
    cols = dfd.columns.values
    cols[:nrows] = [f'example_row_{i}' for i in range(nrows)]
    dfd.columns = cols

    # attach
    dfd = pd.merge(dfi, dfd, how='right', on='ft')
    dfd['is_index'] = dfd['is_index'].fillna(False)
    dfd.set_index('ft', inplace=True)

    # set dtypes categorical
    df_dtypes = get_fts_by_dtype(df.reset_index(), as_dataframe=True)
    dfd['dtype'] = df_dtypes['dtype']
    del df_dtypes

    # attached notes
    df_dd_notes = pd.DataFrame(dd_notes, index=['notes']).T
    df_dd_notes.index.name = 'ft'
    dfd = pd.merge(dfd, df_dd_notes, how='left', left_index=True, right_index=True)

    # write overview
    fileio = BaseFileIO(rootdir=fqp)
    fn = f'_{fn}' if fn != '' else fn
    fqn = fileio.get_path_write(f'datadict{fn}.xlsx')

    writer = pd.ExcelWriter(str(fqn), engine='xlsxwriter')
    dfd.to_excel(writer, sheet_name='overview', index=True)

    # write cats to separate sheets for levels (but not indexes since they're unique)
    for ft in dfd.loc[dfd['dtype'].isin(['categorical', 'cat'])].index.values:
        if ft not in df.index.names:
            dfg = (df[ft].value_counts(dropna=False) / len(df)).to_frame('prop')
            dfg.index.name = 'value'
            dfg.reset_index().to_excel(
                writer,
                sheet_name=f'{ft[:28]}...' if len(ft) >= 31 else ft,
                index=False,
                float_format='%.3f',
                na_rep='NULL',
            )

    writer.close()
    _log.info(f'Written to {str(fqn.resolve())}')
    return fqn
