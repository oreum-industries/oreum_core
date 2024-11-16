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

# eda.eda_io.py
"""EDA File Handling"""
import logging
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure

from ..curate.data_io import PandasExcelIO
from ..utils.file_io import BaseFileIO
from .describe import describe, get_fts_by_dtype

__all__ = ['FigureIO', 'output_data_dict']

_log = logging.getLogger(__name__)

sns.set_theme(
    style='darkgrid',
    palette='muted',
    context='notebook',
    rc={'figure.dpi': 72, 'savefig.dpi': 144, 'figure.figsize': (12, 4)},
)


class FigureIO(BaseFileIO):
    """Helper class to save matplotlib.figure.Figure objects to image file"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write(self, f: figure.Figure, fn: str, *args, **kwargs) -> Path:
        """Accept figure.Figure & fn str incl. dots e.g. `plots/plot.2.png`,
        write to fqn"""
        if Path(fn).suffix != '.png':  # either no suffix or there's dots in fn
            fn = Path(self.snl.clean(fn)).with_suffix('.png')
        fqn = self.get_path_write(fn)
        f.savefig(
            fname=fqn, format='png', bbox_inches='tight', dpi=300, *args, **kwargs
        )
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn

    def read(
        self,
        fqn: Path = None,
        fn: str = None,
        extension: str = '.png',
        title: str = None,
        figsize: tuple = (12, 4),
    ) -> figure.Figure:
        """Hacky way to display pre-created image file in a Notebook such that
        nbconvert can see it and render to PDF
        If don't supply fqn, then this will build fn according to get_path_read
        Render according to usual rcParams (set at module-level)
        NOTE:
        All the alternatives are bad
            1. This is entirely missed by nbconvert --to pdf because it goes
            looking from the dir structure point from which you call nbconvert
            and also missed by --to slides because that even more stupidly
            prepends the output dir onto the URL for the image!!??
            # <img src="img.jpg" style="float:center; width:900px" />

            2. This one used to cause any following markdown to render as
            monospace font in the PDF, but as-at 2024-11-14 it seems to not be
            an issue? Keep this not around for future reference. Still doesn't
            render to slides though.
            # from IPython.display import Image
            # Image("./assets/img/oreum_eloss_blueprint3.jpg", retina=True)
        """
        if fn is not None:
            fqn = self.get_path_read(Path(fn).with_suffix(extension))
        _log.info(f'Read image from {str(fqn.resolve())}')
        img = mpimg.imread(fqn)
        with plt.ioff():
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
                _ = f.suptitle(f'{title}', fontsize=12, y=1.0)
            _ = f.tight_layout()
        plt.show()
        return f


def output_data_dict(
    df: pd.DataFrame, dd_notes: dict[str, str], fqp: Path, fn: str = ''
) -> None:
    """Helper fn to output a data dict with automatic eda.describe"""

    # flag if is index
    idx_names = list(df.index.names)
    dfi = pd.DataFrame({'ft': idx_names, 'is_index': np.ones(len(idx_names))})

    # get desc overview
    nrows = 3
    dfd = describe(df, nrows=nrows, return_df=True)
    cols = dfd.columns.values
    cols[:nrows] = [f'example_row_{i}' for i in range(nrows)]
    dfd.columns = cols

    # attach
    dfd = pd.merge(dfi, dfd, how='right', on='ft')
    dfd['is_index'] = dfd['is_index'].fillna(0).astype(bool)
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
    excelio = PandasExcelIO(rootdir=fqp)
    fn = f'_{fn}' if fn != '' else fn
    fn = f'datadict{fn}'
    excelio.writer_open(fn)
    excelio.writer_write(dfd, sheet_name='overview', index=True)

    # write cats to separate sheets for levels (but not indexes since they're unique)
    for ft in dfd.loc[dfd['dtype'].isin(['categorical', 'cat'])].index.values:
        if ft not in df.index.names:
            dfg = df[ft].value_counts(dropna=False).to_frame('count')
            dfg['prop'] = dfg['count'] / len(df)
            dfg.index.name = 'value'
            excelio.writer_write(
                dfg,
                sheet_name=f'cat_{ft[:24]}...' if len(ft) >= 27 else f'cat_{ft}',
                index=True,
                float_format='%.3f',
                na_rep='NULL',
            )

    excelio.writer_close()
