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

# curate.data_transform.py
"""Data Transformations"""
import logging
import re

import numpy as np
import pandas as pd
import patsy as pt

from ..utils.snakey_lowercaser import SnakeyLowercaser

__all__ = [
    'DatatypeConverter',
    'DatasetReshaper',
    'Transformer',
    'Standardizer',
    'compress_factor_levels',
]

_log = logging.getLogger(__name__)


class DatatypeConverter:
    """Force correct datatypes according to what model expects"""

    def __init__(self, ftsd: dict, ftslvlcat: dict = {}, date_format: str = '%Y-%m-%d'):
        """Initialise with fts and optionally specify factors with specific levels

        Use with a fts dict of form:
            ftsd = dict(
                fcat = [],
                fstr = [],
                fbool = [],
                fdate = [],
                fyear = [],
                fint = [],
                ffloat = [],
                fverbatim = [],        # maintain in current dtype)
        """
        self.ftsd = dict(
            fcat=ftsd.get('fcat', []),
            fstr=ftsd.get('fstr', []),
            fbool=ftsd.get('fbool', []),
            fdate=ftsd.get('fdate', []),
            fyear=ftsd.get('fyear', []),
            fint=ftsd.get('fint', []),
            ffloat=ftsd.get('ffloat', []),
            fverbatim=ftsd.get('fverbatim', []),  # keep verbatim
        )
        self.ftslvlcat = ftslvlcat
        self.rx_number_junk = re.compile(r'[#$€£₤¥,;%\s]')
        self.date_format = date_format
        inv_bool_dict = {
            True: ['yes', 'y', 'true', 't', '1', 1, 1.0],
            False: ['no', 'n', 'false', 'f', '0', 0, 0.0],
        }
        self.bool_dict = {v: k for k, vs in inv_bool_dict.items() for v in vs}
        self.strnans = ['none', 'nan', 'null', 'na', 'n/a', 'missing', 'empty', '']

    def _force_dtypes(self, dfraw):
        """Select fts and convert dtypes. Return cleaned df"""
        snl = SnakeyLowercaser()

        # subselect desired fts
        # TODO make this optional
        fts_all = [w for _, v in self.ftsd.items() for w in v]
        df = dfraw[fts_all].copy()

        for ft in self.ftsd['fcat'] + self.ftsd['fstr']:
            # tame string, clean, handle nulls
            idx = df[ft].notnull()
            vals = df.loc[idx, ft].astype(str, errors='raise').apply(snl.clean)
            df.drop(ft, axis=1, inplace=True)
            df.loc[~idx, ft] = 'nan'
            df.loc[idx, ft] = vals
            if ft in self.ftsd['fcat']:
                df[ft] = pd.Categorical(df[ft].values, ordered=True)
            else:
                df[ft] = df[ft].astype('string')

        for ft in self.ftsd['fbool']:
            # tame string, strip, lower, use self.bool_dict, use pd.NA
            if df.dtypes[ft] == object:
                df[ft] = df[ft].apply(lambda x: str(x).strip().lower())
                df.loc[df[ft].isin(self.strnans), ft] = np.nan
                df[ft] = df[ft].apply(lambda x: self.bool_dict.get(x, x))

                if set(df[ft].unique()) != set([True, False, np.nan]):
                    # avoid converting anything not yet properly mapped
                    continue
            df[ft] = df[ft].convert_dtypes(convert_boolean=True)

        for ft in self.ftsd['fyear']:
            if df.dtypes[ft] == object:
                df[ft] = (
                    df[ft]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(lambda x: self.rx_number_junk.sub('', x))
                )
                df.loc[df[ft].isin(self.strnans), ft] = np.nan
            df[ft] = df[ft].astype(float, errors='raise')
            if pd.isnull(df[ft]).sum() == 0:
                df[ft] = df[ft].astype(int, errors='raise')
            df[ft] = pd.to_datetime(df[ft], errors='raise', format='%Y')

        for ft in self.ftsd['fdate']:
            df[ft] = pd.to_datetime(df[ft], errors='raise', format=self.date_format)
        try:
            for ft in self.ftsd['fint']:
                if df.dtypes[ft] == object:
                    df[ft] = (
                        df[ft]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(lambda x: self.rx_number_junk.sub('', x))
                    )
                    df.loc[df[ft].isin(self.strnans), ft] = np.nan
                df[ft] = df[ft].astype(float, errors='raise')
                if pd.isnull(df[ft]).sum() == 0:
                    df[ft] = df[ft].astype(int, errors='raise')
        except Exception as e:
            raise Exception(f'{str(e)} in ft: {ft}').with_traceback(
                e.__traceback__
            )  # from e

        try:
            for ft in self.ftsd['ffloat']:
                if df.dtypes[ft] == object:
                    df[ft] = (
                        df[ft]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(lambda x: self.rx_number_junk.sub('', x))
                    )
                    df.loc[df[ft].isin(self.strnans), ft] = np.nan
                df[ft] = df[ft].astype(float, errors='raise')
        except Exception as e:
            raise e(ft)

        # NOTE verbatim will simply remain. We're now at the end of the columns
        # for ft in self.fts['fverbatim']:
        #     _log.info(f'Kept ft verbatim: {ft}')

        return df

    def convert_dtypes(self, df):
        """Force dtypes for recognised features (fts) in df"""
        dfclean = self._force_dtypes(df)

        for ft, lvls in self.ftslvlcat.items():
            dfclean[ft] = pd.Categorical(
                dfclean[ft].values, categories=lvls, ordered=True
            )

        return dfclean


class DatasetReshaper:
    """Convenience functions to reshape whole datasets"""

    def __init__(self):
        pass

    def create_dfcmb(self, df: pd.DataFrame, fts: dict) -> pd.DataFrame:
        """Create a combination dataset `dfcmb` from inputted `df`.

        *Not* a big groupby (equiv to cartesian join) for factor values
        and concats numerics. Instead, this concats unique vals in cols
        which yields a far more compact dfcmb. Note that the columns will
        be ragged so need to fill NULLS with (any) value from that column

        The shape and datatypes matter.

        We use this because as of the current patsy (0.5.1), design_info
        objects still can't be pickled, which means that when you want to
        transform a test / holdout sample according to a training dataset
        you have to instantiate a new design_info object in memory.

        In this codebase we set the design_info attribute on the
        Transformer object, so you can keep that object around in memory,
        but upon memory-loss you need to call it with the dfcmb created
        here e.g. Transformer.fit_transform(dfcmb).

        I recommend storing dfcmb in a DB: it only needs to be updated if
        the modelled features change (e.g. include a new feature) or if
        factor values change (e.g. a new construction_type).

        NOTE: only takes datatypes as understood by patsy: factor-values
            (aka categoricals aka strings), or ints or floats. No dates.
        """
        dfcmb = pd.DataFrame(index=[0])
        fts_factor = fts.get('fcat', []) + fts.get('fbool', [])
        for ft in fts_factor:
            colnames_pre = list(dfcmb.columns.values)
            s = pd.Series(np.unique(df[ft]), name=ft)
            dfcmb = pd.concat([dfcmb, s], axis=1, join='outer', ignore_index=True)
            dfcmb.columns = colnames_pre + [ft]

        for ft in dfcmb.columns:
            dfcmb[ft] = dfcmb[ft].fillna(dfcmb[ft][:1].values[0])
            dfcmb[ft] = dfcmb[ft].astype(df[ft].dtype)

            # TODO: force order for categorical
            # df['fpc_aais_ctgry'] = pd.Categorical(df['fpc_aais_ctgry'].values, categories=vals, ordered=True)

        for ft in fts.get('fint'):
            dfcmb[ft] = 1

        for ft in fts.get('ffloat'):
            dfcmb[ft] = 1.0

        _log.info(
            'Reduced df ({} rows, {:,.0f} KB) to dfcmb ({} rows, {:,.0f} KB)'.format(
                df.shape[0],
                df.values.nbytes / 1e3,
                dfcmb.shape[0],
                dfcmb.values.nbytes / 1e3,
            )
        )

        return dfcmb

    def _create_dfcmb_big(self, df: pd.DataFrame, fts: dict) -> pd.DataFrame:
        """Create a combination dataset `dfcmb` from inputted `df`.
        Just a big groupby (equiv to cartesian join) for factor values
        and concats numerics. The shape and datatypes matter.

        We use this because as of the current patsy (0.5.1), design_info
        objects still can't be pickled, which means that when you want to
        transform a test / holdout sample according to a training dataset
        you have to instantiate a new design_info object in memory.

        In this codebase we set the design_info attribute on the
        Transformer object, so you can keep that object around in memory,
        but upon memory-loss you need to call it with the dfcmb created
        here e.g. Transformer.fit_transform(dfcmb).

        I recommend storing dfcmb in a DB: it only needs to be updated if
        the modelled features change (e.g. include a new feature) or if
        factor values change (e.g. a new construction_type).

        NOTE: only takes datatypes as understood by patsy: factor-values
            (aka categoricals aka strings), or ints or floats. No dates.
        """
        dfcmb = pd.DataFrame(index=[0])
        fts_factor = fts.get('fcat', []) + fts.get('fbool', [])
        if len(fts_factor) > 0:
            dfcmb = df.groupby(fts_factor).size().reset_index().iloc[:, :-1]

        for ft in fts.get('fint'):
            dfcmb[ft] = 1

        for ft in fts.get('ffloat'):
            dfcmb[ft] = 1.0

        _log.info(
            'Reduced df ({} rows, {:,.0f} KB) to dfcmb ({} rows, {:,.0f} KB)'.format(
                df.shape[0],
                df.values.nbytes / 1e3,
                dfcmb.shape[0],
                dfcmb.values.nbytes / 1e3,
            )
        )

        return dfcmb


class Transformer:
    """Model-agnostic patsy transformer from row-wise natural observations
    into dmatrix according to patsy formula or patsy design_info object
    NOTES:
        + design_info is stateful
        + it's reasonable to initialise this per-observation but far more
          efficient to initialise once and persist in-memory
        + allows for F() and F():F() terms
    """

    def __init__(self):
        self.design_info = None
        self.col_idx_numerics = None
        self.rx_get_f = re.compile(r'(F\(([a-z0-9_:]+?)\))')
        self.fts_fact_mapping = {}
        self.original_fml = None

    def fit_transform(
        self, fml: str, df: pd.DataFrame, propagate_nans: bool = False
    ) -> pd.DataFrame:
        """Fit a new design_info attribute for this instance according to
        `fml` acting upon `df`. Return the transformed dmatrix (np.array)
        Use this for a new training set or to initialise the transformer
        based on dfcmb.
        """
        self.original_fml = fml
        # deal w/ any fml components F(), use set() to uniqify
        fts_f = list(set(self.rx_get_f.findall(fml)))

        if len(fts_f) > 0:
            df = df.copy()
            for ft_f in fts_f:
                dt = df[ft_f[1]].dtype.name
                if dt != 'category':
                    raise AttributeError(
                        f'fml contains F({ft_f[1]}),'
                        + f' dtype={dt}, but {ft_f[1]} is not categorical'
                    )
                # map feature to int based on its preexisting categorical order
                # https://stackoverflow.com/a/55304375/1165112
                map_int_to_fact = dict(enumerate(df[ft_f[1]].cat.categories))
                map_fact_to_int = {v: k for k, v in map_int_to_fact.items()}
                self.fts_fact_mapping[ft_f[1]] = map_fact_to_int
                df[ft_f[1]] = df[ft_f[1]].map(map_fact_to_int).astype(np.int)

                # replace F() in fml so patsy can work as normal w/ our new int type
                fml = fml.replace(ft_f[0], ft_f[1])
        _log.info(f'Created fml: {fml}')
        # TODO add option to output matrix   # np.asarray(mx_ex)
        # TODO add check for fml contains `~` and handle accordingly

        # do nothing, see https://stackoverflow.com/a/51641183/1165112
        na_action = pt.NAAction(NA_types=[]) if propagate_nans else 'raise'

        df_ex = pt.dmatrix(fml, df, NA_action=na_action, return_type='dataframe')
        self.design_info = df_ex.design_info

        # force patsy transform of an F() to int feature back to int not float
        fts_force_to_int = ['intercept']  # also force intercept
        fts_force_to_int = list(self.fts_fact_mapping.keys())
        if len(fts_force_to_int) > 0:
            df_ex[fts_force_to_int] = df_ex[fts_force_to_int].astype(np.int64)

        return df_ex

    def transform(self, df: pd.DataFrame, propagate_nans: bool = False) -> pd.DataFrame:
        """Transform input `df` to dmatrix according to pre-fitted
        `design_info`. Return transformed dmatrix (np.array)

        NEW FUNCTIONALITY: 2021-03-11
            factorize components marked as F(), must be pd.Categorical
        """
        if self.design_info is None:
            raise AttributeError('No design_info, run `fit_transform()` first')

        # # hacky way to get F():F() components
        # fts_ff = set(self.rx_get_ff.findall(self.original_fml))

        # map any features noted in fts_fact_mapping
        try:
            df = df.copy()
            for ft, map_fact_to_int in self.fts_fact_mapping.items():
                df[ft] = df[ft].map(map_fact_to_int).astype(np.int64)
        except AttributeError:
            # self.fts_fact_mapping was never created for this instance
            # simply because no F() in fml
            pass

        # TODO add option to output matrix
        # mx_ex = pt.dmatrix(self.design_info, df, NA_action='raise',
        #                       return_type='matrix')
        # return np.asarray(mx_ex)

        # do nothing, see https://stackoverflow.com/a/51641183/1165112
        na_action = pt.NAAction(NA_types=[]) if propagate_nans else 'raise'

        df_ex = pt.dmatrix(
            self.design_info, df, NA_action=na_action, return_type='dataframe'
        )
        self.design_info = df_ex.design_info

        # force patsy transform of an index feature back to int!
        # there might be a better way to do this
        fts_force_to_int = []
        fts_force_to_int = list(self.fts_fact_mapping.keys())
        if len(fts_force_to_int) > 0:
            df_ex[fts_force_to_int] = df_ex[fts_force_to_int].astype(np.int64)

        return df_ex


class Standardizer:
    """Model-agnostic standardizer from pre-transformed dmatrix
    dmatrix according to patsy formula or patsy design_info object
    NOTES:
    + means, sdevs and scale are stateful: anticipate never changing
    + must be initialised with Transformer.design_info, consider refactoring
    + it's reasonable to initialise this per-observation but far more
        efficient to initialise once and persist in-memory
    + note these standardizations use nanmean, nanstd, so they're
        already compatible with recent changes (2021-03-31) to
        transformer to propagate nans

    NEW FUNCTIONALITY:
    + apply standardization using a mask. allows us to exclude any col
    + rework to I/O dataframes

    TODO: introduce minmax scaling as an option
    """

    def __init__(self, design_info: pt.design_info.DesignInfo, fts_exclude: list = []):
        """Optionally exclude from standardization a list of named fts that
        are numeric and would otherwise get standardardized"""

        self.design_info = design_info
        col_num_excl = [0] + [
            i
            for i, n in enumerate(self.design_info.column_names)
            if (n in fts_exclude) or re.search(r'\[T\.', n)
        ]

        # col_mask is True where we want to exclude the col from standardization
        self.col_mask = [
            True if i in col_num_excl else False
            for i in np.arange(len(self.design_info.column_names))
        ]

        self.means = None  # will become np.ndarray
        self.sdevs = None  # will become np.ndarray
        self.scale = None  # will become int

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize input df to mean-centered, 2sd unit variance,
        Retain the fitted means and sdevs for later use in standardize()
        """
        if any([v is None for v in [self.means, self.sdevs, self.scale]]):
            raise AttributeError(
                'No mns, sdevs or scale, '
                + 'run `fit_standardize()` on training set first'
            )

        df_s = (df - self.means) / (self.sdevs * self.scale)
        mask = np.tile(self.col_mask, (len(df), 1))

        # fill original df w/ standardized to more easily preseve dtype of ints
        df_exs = df.mask(~mask, df_s)  # replace values where condition is True
        return df_exs

    def fit_standardize(self, df: pd.DataFrame, scale: int = 2) -> pd.DataFrame:
        """Standardize numeric features of df with variable scale
        Retain the fitted means and sdevs for later use in standardize()
        """
        self.means = np.where(self.col_mask, np.nan, np.nanmean(df, axis=0))
        self.sdevs = np.where(self.col_mask, np.nan, np.nanstd(df, axis=0))
        self.scale = scale
        return self.standardize(df)

    def standardize_mx(self, mx: np.ndarray) -> np.ndarray:
        """Standardize input mx to mean-centered, 2sd unit variance,
        Retain the fitted means and sdevs for later use in standardize()
        """
        if any([v is None for v in [self.means, self.sdevs, self.scale]]):
            raise AttributeError(
                'No mns, sdevs or scale, '
                + 'run `fit_standardize()` on training set first'
            )

        mxs_all = (mx - self.means) / (self.sdevs * self.scale)
        mask = np.tile(self.col_mask, (len(mx), 1))
        mxs = np.where(mask, mx, mxs_all)
        return mxs

    def fit_standardize_mx(self, mx: np.ndarray, scale: int = 2) -> np.ndarray:
        """Standardize numeric features of mx with variable scale
        Retain the fitted means and sdevs for later use in standardize()
        """
        # TODO add option to output dataframe
        self.means = np.where(self.col_mask, np.nan, np.nanmean(mx, axis=0))
        self.sdevs = np.where(self.col_mask, np.nan, np.nanstd(mx, axis=0))
        self.scale = scale
        return self.standardize(mx)

    def get_scale(self):
        """Get values followuing fit_standardize. Persist values over time."""
        means_sdevs = pd.DataFrame(
            {'means': self.means, 'sdevs': self.sdevs},
            index=self.design_info.column_names,
        )

        return means_sdevs, self.scale

    def set_scale(self, means_sdevs, scale: int = 2):
        """Set values saved from a prior fit_standardize. Now can run
        standardize for new data
        """
        self.means = means_sdevs['means'].values
        self.sdevs = means_sdevs['sdevs'].values
        self.scale = scale


def compress_factor_levels(df: pd.DataFrame, fts: list, topn: int = 20) -> pd.DataFrame:
    """Crude compression for factor levels, into the topn + 1 (other)
    Return new dataframe for fts
    """
    newdf = pd.DataFrame(index=df.index)
    for ft in fts:
        vc = df[ft].value_counts(dropna=False)
        _log.info(f'{ft}: compress {vc[:topn].sum()} ({vc[:topn].sum()/vc.sum():.1%})')
        _log.info(
            f'{ft}: compressed {len(vc)}-{topn} ({len(vc)-topn}) levels '
            + 'into `other`, '
            + f'{vc[topn:].sum()} rows ({vc[topn:].sum() / len(df):.1%}) affected'
        )
        vc_map = {
            k: (k if i < topn else 'other')
            for i, (k, v) in enumerate(vc.to_dict().items())
        }
        newdf[f'{ft}_topn'] = df[ft].map(vc_map)

    return newdf
