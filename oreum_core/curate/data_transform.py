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

# curate.data_transform.py
"""Data Transformations"""
import logging
import re

import numpy as np
import pandas as pd
import patsy as pat

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

    def __init__(self, ftsd: dict, date_format: str = '%Y-%m-%d'):
        """Initialise with fts and optionally specify factors with specific levels

        Use with a fts dict of form:
            ftsd = dict(
                fcat = [],           # use for unordered categoricals
                ford = {ft0: lvls0, ft1: lvls:1, ... },  # use for ordinals
                fstr = [],  # rarely used e.g. for for freetext strings
                fbool = [],
                fbool_nan_to_false = [],
                fdate = [],
                fyear = [],
                fint = [],
                ffloat = [],
                fverbatim = [],        # maintain in current dtype)

        Use ftsordlvl for ordinal categoricals
        """
        self.ftsd = dict(
            fcat=ftsd.get('fcat', []),
            ford=ftsd.get('ford', {}),
            fstr=ftsd.get('fstr', []),
            fbool=ftsd.get('fbool', []),
            fbool_nan_to_false=ftsd.get('fbool_nan_to_false', []),
            fdate=ftsd.get('fdate', []),
            fyear=ftsd.get('fyear', []),
            fint=ftsd.get('fint', []),
            ffloat=ftsd.get('ffloat', []),
            fverbatim=ftsd.get('fverbatim', []),  # keep verbatim
        )
        self.rx_number_junk = re.compile(r'[#$€£₤¥,;%\s]')
        self.date_format = date_format
        inv_bool_dict = {
            True: ['yes', 'y', 'true', 't', '1', 1, 1.0],
            False: ['no', 'n', 'false', 'f', '0', 0, 0.0],
        }
        self.bool_dict = {v: k for k, vs in inv_bool_dict.items() for v in vs}
        self.strnans = [
            'none',
            'nan',
            'null',
            'na',
            'n/a',
            '<na>',
            'missing',
            'empty',
            '',
        ]

    def convert_dtypes(self, dfraw: pd.DataFrame) -> pd.DataFrame:
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
            df.loc[~idx, ft] = np.nan
            df.loc[idx, ft] = vals
            if ft in self.ftsd['fcat']:
                df[ft] = pd.Categorical(df[ft].values, ordered=False)
            else:
                df[ft] = df[ft].astype('string')

        for ft, lvls in self.ftsd['ford'].items():
            df[ft] = pd.Categorical(df[ft].values, categories=lvls, ordered=True)

        for ft in self.ftsd['fbool'] + self.ftsd['fbool_nan_to_false']:
            # tame string, strip, lower, use self.bool_dict, use pd.NA
            if df.dtypes[ft] == object:
                df[ft] = df[ft].apply(lambda x: str(x).strip().lower())
                df.loc[df[ft].isin(self.strnans), ft] = pd.NA
                df[ft] = df[ft].apply(lambda x: self.bool_dict.get(x, x))

                if ft in self.ftsd['fbool_nan_to_false']:
                    df.loc[df[ft].isnull(), ft] = False

                set_tf_only = set(df[ft].unique())
                if set_tf_only in set([True, False]):  # most common, use np.bool
                    df[ft] = df[ft].astype(bool)
                elif pd.isnull(df[ft]).sum() > 0:  # contains NaNs, use pd.boolean
                    df[ft] = df[ft].convert_dtypes(convert_boolean=True)
                else:
                    # ft not yet properly mapped,
                    raise ValueError(
                        f"{ft} contains values incompatible with np.bool or pd.Boolean"
                    )

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

        return df[fts_all]


class DatasetReshaper:
    """Convenience functions to reshape whole datasets"""

    def __init__(self):
        pass

    def create_dfcmb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a combination dataset `dfcmb` from inputted `df`, which
        MUST have correct / appropriate pandas datatypes.

        Datatypes must be understood by patsy: factor-values as
        pd.Categoricals, or bools, or ints or floats. No dates or strings

        Output `dfcmb` is NOT a big groupby (i.e cartesian join) of factor
        values. Instead, it's a compact concat of unique factor values
        (which can be horizontally ragged so we fill NaNs) and filler values
        for ints a floats

        We use `dfcmb` because as of the current patsy (0.5.1), design_info
        objects still can't be pickled, which means that when you want to
        transform a test / holdout sample according to a training dataset
        you have to instantiate a new design_info object in memory.

        In this codebase we set the design_info attribute on the
        Transformer object, so you can keep that object around in memory,
        but upon memory-loss you need to call it with the dfcmb created
        here e.g. Transformer.fit_transform(dfcmb).

        I recommend storing dfcmb in a DB: it only needs to be updated if
        the modelled features change (e.g. include a new feature) or if
        factor values (aka categorical levels) change
        """
        dfcmb = pd.DataFrame(index=[0])
        sdtypes = df.dtypes

        if (sum(sdtypes == 'object') > 0) | (sum(sdtypes == 'boolean') > 0):
            return (
                ValueError,
                "Valid dtypes are `category`, `bool`, `int`, `float` only",
            )
        cats = list(sdtypes.loc[sdtypes == 'category'].index.values)
        bools = list(sdtypes.loc[sdtypes == 'bool'].index.values)
        ints = list(sdtypes.loc[sdtypes == 'int'].index.values)
        floats = list(sdtypes.loc[sdtypes == 'float'].index.values)

        # create ragged cats
        for ft in cats:
            colnames_pre = list(dfcmb.columns.values)
            dfcmb = pd.concat(
                [dfcmb, pd.Series(df[ft].cat.categories)],
                axis=1,
                join='outer',
                ignore_index=True,
            )
            dfcmb.columns = colnames_pre + [ft]

        # apply categorical dtype again
        for ft in cats:
            dfcmb[ft] = pd.Categorical(
                dfcmb[ft].values,
                categories=df[ft].cat.categories,
                ordered=df[ft].cat.ordered,
            )

        for ft in bools:
            colnames_pre = list(dfcmb.columns.values)
            dfcmb = pd.concat(
                [dfcmb, pd.Series([False, True])],
                axis=1,
                join='outer',
                ignore_index=True,
            )
            dfcmb.columns = colnames_pre + [ft]

        for ft in bools:  # force to bool (we choose to not allow NaNs in bools)
            # dfcmb[ft] = dfcmb[ft].convert_dtypes(convert_boolean=True)
            dfcmb[ft] = dfcmb[ft].astype(bool)

        for ft in ints:
            dfcmb[ft] = 1

        for ft in floats:
            dfcmb[ft] = 1.0

        _log.info(
            f'Reduced df {len(df)} rows, {df.values.nbytes / 1e3:,.0f} KB)'
            + f' to dfcmb ({len(dfcmb)} rows, {dfcmb.values.nbytes / 1e3:,.0f} KB)'
        )

        return dfcmb


class Transformer:
    """Model-agnostic patsy transformer from row-wise natural observations
    into dmatrix according to patsy formula or patsy design_info object
    NOTES:
        + design_info is stateful
        + it's reasonable to initialise this per-observation but far more
          efficient to initialise once and persist in-memory
        + Categoricals must already be a pd.Categorical with all levels assigned
          appropriate to the full data domain. This will feed
          pd.Categorical.cat.codes (ints) representation into patsy
        + Booleans must already be np.bool (not pd.BooleanDtype, no NaNs)
    """

    def __init__(self):
        self.design_info = None
        self.factor_map = {}
        self.snl = SnakeyLowercaser()

    def _get_fts_to_force_to_int(self, dfraw: pd.DataFrame) -> list[str]:
        """Get list of ffts to force to int post patsy conversion"""
        s = dfraw.dtypes
        ints = list(s.loc[s == 'int'].index.values)
        bools = [f'{ft}[T.True]' for ft in s.loc[s == 'bool'].index.values]
        return ints + bools

    def _convert_cats(self, dfraw: pd.DataFrame) -> pd.DataFrame:
        """Common conversion of cats to codes and store mapping
        NOTE Missing values (np.nan) are recorded as cats.codes = -1
        which is helpfully still an int, so we can still return int dtype"""
        sdtypes = dfraw.dtypes
        df = dfraw.copy()
        cats = list(sdtypes.loc[sdtypes == 'category'].index.values)
        for ft in cats:
            map_int_to_fct = dict(enumerate(df[ft].cat.categories))
            map_fct_to_int = {v: k for k, v in map_int_to_fct.items()}
            self.factor_map[ft] = map_fct_to_int
            df[ft] = df[ft].cat.codes.astype(int)
        return df

    def _transform(
        self,
        fml_or_design_info: str | pat.DesignInfo,
        df: pd.DataFrame,
        propagate_nans: bool,
    ) -> pd.DataFrame:
        """Common to fit_transform and transform"""
        # TODO add check for fml contains `~` and handle accordingly
        # TODO add option to output matrix   # np.asarray(mx_ex)
        df = self._convert_cats(df.copy())
        fts_force_to_int = self._get_fts_to_force_to_int(df)
        na_act = pat.NAAction(NA_types=[]) if propagate_nans else 'raise'
        df_ex = pat.dmatrix(
            fml_or_design_info, df, NA_action=na_act, return_type='dataframe'
        )
        design_info = df_ex.design_info

        for ft in fts_force_to_int:
            if ft in df_ex.columns.values:
                df_ex[ft] = df_ex[ft].astype(int)

        return df_ex, design_info

    def fit_transform(
        self, fml: str, df: pd.DataFrame, propagate_nans: bool = True
    ) -> pd.DataFrame:
        """Fit a new `design_info` attribute for this instance according to
        `fml` acting upon `df`. Return transformed dmatrix as a DataFrame.
        Use this for a new training set or to initialise the transformer
        based on dfcmb.
        `fml` maps directly to feature names in `df` i.e. before patsy
        transforms and we take the dtypes from `df`.
        """
        df_ex, design_info = self._transform(fml, df, propagate_nans)
        self.design_info = design_info
        return df_ex

    def transform(self, df: pd.DataFrame, propagate_nans: bool = True) -> pd.DataFrame:
        """Transform input `df` to dmatrix according to pre-fitted
        `design_info`. Return transformed dmatrix (pd.DataFrame)
        """
        if self.design_info is None:
            raise AttributeError('No design_info, run `fit_transform()` first')

        df_ex, _ = self._transform(self.design_info, df, propagate_nans)
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
    + Automatically exclude post-patsy binary [T.x] features

    NEW FUNCTIONALITY:
    + apply standardization using a mask. allows us to exclude any col
    + rework to I/O dataframes

    TODO: introduce minmax scaling as an option
    """

    def __init__(self, tfmr: Transformer, fts_exclude: list = []):
        """Automatically exclude fts in list(tfmr.factor_map.keys()) and
        any named in `fts_exclude` that are numeric and would otherwise get
        standardardized"""

        self.design_info = tfmr.design_info
        self.fts_exclude = fts_exclude + list(tfmr.factor_map.keys())

        col_num_excl = [0] + [
            i
            for i, n in enumerate(self.design_info.column_names)
            if (n in self.fts_exclude) or re.search(r'\[T\.', n)
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

    def get_scale(self) -> tuple[pd.DataFrame, float]:
        """Get values followuing fit_standardize. Persist values over time."""
        means_sdevs = pd.DataFrame(
            {'means': self.means, 'sdevs': self.sdevs},
            index=self.design_info.column_names,
        )

        return means_sdevs, self.scale

    def set_scale(self, df_means_sdevs: pd.DataFrame, scale: int = 2):
        """Set values saved from a prior fit_standardize. Now can run
        standardize for new data
        """
        self.means = df_means_sdevs['means'].values
        self.sdevs = df_means_sdevs['sdevs'].values
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
