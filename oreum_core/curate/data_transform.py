# curate.data_transform.py
# copyright 2022 Oreum Industries
import re

import numpy as np
import pandas as pd
import patsy as pt
from sklearn.model_selection import train_test_split

from .text_clean import SnakeyLowercaser

__all__ = [
    'DatatypeConverter',
    'DatasetReshaper',
    'Transformer',
    'Standardizer',
    'compress_factor_levels',
]


class DatatypeConverter:
    """Force correct datatypes according to what model expects"""

    def __init__(self, fts, ftslvlcat={}, date_format='%Y-%m-%d'):
        """Initialise with fts and fts_dtype_pandas_categorical
        The pandas categorical dtype logically sits on top of a str object
        giving it order which is critical for patsy dmatrix transform
        and thus model structure.

        Use with a fts dict of form:
            fts = dict(
                fid = [],
                fcat = [],
                fbool = [],
                fdate = [],
                fyear = [],
                fint = [],
                ffloat =[],
                fverbatim = [],        # maintain in current dtype)
        """
        self.fts = dict(
            fid=fts.get('fid', []),
            fcat=fts.get('fcat', []),
            fbool=fts.get('fbool', []),
            fdate=fts.get('fdate', []),
            fyear=fts.get('fyear', []),
            fint=fts.get('fint', []),
            ffloat=fts.get('ffloat', []),
            fverbatim=fts.get('fverbatim', []),  # keep verbatim
        )
        self.ftslvlcat = ftslvlcat
        self.rx_number_junk = re.compile(r'[#$€£₤¥,;%]')
        self.date_format = date_format
        inv_bool_dict = {
            True: ['yes', 'y', 'true', 't', '1', 1],
            False: ['no', 'n', 'false', 'f', '0', 0],
        }
        self.bool_dict = {v: k for k, vs in inv_bool_dict.items() for v in vs}
        self.strnans = ['none', 'nan', 'null', 'na', 'n/a', 'missing', 'empty']

    def _force_dtypes(self, dfraw):
        """Select fts and convert dtypes. Return cleaned df"""
        snl = SnakeyLowercaser()

        # subselect desired fts
        fts_all = [w for k, v in self.fts.items() for w in v]
        df = dfraw[fts_all].copy()

        for ft in self.fts['fid'] + self.fts['fcat']:
            idx = df[ft].notnull()
            df.loc[idx, ft] = (
                df.loc[idx, ft].astype(str, errors='raise').apply(snl.clean)
            )

        for ft in self.fts['fbool']:
            # tame string, strip, lower, use self.bool_dict, use pd.NA
            if df.dtypes[ft] == object:
                df[ft] = df[ft].apply(lambda x: str(x).strip().lower())
                df.loc[df[ft].isin(self.strnans), ft] = np.nan
                df[ft] = df[ft].apply(lambda x: self.bool_dict.get(x, x))

                if set(df[ft].unique()) != set([True, False, np.nan]):
                    # avoid converting anything not yet properly mapped
                    continue

            # convert string representation of {'0', '1'}
            # if set(df[ft].unique()) == set(['0', '1']):
            #     df[ft] = df[ft].astype(float, errors='raise')
            # if pd.isnull(df[ft]).sum() == 0:
            #     df[ft] = df[ft].astype(bool)
            df[ft] = df[ft].convert_dtypes(convert_boolean=True)

        for ft in self.fts['fyear']:
            df[ft] = pd.to_datetime(df[ft], errors='raise', format='%Y')

        for ft in self.fts['fdate']:
            df[ft] = pd.to_datetime(df[ft], errors='raise', format=self.date_format)

        for ft in self.fts['fint']:
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

        for ft in self.fts['ffloat']:
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

        # TODO as/when add logging
        # for ft in self.fts['fverbatim]:
        #    log(f'kept verbatim: {ft}')

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

        print(
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

        print(
            'Reduced df ({} rows, {:,.0f} KB) to dfcmb ({} rows, {:,.0f} KB)'.format(
                df.shape[0],
                df.values.nbytes / 1e3,
                dfcmb.shape[0],
                dfcmb.values.nbytes / 1e3,
            )
        )

        return dfcmb

    def split_train_test(
        self,
        df: pd.DataFrame,
        stratify_ft=None,
        test_size=0.2,
        skip=1,
        idx_ids_only=False,
        random_state=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split `df` into training and test sets, optionally by `stratify_ft`"""
        vec = None
        if stratify_ft is not None:
            vec = df.iloc[::skip][stratify_ft]

        df_train, df_test = train_test_split(
            df.iloc[::skip],
            test_size=test_size,
            stratify=vec,
            random_state=random_state,
        )
        if idx_ids_only:
            return df_train.index.values, df_test.index.values
        return df_train, df_test


class Transformer:
    """Model-agnostic patsy transformer from row-wise natural observations
    into dmatrix according to patsy formula or patsy design_info object
    NOTES:
        + design_info is stateful
        + it's reasonable to initialise this per-observation but far more
          efficient to initialise once and persist in-memory
    """

    def __init__(self):
        self.design_info = None
        self.col_idx_numerics = None
        self.rx_get_f_components = re.compile(r'(F\(([a-z0-9_]+?)\))')
        self.fts_fact_mapping = {}

    def fit_transform(
        self, fml: str, df: pd.DataFrame, propagate_nans: bool = False
    ) -> pd.DataFrame:
        """Fit a new design_info attribute for this instance according to
        `fml` acting upon `df`. Return the transformed dmatrix (np.array)
        Use this for a new training set or to initialise the transfomer
        based on dfcmb.

        NEW FUNCTIONALITY: 2021-03-11
            factorize components marked as F(), must be pd.Categorical
        """
        # deal with any fml components marked F()
        fts_fact = self.rx_get_f_components.findall(fml)
        if len(fts_fact) > 0:
            df = df.copy()
            for ft_fact in fts_fact:
                dt = df[ft_fact[1]].dtype.name
                if dt != 'category':
                    raise AttributeError(
                        f'fml contains F({ft_fact[1]}), '
                        + 'dtype={dt}, but it must be categorical'
                    )
                # map feature to int based on its preexisting catgorical order
                # https://stackoverflow.com/a/55304375/1165112
                map_int_to_fact = dict(enumerate(df[ft_fact[1]].cat.categories))
                map_fact_to_int = {v: k for k, v in map_int_to_fact.items()}
                self.fts_fact_mapping[ft_fact[1]] = map_fact_to_int
                df[ft_fact[1]] = df[ft_fact[1]].map(map_fact_to_int).astype(np.int)

                # replace F() in fml so that patsy can work as normal
                # with our new int type feature
                fml = fml.replace(ft_fact[0], ft_fact[1])

        # TODO add option to output matrix   # np.asarray(mx_ex)
        # TODO add check for fml contains `~` and handle accordingly
        na_action = 'raise'
        if propagate_nans:
            # do nothing, see https://stackoverflow.com/a/51641183/1165112
            na_action = pt.NAAction(NA_types=[])

        df_ex = pt.dmatrix(fml, df, NA_action=na_action, return_type='dataframe')
        self.design_info = df_ex.design_info

        # force patsy transform of an index feature back to int!
        # there might be a better way to do this
        fts_force_to_int = list(self.fts_fact_mapping.keys())
        print(fml)
        print(df_ex)
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

        na_action = 'raise'
        if propagate_nans:
            # do nothing, see https://stackoverflow.com/a/51641183/1165112
            na_action = pt.NAAction(NA_types=[])

        df_ex = pt.dmatrix(
            self.design_info, df, NA_action=na_action, return_type='dataframe'
        )
        self.design_info = df_ex.design_info

        # force patsy transform of an index feature back to int!
        # there might be a better way to do this
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
        from standardization
    + rework to I/O dataframes
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

        self.means = None
        self.sdevs = None
        self.scale = None

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

    def get_means_sdevs_scale(self):
        """Get values followuing fit_standardize. Persist values over time."""
        means_sdevs = pd.DataFrame(
            {'means': self.means, 'sdevs': self.sdevs},
            index=self.design_info.column_names,
        )

        return means_sdevs, self.scale

    def set_means_sdevs_scale(self, means_sdevs, scale: int = 2):
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
        # print(f'{ft}: compress {vc[:topn].sum()} ({vc[:topn].sum()/vc.sum():.1%})')
        print(
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
