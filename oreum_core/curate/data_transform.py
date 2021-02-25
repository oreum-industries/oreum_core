# curate.data_transform.py
# copyright 2021 Oreum OÜ
import re
import numpy as np
import pandas as pd
import patsy as pt
import string
from sklearn.model_selection import train_test_split
from .text_clean import SnakeyLowercaser


class DatatypeConverter():
    """ Force correct datatypes according to what model expects 
        Have created this as a class because it knows about the model
        And just reuse the generic function data_cleaner.force_dtypes 
        which you might potentially want to use elsewhere
    """

    def __init__(self, fts, ftslvlcat={}):
        """ Initialise with fts and fts_dtype_pandas_categorical
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
                    ffloat =[])
        """
        self.fts = dict(fid=fts.get('fid', []),
                        fcat=fts.get('fcat', []), 
                        fbool=fts.get('fbool', []),
                        fdate=fts.get('fdate', []),
                        fyear=fts.get('fyear', []),
                        fint=fts.get('fint', []),
                        ffloat=fts.get('ffloat', []))
        self.ftslvlcat = ftslvlcat
        self.rx_number_junk = re.compile(r'[#$€£₤¥,;%]')

    def convert_dtypes(self, df):
        """ Force dtypes for recognised features (fts) in df 
        """
        dfclean = self._force_dtypes(df, **self.fts)

        for ft, lvls in self.ftslvlcat.items():
            dfclean[ft] = pd.Categorical(dfclean[ft].values, categories=lvls, ordered=True)
        
        return dfclean

    def _force_dtypes(self, dfraw, **kwargs):
        """ Select fts and convert dtypes. Return cleaned df 
        """
        snl = SnakeyLowercaser()

        #subselect desired fts
        fts_all = [v for l in kwargs.values() for v in l]
        df = dfraw[fts_all].copy()
        
        for ft in kwargs.get('fid', []) + kwargs.get('fcat', []):
            idx = df[ft].notnull()
            df.loc[idx, ft] = df.loc[idx, ft].astype(str, errors='raise').apply(snl.clean)
            
        for ft in kwargs.get('fbool', []):
            if pd.isnull(df[ft]).sum() == 0:
                df[ft] = df[ft].astype(np.bool)

        for ft in kwargs.get('fyear', []):
            df[ft] = pd.to_datetime(df[ft], errors='raise', format='%Y')

        for ft in kwargs.get('fdate', []):
            df[ft] = pd.to_datetime(df[ft], errors='raise', format='%Y-%m-%d')
            
        for ft in kwargs.get('fint', []):
            if df.dtypes[ft] == np.object:
                df[ft] = df[ft].astype(str).str.strip().str.lower().map(
                            lambda x: self.rx_number_junk.sub('', x))
                df.loc[df[ft].isin(['none', 'nan', 'null', 'na']), ft] = np.nan
            df[ft] = df[ft].astype(np.float64, errors='raise')
            if pd.isnull(df[ft]).sum() == 0:
                df[ft] = df[ft].astype(np.int64, errors='raise')

        for ft in kwargs.get('ffloat', []):
            if df.dtypes[ft] == np.object:
                df[ft] = df[ft].astype(str).str.strip().str.lower().map(
                            lambda x: self.rx_number_junk.sub('', x))
                df.loc[df[ft].isin(['none', 'nan', 'null', 'na']), ft] = np.nan
            df[ft] = df[ft].astype(np.float64, errors='raise')
                
        return df


class DatasetReshaper():
    """ Convenience functions to reshape whole datasets """

    def __init__(self):
        pass

    def create_dfcmb(self, df, fts):
        """ Create a combination dataset `dfcmb` from inputted `df`.
            
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
            dfcmb[ft] = 1.  
        
        print('Reduced df ({} rows, {:,.0f} KB) to dfcmb ({} rows, {:,.0f} KB)'.format(
                df.shape[0], df.values.nbytes / 1e3,
                dfcmb.shape[0], dfcmb.values.nbytes / 1e3))
    
        return dfcmb


    def _create_dfcmb_big(self, df, fts):
        """ Create a combination dataset `dfcmb` from inputted `df`.
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
            dfcmb[ft] = 1.  
        
        print('Reduced df ({} rows, {:,.0f} KB) to dfcmb ({} rows, {:,.0f} KB)'.format(
                df.shape[0], df.values.nbytes / 1e3,
                dfcmb.shape[0], dfcmb.values.nbytes / 1e3))
    
        return dfcmb


    def split_train_test(self, df, stratify_ft=None, test_size=0.2, skip=1, 
                        idx_ids_only=False, random_state=None):
        """ Split `df` into training and test sets, optionally by `stratify_ft` 
        """
        vec = None
        if stratify_ft is not None:
            vec = df.iloc[::skip][stratify_ft]
        
        df_train, df_test = train_test_split(df.iloc[::skip], 
                                    test_size=test_size, stratify=vec,
                                    random_state=random_state)
        if idx_ids_only:
            return df_train.index.values, df_test.index.values
        return df_train, df_test


class Transformer():
    """ Model-agnostic transformer from row-wise natural observations into
        dmatrix according to patsy formula or patsy design_info object
        NOTES: 
            + design_info is stateful
            + it's reasonable to initialise this per-observation but far more 
              efficient to initialise once and persist in-memory
    """

    def __init__(self):
        self.design_info = None
        self.col_idx_numerics = None


    def fit_transform(self, fml, df):
        """ Fit a new design_info attribute for this instance according to 
            `fml` acting upon `df`. Return the transformed dmatrix (np.array)
            Use this for a new training set or to initialise the transfomer
            based on dfcmb.
        """
        mx_ex = pt.dmatrix(fml, df, NA_action='raise', return_type='matrix')
        self.design_info = mx_ex.design_info
        self.col_idx_numerics = 1 + sum([1 for n in self.design_info.column_names 
                                                if re.search(r'\[T\.', n)])
        return np.asarray(mx_ex)


    def transform(self, df):
        """ Transform input `df` to dmatrix according to pre-fitted 
            `design_info`. Return transformed dmatrix (np.array)
        """
        if self.design_info is None:
            raise AttributeError('No design_info, run `fit_transform()` first')
        
        mx_ex = pt.dmatrix(self.design_info, df, NA_action='raise', return_type='matrix')
        return np.asarray(mx_ex)


class Standardizer():
    """ Model-agnostic standardizer from pre-transformed dmatrix
        dmatrix according to patsy formula or patsy design_info object
        NOTES: 
            + means, sdevs and scale are stateful: anticipate never changing
            + must be initialised with Transformer.design_info, consider refactoring
            + it's reasonable to initialise this per-observation but far more 
              efficient to initialise once and persist in-memory
            + stop_standardizing_numerics_at_ft is a dirty hack to force ignore 
              standardizing numerics that you put at the end of the formula
              because you dont want to standardise them, starting at ft name
    """

    def __init__(self, design_info, stop_standardizing_numerics_at_ft=''):
        self.design_info = design_info
        self.stdz_start = (1 + 
                                sum([1 for n in self.design_info.column_names 
                                                if re.search(r'\[T\.', n)]))
        self.stdz_stop = self.design_info.column_name_indexes.get(
                                    stop_standardizing_numerics_at_ft, 
                                    len(self.design_info.column_names))
        self.means = None
        self.sdevs = None
        self.scale = None


    def standardize(self, mx):
        """ Standardize input mx to mean-centered, 2sd unit variance,
            Retain the fitted means and sdevs for later use in standardize()
        """
        if any([v is None for v in [self.means, self.sdevs, self.scale]]):
            raise AttributeError('No mns, sdevs or scale, ' + 
                                 'run `standardize()` on training set first')
        mxs = ((mx[:, self.stdz_start:self.stdz_stop] - self.means) /
               (self.sdevs * self.scale))
        return np.concatenate((mx[:, :self.stdz_start],
                               mxs,
                               mx[:, self.stdz_stop:]), axis=1)


    def fit_standardize(self, mx, scale=2):
        """ Standardize numeric features of mx with variable scale
            Retain the fitted means and sdevs for later use in standardize()
        """
        self.means = np.nanmean(mx[:, self.stdz_start:self.stdz_stop], axis=0)
        self.sdevs = np.nanstd(mx[:, self.stdz_start:self.stdz_stop], axis=0)
        self.scale = scale
        return self.standardize(mx)


    def get_means_sdevs_scale(self):
        """ Get values followuing fit_standardize. Persist values over time.
        """
        means_sdevs = pd.DataFrame({'means': self.means, 'sdevs':self.sdevs}, 
                index=self.design_info.column_names[self.stdz_start:self.stdz_stop])

        return means_sdevs, self.scale
                
                
    def set_means_sdevs_scale(self, means_sdevs, scale=2):
        """ Set values saved from a prior fit_standardize. Now can run 
            standardize for new data
        """
        self.means = means_sdevs['means'].values
        self.sdevs = means_sdevs['sdevs'].values
        self.scale = scale
