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

# model.base.py
"""Base Model Class"""
import logging

import arviz as az
import pymc as pm

__all__ = ['BasePYMCModel']

_log = logging.getLogger(__name__)
_log_pymc = logging.getLogger('pymc')  # force pymc chatty prints to log
_log_pymc.setLevel(logging.WARNING)


class BasePYMCModel:
    """Base handler / wrapper to build, sample, store traces for a pymc model.
    NOTE:
    + This class is to be inherited e.g. `super().__init__(*args, **kwargs)`
    + Children must declare a _build() method and define obs within __init__()
    + To get prior, posterior, and ppc traces, please use the .idata attribute
    """

    rsd = 42

    def __init__(self, **kwargs):
        """Expect obs as dfx pd.DataFrame(mx_en, mx_exs)
        Options for init are often very important!
        https://github.com/pymc-devs/pymc/blob/ed74406735b2faf721e7ebfa156cc6828a5ae16e/pymc/sampling.py#L277
        Usage note: override kws directly on downstream instance e.g.
        def __init__(self, **kwargs):
            super().__init__(*args, **kwargs)
            self.sample_kws.update(dict(tune=1000, draws=500, target_accept=0.85))
        """
        self.model = None
        self._idata = None
        self.sample_prior_predictive_kws = dict(samples=2000)
        self.sample_posterior_predictive_kws = dict(fast=True, store_ppc=False)
        self.sample_kws = dict(
            init='auto',  # aka jitter+adapt_diag
            tune=2000,  # NOTE: often need to bump this much higher e.g. 5000
            draws=500,
            chains=4,
            cores=4,
            target_accept=0.8,
            idata_kwargs=None,
            progressbar=True,
        )
        self.rvs_for_posterior_plots = []
        self.name = getattr(self, 'name', 'unnamed_model')
        self.version = getattr(self, 'version', 'unversioned_model')
        self.name = f"{self.name}{kwargs.pop('name_ext', '')}"

    def _get_posterior(self):
        """Returns posterior from idata from previous run of sample"""
        try:
            self.idata.posterior
        except AttributeError as e:
            raise e("Run sample() first")
        return self.idata.posterior

    @property
    def idata(self):
        """Returns Arviz InferenceData built from sampling to date"""
        assert self._idata, "Run update_idata() first"
        return self._idata

    def build(self, **kwargs):
        """Build the model"""
        helper_txt = 'B' if self.model is None else 'Reb'
        try:
            self._build(**kwargs)
            _log.info(f'{helper_txt}uilt model {self.name} {self.version}')
        except AttributeError:
            raise NotImplementedError(
                'Create a method _build() in your'
                + ' subclass, containing the model definition'
            )

    def extend_build(self, **kwargs):
        """Extend build, initially developed to help PPC of GRW and missing value imputation"""
        try:
            self._extend_build(**kwargs)
            _log.info(f'Extended build of model {self.name} {self.version}')
        except AttributeError:
            raise NotImplementedError(
                'Create a method _extend_build() in your'
                + ' subclass, containing the model extension definition'
            )

    def sample_prior_predictive(self, **kwargs):
        """Sample prior predictive:
            use base class defaults self.sample_prior_predictive_kws
            or passed kwargs for pm.sample_prior_predictive()
        NOTE:
        + It's not currently possible to run Prior Predictive Checks on a
          model with missing values, per my detailed
          [MRE Notebook gist](https://gist.github.com/jonsedar/070319334bcf033773cc3e9495c79ea0)
          that illustrates the issue.
        + I have created and tested a fix as described in my
        [issue ticket](https://github.com/pymc-devs/pymc/issues/4598)
        """
        kws = dict(
            random_seed=kwargs.pop('random_seed', self.rsd),
            samples=kwargs.pop('samples', self.sample_prior_predictive_kws['samples']),
        )

        with self.model:
            prior = pm.sample_prior_predictive(**{**kws, **kwargs})
        _ = self.update_idata(prior)
        _log.info(f'Sampled prior predictive for {self.name} {self.version}')
        return None

    def sample(self, **kwargs):
        """Sample posterior: use base class defaults self.sample_kws
        or passed kwargs for pm.sample()
        """
        kws = dict(
            init=kwargs.pop('init', self.sample_kws['init']),
            random_seed=kwargs.pop('random_seed', self.rsd),
            tune=kwargs.pop('tune', self.sample_kws['tune']),
            draws=kwargs.pop('draws', self.sample_kws['draws']),
            chains=kwargs.pop('chains', self.sample_kws['chains']),
            cores=kwargs.pop('cores', self.sample_kws['cores']),
            progressbar=kwargs.pop('progressbar', self.sample_kws['progressbar']),
        )

        target_accept = kwargs.pop('target_accept', self.sample_kws['target_accept'])
        step = kwargs.pop('step', 'nuts')

        with self.model:
            common_stepper_options = {
                'nuts': pm.NUTS(target_accept=target_accept),
                'metropolis': pm.Metropolis(target_accept=target_accept),
                'advi': pm.ADVI(),
            }
            kws['step'] = common_stepper_options.get(step, None)

            try:
                posterior = pm.sample(**{**kws, **kwargs})
            except UserWarning as e:
                _log.warning('Warning in sample()', exc_info=e)
            finally:
                _ = self.update_idata(posterior)

        _log.info(f'Sampled posterior for {self.name} {self.version}')
        return None

    def sample_posterior_predictive(self, **kwargs):
        """Sample posterior predictive
        use self.sample_posterior_predictive_kws or passed kwargs
        Note defaults aimed toward PPC in production
            + Use pm.fast_sample_posterior_predictive()
            + Don't store ppc on model object and just return an updated idata
        """
        fast = kwargs.pop('fast', self.sample_posterior_predictive_kws['fast'])
        store_ppc = kwargs.pop(
            'store_ppc', self.sample_posterior_predictive_kws['store_ppc']
        )
        kws = dict(
            trace=self._get_posterior(),
            random_seed=kwargs.pop('random_seed', self.rsd),
            samples=kwargs.pop('samples', None),
        )

        with self.model:
            if fast:
                ppc = pm.fast_sample_posterior_predictive(**{**kws, **kwargs})
            else:
                ppc = pm.sample_posterior_predictive(**{**kws, **kwargs})

        _log.info(f'Sampled ppc for {self.name} {self.version}')

        if store_ppc:
            _ = self.update_idata(ppc)
        else:
            return ppc

    def replace_obs(self, new_obs) -> None:
        """Replace the observations
        Assumes data lives in pm.Data containers in your _build() function
        You must call `build()` afterward
        Optionally afterwards call `extend_build()` for future time-dependent PPC
        """
        self.obs = new_obs
        _log.info(f'Replaced obs in {self.name} {self.version}')

    def update_idata(self, idata: az.InferenceData, replace: bool = False) -> None:
        """Create (and update) an Arviz InferenceData object on-model from a
        passed-in presampled InferenceData object
        """
        if self._idata is None:
            self._idata = idata
        else:
            side = 'right' if replace else 'left'
            self._idata = self.idata.extend(idata, join=side)
