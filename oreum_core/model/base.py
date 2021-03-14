# model.base.py
# copyright 2021 Oreum OÜ
import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm


class BasePYMC3Model():
    """ Handle the build, sample and exploration of a PyMC3 model"""

    random_seed = 42

    def __init__(self, obs:pd.DataFrame=None, **kwargs):
        """ Expect obs as dfx pd.DataFrame(mx_en, mx_exs) """
        self.model = None
        self._trace = None
        self._trace_prior = None
        self._inference_data = None
        self.obs = obs
        self.sample_prior_predictive_kws = dict(samples=500)
        self.sample_kws = dict(init='jitter+adapt_diag', 
                            random_seed=self.random_seed, tune=1000, draws=500, 
                            chains=4, cores=4, target_accept=0.8)
        self.rvs_for_posterior_plots = []

    @property
    def n_divergences(self):
        """ Returns the number of divergences from the current trace """
        assert self._trace != None, "Must run sample() first!"
        return self._trace["diverging"].nonzero()[0].size

    # not needed, see oreum_core.model.create_azid
    # @property
    # def inference_data(self):
    #     """ Returns an Arviz InferenceData object """
    #     assert self._trace, "Must run sample() first!"

    #     with self.model:
    #         posterior_predictive = pm.sample_posterior_predictive(self.trace)

    #     _inference_data = az.from_pymc3(
    #         trace=self._trace,
    #         posterior_predictive=posterior_predictive,
    #     )
    #     _inference_data.posterior.attrs["model_version"] = self.version

    #     return _inference_data

    @property
    def trace_prior(self):
        """ Returns trace_prior from a previous sample_prior_predictive() """
        assert self._trace_prior, "Must run sample_prior_predictive() first!"
        return self._trace_prior

    @property
    def trace(self):
        """ Returns trace from a previous sample() """
        assert self._trace, "Must run sample() first!"
        return self._trace

    @property
    def posterior_predictive(self):
        """ Returns the posterior predictive from a previous sample_posterior_predictive() """
        assert self._posterior_predictive, "Must run sample_posterior_predictive() first!"
        return self._posterior_predictive   


    def build(self):
        try:
            self._build()
            print(f'Built model {self.name}')
        except AttributeError:
            raise NotImplementedError('You must create a method _build()' + 
                            ' in your subclass, containing your model definition')
        

    def sample_prior_predictive(self, **kwargs):
        """ Sample prior predictive, use base class defaults 
            self.sample_prior_predictive_kws or passed kwargs for
            pm.sample_prior_predictive()
        """
        samples = kwargs.get('samples', self.sample_prior_predictive_kws['samples'])
        random_seed = kwargs.get('random_seed', self.sample_kws['random_seed'])
        
        if self.model is None:
            self.build()

        with self.model:
            self._trace_prior = pm.sample_prior_predictive(samples=samples, 
                                            random_seed=random_seed, **kwargs)
        return None


    def sample(self, **kwargs):
        """ Sample posterior, use base class defaults self.sample_kws
            or passed pm.sample() kwargs 
        """
        init = kwargs.get('init', self.sample_kws['init'])
        random_seed = kwargs.get('random_seed', self.sample_kws['random_seed'])
        tune = kwargs.get('tune', self.sample_kws['tune'])
        draws = kwargs.get('draws', self.sample_kws['draws'])
        chains = kwargs.get('chains', self.sample_kws['chains'])
        cores = kwargs.get('cores', self.sample_kws['cores'])
        target_accept = kwargs.get('target_accept', self.sample_kws['target_accept'])

        if self.model is None:
            self.build()

        with self.model:
            self._trace = pm.sample(init=init, random_seed=random_seed,
                            tune=tune, draws=draws, chains=chains, cores=cores,
                            target_accept=target_accept, 
                            return_inferencedata=False, **kwargs)
                            # TODO consider return_inferencedata=True
        return None 


    def sample_posterior_predictive(self, fast=False, **kwargs):
        """ Sample posterior predictive, use base class defaults 
            self.sample_posterior_predictive_kws or passed kwargs for
            pm.{fast}sample_posteriors_predictive()
        """
        random_seed = kwargs.get('random_seed', self.sample_kws['random_seed'])
        
        if self.model is None:
            self.build()

        with self.model:
            if fast:
                self._posterior_predictive = pm.fast_sample_posterior_predictive(
                                self.trace, random_seed=random_seed, **kwargs)
            else:
                self._posterior_predictive = pm.sample_posterior_predictive(
                                self.trace, random_seed=random_seed, **kwargs)
        return None 


    def replace_obs(self, new_obs):
        self.obs = new_obs
