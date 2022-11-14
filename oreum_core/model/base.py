# model.base.py
# copyright 2022 Oreum Industries
import arviz as az
import pandas as pd
import pymc3 as pm


class BasePYMC3Model:
    """Base handler to build, sample, store traces for PyMC3 model.
    NOTE:
    + This class is to be inherited e.g. `super().__init__(*args, **kwargs)`
    + Children must declare a _build() method and define obs within __init__()
    + To get prior, posterior, and ppc traces, please use the .idata attribute
    """

    RSD = 42

    def __init__(self, obs: pd.DataFrame = None, **kwargs):
        """Expect obs as dfx pd.DataFrame(mx_en, mx_exs)
        Options for init are often very important!
        https://github.com/pymc-devs/pymc/blob/ed74406735b2faf721e7ebfa156cc6828a5ae16e/pymc3/sampling.py#L277
        """
        self.model = None
        self._idata = None
        self.sample_prior_predictive_kws = dict(draws=500)
        self.sample_posterior_predictive_kws = dict(fast=True, store_ppc=False)
        self.sample_kws = dict(
            init='auto',  # aka jitter+adapt_diag
            random_seed=self.RSD,
            tune=2000,  # NOTE: often need to bump this much higher e.g. 5000
            draws=500,
            chains=4,
            cores=4,
            target_accept=0.8,
            idata_kwargs=None,
        )
        self.rvs_for_posterior_plots = []
        self.name = getattr(self, 'name', 'unnamed_model')
        self.version = getattr(self, 'version', 'unversioned_model')
        self.name = f"{self.name}{kwargs.pop('name_ext', '')}"

    @property
    def idata(self):
        """Returns Arviz InferenceData built from sampling to date"""
        assert self._idata, "Run update_idata() first"
        return self._idata

    def build(self, **kwargs):
        """Build the model"""
        helper_txt = '' if self.model is None else 're'
        try:
            self._build(**kwargs)
            print(f'{helper_txt}built model {self.name} {self.version}')
        except AttributeError:
            raise NotImplementedError(
                'Create a method _build() in your'
                + ' subclass, containing the model definition'
            )

    def extend_build(self, **kwargs):
        """Rebuild and extend build, initially developed to help PPC of GRW"""
        try:
            self._build(**kwargs)
            self._extend_build(**kwargs)
            print(f'extended build of model {self.name} {self.version}')
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
        [issue ticket](https://github.com/pymc-devs/pymc3/issues/4598)
        """
        draws = kwargs.pop('draws', self.sample_prior_predictive_kws['draws'])
        random_seed = kwargs.pop('random_seed', self.sample_kws['random_seed'])

        with self.model:
            prior = pm.sample_prior_predictive(
                samples=draws, random_seed=random_seed, **kwargs
            )
        _ = self.update_idata(prior=prior)
        return None

    def sample(self, **kwargs):
        """Sample posterior: use base class defaults self.sample_kws
        or passed kwargs for pm.sample()
        """
        init = kwargs.pop('init', self.sample_kws['init'])
        random_seed = kwargs.pop('random_seed', self.sample_kws['random_seed'])
        tune = kwargs.pop('tune', self.sample_kws['tune'])
        draws = kwargs.pop('draws', self.sample_kws['draws'])
        chains = kwargs.pop('chains', self.sample_kws['chains'])
        cores = kwargs.pop('cores', self.sample_kws['cores'])
        target_accept = kwargs.pop('target_accept', self.sample_kws['target_accept'])

        with self.model:
            posterior = pm.sample(
                init=init,
                random_seed=random_seed,
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                # step=self.steppers,
                return_inferencedata=False,
                **kwargs,
            )
        _ = self.update_idata(posterior=posterior)
        return None

    def _get_posterior(self):
        """Returns posterior from idata from previous run of sample"""
        try:
            self.idata.posterior
        except AttributeError as e:
            raise e("Run sample() first")
        return self.idata.posterior

    def sample_posterior_predictive(self, **kwargs):
        """Sample posterior predictive
        use self.sample_posterior_predictive_kws or passed kwargs
        Note defaults aimed toward PPC in production
            + Use pm.fast_sample_posterior_predictive()
            + Don't store ppc on model object and just return an updated idata
        """
        random_seed = kwargs.pop('random_seed', self.sample_kws['random_seed'])
        fast = kwargs.pop('fast', self.sample_posterior_predictive_kws['fast'])
        store_ppc = kwargs.pop(
            'store_ppc', self.sample_posterior_predictive_kws['store_ppc']
        )
        # expect n_samples as default None, but allow for exceptional override
        n_samples = kwargs.get('n_samples')

        with self.model:
            if fast:
                ppc = pm.fast_sample_posterior_predictive(
                    trace=self._get_posterior(),
                    random_seed=random_seed,
                    samples=n_samples,
                    **kwargs,
                )
            else:
                ppc = pm.sample_posterior_predictive(
                    trace=self._get_posterior(),
                    random_seed=random_seed,
                    samples=n_samples,
                    **kwargs,
                )

        if store_ppc:
            _ = self.update_idata(posterior_predictive=ppc)
        else:
            return self._create_idata(posterior_predictive=ppc)

    def replace_obs(self, new_obs):
        """Replace the observations
        Assumes data lives in pm.Data contrainers in your _build() fn
        Optionally afterwards call `extend_build()` for future time-dependent PPC
        """
        self.obs = new_obs

    def _create_idata(self, **kwargs):
        """Create Arviz InferenceData object
        NOTE: use ordered exceptions, with assumption that we always use
            an ordered workflow: prior, trace, ppc
        """
        k = dict(model=self.model)

        if (
            idata_kwargs := kwargs.get('idata_kwargs', self.sample_kws['idata_kwargs'])
        ) is not None:
            k.update(**idata_kwargs)

        # update dict with pymc3 outputs
        if (prior := kwargs.get('prior', None)) is not None:
            k['prior'] = prior

        if (ppc := kwargs.get('posterior_predictive', None)) is not None:
            k['posterior_predictive'] = ppc
            # by logic in sample_postrior_predictive there exists self.idata.posterior
        elif (posterior := kwargs.get('posterior', None)) is not None:
            k['trace'] = posterior
        else:
            pass

        idata = az.from_pymc3(**k)

        # extend idata with any other older data
        try:
            idata.extend(self.idata, join='left')
        except AssertionError:
            pass  # idata doesnt exist

        return idata

    def update_idata(self, idata: az.InferenceData = None, **kwargs):
        """Create (and updated) an Arviz InferenceData object on-model
        from current set of self.attributes
        or from a passed-in presampled idata object
        """
        if idata is not None:
            self._idata = idata
        else:
            self._idata = self._create_idata(**kwargs)
        return None
