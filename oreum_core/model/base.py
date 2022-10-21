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
    """

    RSD = 42

    def __init__(self, obs: pd.DataFrame = None, **kwargs):
        """Expect obs as dfx pd.DataFrame(mx_en, mx_exs)
        Options for init are often very important!
        https://github.com/pymc-devs/pymc/blob/ed74406735b2faf721e7ebfa156cc6828a5ae16e/pymc3/sampling.py#L277
        """
        self.model = None
        self._trace_prior = None
        self._trace = None
        self._posterior_predictive = None
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
    def trace_prior(self):
        """Returns trace_prior from a previous run of
        sample_prior_predictive()
        """
        assert self._trace_prior, "Run sample_prior_predictive() first"
        return self._trace_prior

    @property
    def trace(self):
        """Returns trace from a previous sample()"""
        assert self._trace, "Must run sample() first!"
        return self._trace

    @property
    def posterior_predictive(self):
        """Returns the posterior predictive from a previous run of
        sample_posterior_predictive()
        """
        assert self._posterior_predictive, "Run sample_posterior_predictive() first"
        return self._posterior_predictive

    @property
    def idata(self):
        """Returns Arviz InferenceData built from sampling to date"""
        assert self._idata, "Run update_idata() first"
        return self._idata

    @property
    def n_divergences(self):
        """Returns the number of divergences from the current trace"""
        assert self._trace, "Must run sample() first!"
        return self._trace["diverging"].nonzero()[0].size

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
        """Sample prior predictive, use base class defaults
        self.sample_prior_predictive_kws or passed kwargs for
        pm.sample_prior_predictive()

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
            self._trace_prior = pm.sample_prior_predictive(
                samples=draws, random_seed=random_seed, **kwargs
            )

        _ = self._update_idata()
        return None

    def sample(self, **kwargs):
        """Sample posterior, use base class defaults self.sample_kws
        or passed pm.sample() kwargs
        """
        init = kwargs.pop('init', self.sample_kws['init'])
        random_seed = kwargs.pop('random_seed', self.sample_kws['random_seed'])
        tune = kwargs.pop('tune', self.sample_kws['tune'])
        draws = kwargs.pop('draws', self.sample_kws['draws'])
        chains = kwargs.pop('chains', self.sample_kws['chains'])
        cores = kwargs.pop('cores', self.sample_kws['cores'])
        target_accept = kwargs.pop('target_accept', self.sample_kws['target_accept'])

        with self.model:
            self._trace = pm.sample(
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
        # TODO can put idata_kwargs into sample() above as/when allow return_inferencedata=True
        _ = self._update_idata()
        return None

    def sample_posterior_predictive(self, **kwargs):
        """Sample posterior predictive
        use self.sample_posterior_predictive_kws or passed kwargs
        Note defaults aimed toward PPC in production
            + Use pm.fast_sample_posterior_predictive()
            + Don't store ppc on model object and just return a new azid
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
                    self.trace, random_seed=random_seed, samples=n_samples, **kwargs
                )
            else:
                ppc = pm.sample_posterior_predictive(
                    self.trace, random_seed=random_seed, samples=n_samples, **kwargs
                )

        if store_ppc is False:
            # the default as expected for forward-pass stateless prediction
            return self._create_idata(ppc)
        else:
            self._posterior_predictive = ppc
            _ = self._update_idata()
            return None

    def replace_obs(self, new_obs):
        """Replace the observations
        Assumes data lives in pm.Data contrainers in your _build() fn
        Optionally afterwards call `extend_build()` for future time-dependent PPC
        """
        self.obs = new_obs

    def _create_idata(self, ppc=None):
        """Create Arviz InferenceData object
        NOTE: use ordered exceptions, with assumption that we always use
            an ordered workflow: prior, trace, ppc
        """
        idata_kwargs = self.sample_kws['idata_kwargs']

        k = dict(model=self.model)
        if idata_kwargs is not None:
            k.update(**idata_kwargs)

        if ppc is not None:
            k['posterior_predictive'] = ppc
        else:
            try:
                k['prior'] = self.trace_prior
            except AssertionError:
                pass
            try:
                k['trace'] = self.trace
                k['posterior_predictive'] = self.posterior_predictive
            except AssertionError:
                pass
        idata = az.from_pymc3(**k)
        return idata

    def _update_idata(self):
        # raise ValueError(kwargs)
        """Create and update Arviz InferenceData object on-model"""
        self._idata = self._create_idata()
        return None


# TODO save and check cache e.g
# https://discourse.pymc.io/t/jupyter-idiom-for-cached-results/6782
# idata_file = "myfilename.nc"
# if os.path.exists(idata_file):
# idata = az.from_netcdf(idata_file)
# else:
# idata = <some expensive computation>
# if not os.path.exists(idata_file):
# az.to_netcdf(idata, idata_file)
# also:
# _inference_data.posterior.attrs["model_version"] = self.version
