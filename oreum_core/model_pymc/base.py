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

# model.base.py
"""Base Model Class"""

import logging
import re

import arviz as az
import pymc as pm
import xarray as xr
from pymc.testing import assert_no_rvs

from .calc import compute_log_likelihood_for_potential
from ..utils.snakey_lowercaser import SnakeyLowercaser

__all__ = ["BasePYMCModel"]

_log = logging.getLogger(__name__)
_log_pymc = logging.getLogger("pymc")  # force pymc chatty prints to log
_log_pymc.setLevel(logging.ERROR)
# logging.captureWarnings(True) # further force chatty pymc warnings to log (py.warnings)


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
        Usage note: override kwargs directly on downstream instance e.g.
        def __init__(self, **kwargs):
            super().__init__(*args, **kwargs)
            self.sample_kws.update(dict(tune=1000, draws=500, target_accept=0.85))
        Issue encountered 2025-01-26 in ParallelSampler, details here
        https://discourse.pymc.io/t/dataset-size-dependent-eoferror/11977/13
        solution is to undo the mp_ctx set by pymc to use spawn instead
        though it might be a good idea to leave as None (fork) by default
        """
        self.model = None
        self._idata = None
        self.replace_idata = True
        self.sample_prior_pred_kws = dict(samples=500, return_inferencedata=True)
        self.sample_post_pred_kws = dict(store_ppc=True, insamp=False)
        self.sample_kws = dict(
            init="auto",  # aka jitter+adapt_diag
            tune=2000,  # twice the 1000 of pymc default
            draws=500,
            chains=4,
            cores=4,
            progressbar=True,
            mp_ctx=None,  # default 'fork' (thanks to pymc) alternative 'spawn'
            nuts_sampler="pymc",  # alternative “nutpie”
            nuts_sampler_kwargs=dict(
                target_accept=0.8,  # default, this is a reminder
                max_treedepth=10,  # default, this is a reminder
                early_max_treedepth=8,  # default, this is a reminder
                step_scale=0.25,  # default, this is a reminder
            ),
            idata_kwargs=dict(
                log_likelihood=True,  # usually useful
                log_prior=True,  # possibly useful?
            ),
        )
        self.rvs_for_posterior_plots = []
        self.calc_loglike_of_potential = False
        self.rvs_potential_loglike = None
        self.name = getattr(self, "name", "unnamed_model")
        self.version = getattr(self, "version", "unversioned_model")

    @property
    def mdl_id(self) -> str:
        """Get model id (name, version, obs name)
        NOTE: By convention we'll have a single name to cover all observation
        datasets included in the model (i.e several dfx)
        """
        obs_nm = getattr(self, "obs_nm", "unnamed_obs")
        return f"{self.name}_v{self.version}_{obs_nm}"

    @property
    def mdl_id_fn(self) -> str:
        """Get model id (name, version, obs name) safe for filename"""
        snl = SnakeyLowercaser()
        return snl.clean(re.sub(r"\.", "", self.mdl_id))

    @property
    def posterior(self) -> xr.Dataset:
        """Returns posterior from idata from previous run of sample"""
        try:
            s = self.idata.posterior
        except AttributeError as e:
            raise e("Run sample() first") from e
        return s

    @property
    def idata(self) -> az.InferenceData:
        """Returns Arviz InferenceData built from sampling to date"""
        assert self._idata, "Run update_idata() first"
        return self._idata

    def get_rvs(self) -> dict[list]:
        """Returns a dict of lists of RVs"""
        return dict(
            basic=self.model.basic_RVs,
            unobserved=self.model.unobserved_RVs,
            observed=self.model.observed_RVs,
            free=self.model.free_RVs,
            potentials=self.model.potentials,
            deterministics=self.model.deterministics,
        )

    def build(self, **kwargs):
        """Build the model"""
        helper_txt = "B" if self.model is None else "Reb"
        try:
            self._build(**kwargs)
            _log.info(f"{helper_txt}uilt model {self.mdl_id}")
        except AttributeError as e:
            raise NotImplementedError(
                "Create a method _build() in your"
                + " subclass, containing the model definition"
            ) from e

    def extend_build(self, **kwargs):
        """Extend build, initially developed to help PPC of GRW and missing value imputation"""
        try:
            self._extend_build(**kwargs)
            _log.info(f"Extended build of model {self.mdl_id}")
        except AttributeError as e:
            raise NotImplementedError(
                "Create a method _extend_build() in your"
                + " subclass, containing the model extension definition"
            ) from e

    def sample_prior_predictive(self, **kwargs):
        """Sample prior predictive:
            use base class defaults self.sample_prior_pred_kws
            or passed kwargs for pm.sample_prior_predictive()
        NOTE:
        + It's not currently possible to run Prior Predictive Checks on a
          model with missing values, per my detailed
          [MRE Notebook gist](https://gist.github.com/jonsedar/070319334bcf033773cc3e9495c79ea0)
          that illustrates the issue.
        + See https://github.com/pymc-devs/pymc/issues/4598
        """
        kws = dict(
            random_seed=kwargs.pop("random_seed", self.rsd),
            samples=kwargs.pop("samples", self.sample_prior_pred_kws["samples"]),
            return_inferencedata=kwargs.pop(
                "return_inferencedata",
                self.sample_prior_pred_kws["return_inferencedata"],
            ),
        )
        replace = kwargs.pop("replace", self.replace_idata)

        with self.model:
            try:
                prior_pred = pm.sample_prior_predictive(**{**kws, **kwargs})
            except UserWarning as e:
                _log.warning("Warning in mdl.sample_prior_predictive()", exc_info=e)
            finally:
                _ = self.update_idata(prior_pred, replace=replace)
            _log.info(f"Sampled prior predictive for {self.mdl_id}")
        return None

    def sample(self, **kwargs):
        """Sample posterior: use base class defaults self.sample_kws
        or passed kwargs for pm.sample()
        """
        kws = dict(
            init=kwargs.pop("init", self.sample_kws["init"]),
            random_seed=kwargs.pop("random_seed", self.rsd),
            tune=kwargs.pop("tune", self.sample_kws["tune"]),
            draws=kwargs.pop("draws", self.sample_kws["draws"]),
            chains=kwargs.pop("chains", self.sample_kws["chains"]),
            cores=kwargs.pop("cores", self.sample_kws["cores"]),
            progressbar=kwargs.pop("progressbar", self.sample_kws["progressbar"]),
            nuts_sampler=kwargs.pop("nuts_sampler", self.sample_kws["nuts_sampler"]),
            mp_ctx=kwargs.pop("mp_ctx", self.sample_kws["mp_ctx"]),
        )
        # these kwargs are dicts, so need to carefully update
        idata_kwargs = self.sample_kws["idata_kwargs"]
        idata_kwargs.update(kwargs.pop("idata_kwargs", {}))
        kws.update({"idata_kwargs": idata_kwargs})
        nuts_sampler_kwargs = self.sample_kws["nuts_sampler_kwargs"]
        nuts_sampler_kwargs.update(kwargs.pop("nuts_sampler_kwargs", {}))
        kws.update({"nuts_sampler_kwargs": nuts_sampler_kwargs})

        with self.model:
            try:
                posterior = pm.sample(**{**kws, **kwargs})
            except UserWarning as e:
                _log.warning("Warning in mdl.sample()", exc_info=e)
                pass
            except NotImplementedError as e:
                _log.error("NotImplementedError in mdl.sample()", exc_info=e)
                raise e
            except Exception as e:
                _log.error("Uncaught exception in mdl.sample()", exc_info=e)
                raise e
            else:
                _ = self.update_idata(posterior)

                _log.info(f"Sampled posterior for {self.mdl_id}")

                # optional manually calculate log_likelihood for potentials
                if self.calc_loglike_of_potential:
                    self.idata.add_groups(
                        dict(
                            log_likelihood=compute_log_likelihood_for_potential(
                                idata=self.idata,
                                model=self.model,
                                var_names=self.rvs_potential_loglike,
                                extend_inferencedata=False,
                            )
                        )
                    )
                    # rename to have exact same name as observedRVs
                    for nm in self.rvs_potential_loglike:
                        rx_pot = re.compile(r"^pot_")
                        nm0 = rx_pot.sub(r"", nm)
                        self.idata["log_likelihood"][nm0] = self.idata[
                            "log_likelihood"
                        ][nm]
                        del self.idata["log_likelihood"][nm]

        return None

    def sample_posterior_predictive(self, **kwargs) -> az.InferenceData | None:
        """Sample posterior predictive for kwarg var_names=[]
        Also uses self.sample_post_pred_kws or passed kwargs
        Note by default aimed toward out-of-sample PPC in production
        """
        store_ppc = kwargs.pop("store_ppc", self.sample_post_pred_kws["store_ppc"])
        posterior = kwargs.pop("posterior", self.posterior)
        kws = dict(
            trace=posterior,
            random_seed=kwargs.pop("random_seed", self.rsd),
            predictions=not kwargs.pop("insamp", self.sample_post_pred_kws["insamp"]),
        )
        with self.model:
            ppc = pm.sample_posterior_predictive(**{**kws, **kwargs})

        _log.info(f"Sampled posterior predictive for {self.mdl_id}")

        if store_ppc:
            _ = self.update_idata(ppc)
        else:
            return ppc

    def replace_obs(self, obsd: dict = None, obs_nm: str = None) -> None:
        """Replace the observation dataset(s)
        Data must live in (mutable) pm.Data containers in your _build() function
        must be: obsd = {internal_name_of_obs_variable: obs_dataframe}
        You must call `build()` afterward
        Optionally afterwards call `extend_build()` for future time-dependent PPC
        Optionally set `obs_nm` (useful for downstream plotting etc)
        """
        if obs_nm is not None:
            self.obs_nm = obs_nm
        for k, v in obsd.items():
            setattr(self, k, v)
            _log.info(f"Replaced obs {k} in {self.mdl_id}")

    def update_idata(self, idata: az.InferenceData, replace: bool = False) -> None:
        """Create (and update) an Arviz InferenceData object on-model from a
        passed-in presampled InferenceData object
        """
        # TODO improve this logic to use self.idata
        if self._idata is None:
            self._idata = idata
        else:
            side = "right" if replace else "left"
            self._idata.extend(idata, join=side)

    def debug(self) -> str:
        """Convenience to validate the parameterization: run debug on logp and
        random, and assert no MeasurableVariable nodes in the graph
        TODO capture the pymc debug outputs into the log"""
        msg = []
        if self.model is not None:
            assert_no_rvs(self.model.logp())
            msg.append("test: assert_no_rvs(logp)")
            _ = self.model.debug(fn="random", verbose=False)
            msg.append("debug: random")
            try:
                _ = self.model.debug(fn="logp", verbose=False)
                msg.append("debug: logp")
            except (TypeError, ValueError):
                _log.error("Model contains Potentials, debug logp not compatible")
                # _log.exception(
                #     "Model contains Potentials, debug logp not compatible",
                #     exc_info=True,
                # )
        return f"Ran {len(msg)} checks: [" + ", ".join(msg) + "]"
