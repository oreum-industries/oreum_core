# model.distributions.py
# copyright 2022 Oreum Industries
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.continuous import PositiveContinuous, assert_negative_support
from pymc3.distributions.dist_math import alltrue_elemwise, bound, logpow
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.theanof import floatX
from scipy import special, stats

RSD = 42
rng = np.random.default_rng(seed=RSD)

# NOTE hack to clip values away from {0, 1} for invcdfs
# Whilst value = {0, 1} is theoretically allowed, is seems to cause a
# numeric compuational issue somewhere in tt.erfcinv which throws infs.
# This screws up the downstream, so clip slightly away from {0, 1}
CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS = 1e-15  # 1e-18 too small

__all__ = [
    'boundzero_numpy',
    'boundzero_theano',
    'boundlog_numpy',
    'logpow_numpy',
    'Gamma',
    'GammaNumpy',
    'Gumbel',
    'InverseWeibull',
    'InverseWeibullNumpy',
    'ZeroInflatedInverseWeibull',
    'ZeroInflatedInverseWeibullNumpy',
    'Kumaraswamy',
    'Lognormal',
    'LognormalNumpy',
    'ZeroInflatedLognormal',
    'ZeroInflatedLognormalNumpy',
    'Normal',
    'NormalNumpy',
]


def boundzero_numpy(vals, *conditions):
    """Bound natural unit distribution params, return 0 for out-of-bounds
    Copy from pymc.bound pymc3.distributions.dist_math.py
    """
    return np.where(alltrue_elemwise(conditions), vals, 0.0)


def boundzero_theano(vals, *conditions):
    """Bound natural unit distribution params, return 0 for out-of-bounds
    Copy from pymc.bound pymc3.distributions.dist_math.py
    """
    return tt.switch(alltrue_elemwise(conditions), vals, 0.0)


def boundlog_numpy(vals, *conditions):
    """Bound log unit distribution params, return -inf for out-of-bounds
    Copy from pymc.bound pymc3.distributions.dist_math.py
    """
    return np.where(alltrue_elemwise(conditions), vals, -np.inf)


def logpow_numpy(x, m):
    """Copy from pymc3
    Safe calc log(x**m) since m*log(x) will fail when m, x = 0.
    """
    return np.where(x == 0, np.where(m == 0, 0.0, -np.inf), m * np.log(x))


class Gamma(pm.Gamma):
    """Inherit the pymc class, add cdf and invcdf"""

    def __init__(self):

        raise NotImplementedError(
            """Consider that InvCDF is hard to calculate: even scipy uses C 
            functions Recommend use different dist in practice"""
        )


class GammaNumpy:
    """Gamma PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations used in pymc3 custom distributions
    Helpful to compare these to scipy to confirm my correct implementation
    Ref: https://en.wikipedia.org/wiki/Gamma_distribution
    Params: x > 0, u in [0, 1], a (shape) > 0, b (rate) > 0
    """

    def __init__(self):
        self.name = 'Gamma'
        self.notation = {'notation': r'x \sim Gamma(\alpha, \beta)'}
        self.dist_natural = {
            'pdf': r'f(x \mid \alpha, \beta) = \frac{1}{\Gamma(\alpha)} \beta^{\alpha} x^{\alpha-1} e^{- \beta x}',
            'cdf': r'F(x \mid \alpha, \beta) = \frac{1}{\Gamma(\alpha)} \gamma(\alpha, \beta x)',
            'invcdf': r'F^{-1}(u \mid \alpha, \beta) = ',
        }
        self.dist_log = {
            'logpdf': r'\log f(x \mid \alpha, \beta) = -\log \Gamma(\alpha) + \log \beta^{\alpha} + \log x^{\alpha-1} - \beta x',
            'logcdf': r'\log F(x \mid \alpha, \beta) = -\log \Gamma(\alpha) + \log \gamma(\alpha, \beta x)',
            'loginvcdf': r'\log F^{-1}(u \mid \alpha, \beta) = ',
        }
        self.conditions = {
            'parameters': r'\alpha > 0 \, \text{(shape)}, \; \beta > 0 \, \text{(rate)}',
            'support': r'x \in (0, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {
            'mean': r'\frac{\alpha}{\beta}',
            'mode': r'\frac{\alpha - 1}{\beta}, \; \text{for} \alpha \geq 1',
            'variance': r'\frac{\alpha}{\beta^{2}}',
        }

    def pdf(self, x, a, b):
        """Gamma PDF
        compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2595
        """
        fn = (
            (1 / special.gamma(a))
            * np.power(b, a)
            * np.power(x, a - 1)
            * np.exp(-b * x)
        )
        return boundzero_numpy(fn, a > 0, b > 0, x >= 0)

    def cdf(self, x, a, b):
        """Gamma CDF:
        where $\gamma(a, bx)$ is lower incomplete gamma function [0, lim)
        compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2602
        """
        # fn = (1 / special.gamma(a)) * special.gammainc(a, b * x)
        fn = special.gammainc(a, b * x)
        return boundzero_numpy(fn, a > 0, b > 0, x >= 0)

    def invcdf(self, u, a, b):
        """Gamma Inverse CDF aka PPF:
        compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2608
        see sc.gammainc()
        """
        raise NotImplementedError('TODO gamma inverse CDF')

    def logpdf(self, x, a, b):
        """Gamma log PDF
        compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2599
        """
        fn = -special.gammaln(a) + logpow_numpy(b, a) + logpow_numpy(x, a - 1) - b * x
        return boundlog_numpy(fn, a > 0, b > 0, x > 0)

    def logcdf(self, x, a, b):
        """Gamma log CDF:
        where $\gamma(a, bx)$ is lower incomplete gamma function [0, lim)
        compare to https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L2614
        """
        return boundlog_numpy(
            (-special.gammaln(a)) + special.gammainc(a, b * x), a > 0, b > 0, x > 0
        )

    def loginvcdf(self, u, a, b):
        """Gamma log Inverse CDF aka log PPF:
        see sc.gammaincinv()
        """
        raise NotImplementedError('TODO gamma log inverse CDF')


class Gumbel(pm.Gumbel):
    """Inherit the pymc class, add cdf, logcdf and invcdf, loginvcdf
    Also clobber logp (!)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, defaults=("mode",), **kwargs)

    def logp(self, value):
        """
        JS patch refactored code to align with other distributions

        Calculate log-probability of Gumbel distribution at specified value.

        z = (x - mu) / b
        pdf = (1 / b) * exp(-z - exp(-z))
        logpdf = -log(b) - z - exp(-z)

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the
            log probabilities for multiple values are desired the values must
            be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """

        mu = self.mu
        beta = self.beta
        z = (value - mu) / beta

        logp = -tt.log(beta) - z - tt.exp(-z)

        return bound(logp, beta > 0)

    def logcdf(self, value):
        """
        JS patch refactored code to align with other distributions

        cdf = exp(-exp(-(X - mu) / b))
        logcdf = -exp(-(X-mu)/b)

        Compute the log of the cumulative distribution function for
        Gumbel distribution at the specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a
            numpy array or theano tensor.

        Returns
        -------
        TensorVariable
        """
        beta = self.beta
        mu = self.mu

        logcdf = -tt.exp(-(value - mu) / beta)

        return bound(logcdf, beta > 0)

    def loginvcdf(self, value):
        """
        JS new function

        invcdf = mu - b * log(-log(u))
        loginvcdf = log(mu) + log(1 - (b * log(-log(u))/mu))

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the
            log probabilities for multiple values are desired the values must
            be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """

        beta = self.beta
        mu = self.mu

        loginvcdf = tt.log(mu) + tt.log(1 - (beta * tt.log(-tt.log(value)) / mu))

        return bound(loginvcdf, beta > 0)


class InverseWeibull(PositiveContinuous):
    r"""
    Inverse Weibull log-likelihood, the reciprocal of the Weibull distribution,
    also known as the Fréchet distribution, a special case of the generalized
    extreme value distribution.

    See scipy for reference
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html
    https://github.com/scipy/scipy/blob/v1.6.0/scipy/stats/_continuous_distns.py

    The pdf of this distribution is
    .. math::
       f(x \mid \alpha, s, m) =
           \frac{\alpha }{s}} \; \left({\frac{x-m}{s}}\right)^{{-1-\alpha }}\;e^{{-({\frac{x-m}{s}})^{{-\alpha }}}
    .. plot::
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 500)
        alphas = [1., 2., 3., 3.]
        betas = [1., 1., 1., .5]
        for a, b in zip(alphas, betas):
            pdf = st.invgamma.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()
    ========  ======================================================
    Support   :math:`x \in (-\infty, \infty)`
    Mean      :math:`{\begin{cases}\ m+s\Gamma \left(1-{\frac  {1}{\alpha }}\right)&{\text{for }}\alpha >1\\\ \infty &{\text{otherwise}}\end{cases}}`
    Variance  :math:`{\begin{cases}\ s^{2}\left(\Gamma \left(1-{\frac  {2}{\alpha }}\right)-\left(\Gamma \left(1-{\frac{1}{\alpha }}\right)\right)^{2}\right)&{\text{for }}\alpha >2\\\ \infty &{\text{otherwise}}\end{cases}}` # noqa: W505

    ========  ======================================================
    Parameters
    ----------
    alpha: float
        Shape parameter (alpha > 0).
    s: float
        Scale parameter (s > 0), default = 1
    ## m: float
    ##     Location parameter (mu in (-inf, inf)), default = 0
    """

    def __init__(self, alpha=None, s=1.0, *args, **kwargs):
        super().__init__(*args, defaults=("mode",), **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.s = s = tt.as_tensor_variable(floatX(s))
        self.mode = s * tt.power(alpha / (1.0 + alpha), 1.0 / alpha)

        assert_negative_support(alpha, "alpha", "InverseWeibull")
        assert_negative_support(s, "s", "InverseWeibull")

    def _distr_parameters_for_repr(self):
        return ["alpha", 's']

    def random(self, point=None, size=None):
        """
        Draw random values from InverseWeibull PDF distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        alpha, s = draw_values([self.alpha, self.s], point=point, size=size)
        return generate_samples(
            stats.invweibull.rvs,
            c=alpha,
            scale=s,
            loc=0.0,
            dist_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        """
        Calculate log-probability of InverseWeibull distribution at specified value.
        pdf: https://www.wolframalpha.com/input/?i=%28a%2Fs%29+*+%28x%2Fs%29**%28-1-a%29+*+exp%28-%28x%2Fs%29**-a%29
        alt form according to WA: a e^(-(s/x)^a) s^a x^(-1 - a)

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor
        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        s = self.s
        return bound(
            (
                tt.log(alpha)
                - tt.log(s)
                + logpow(s / value, 1.0 + alpha)
                - tt.power(
                    s / value, alpha
                )  # this term grossly dominates if alpha >> 2
            ),
            value > 0.0,
            alpha > 0.0,
            s > 0.0,
        )

    def cdf(self, value):
        """InverseWeibull CDF"""
        alpha = self.alpha
        s = self.s
        fn = tt.exp(-tt.power(value / s, -alpha))
        return boundzero_theano(fn, alpha > 0, s > 0, value > 0)

    def logcdf(self, value):
        """InverseWeibull log CDF
        ref: ? manually calced and confirmed vs scipy
        """
        alpha = self.alpha
        s = self.s
        fn = -tt.power(value / s, -alpha)
        return bound(fn, alpha > 0, s > 0, value > 0)

    def invcdf(self, value):
        """InverseWeibull Inverse CDF aka PPF"""
        alpha = self.alpha
        s = self.s
        value = tt.clip(
            value,
            CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS,
            1 - CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS,
        )
        fn = s * tt.power(-tt.log(value), -1.0 / alpha)
        return boundzero_theano(fn, alpha > 0, s > 0, value >= 0, value <= 1)

    def loginvcdf(self, value):
        """InverseWeibull log Inverse CDF aka log PPF
        ref: ? manually calced and confirmed vs scipy
        """
        alpha = self.alpha
        s = self.s
        fn = tt.log(s) - (1.0 / alpha) * tt.log(-tt.log(value))
        return bound(fn, alpha > 0, s > 0, value >= 0, value <= 1)


class InverseWeibullNumpy:
    """Inverse Weibull PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations potentially used if needed in pymc3 custom distributions
    Helpful to compare these to scipy to confirm my correct implementation
    NOTE: I'm lazy and have set m=0 throughout: this suits my usecase anyhow
    Ref: https://en.wikipedia.org/wiki/Fréchet_distribution
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html?highlight=inverse%20weibull
    Params: alpha (shape) > 0, s (scale) > 0, m (location of minimum) = 0
    Support: x > 0, u in [0, 1]
    """

    def __init__(self):
        self.name = 'InverseWeibull'
        self.notation = {'notation': r'x \sim InverseWeibull(\alpha, s, m=0)'}
        self.dist_natural = {
            'pdf': r"""f(x \mid \alpha, s, m=0) = \frac{\alpha}{s} \;
                                                  \left( \frac{x}{s} \right)^{-1-\alpha} \;
                                                  \exp \left( -\left( \frac{x}{s} \right)^{-\alpha} \right)""",
            'cdf': r'F(x \mid \alpha, s, m=0) = \exp \left( -\left( \frac{x}{s} \right)^{-\alpha} \right)',
            'invcdf': r"""F^{-1}(u \mid \alpha, s, m=0) = s \log(u)^{-\frac{1}{\alpha}}""",
        }
        self.dist_log = {
            'logpdf': r"""\log f(x \mid \alpha, s, m=0) = \log{\alpha} - (1+\alpha)\log{x} + 
                        \alpha \log{s} - \left( \frac{x}{s} \right)^{-\alpha}""",
            'logcdf': r'\log F(x \mid \alpha, s, m=0) = - \left( \frac{x}{s} \right)^{-\alpha}',
            'loginvcdf': r'\log F^{-1}(u \mid \alpha, s, m=0) = \log(s) - \frac{1}{\alpha} * \log(-\log(u))',
        }
        self.conditions = {
            'parameters': r"""\alpha > 0 \, \text{(shape)}, \; 
                            s > 0 \, \text{(scale, default } s=1 \text{)}, \; 
                            m \in (-\infty, \infty) \, \text{(location of minimum, default } m=0 \text{)}""",
            'support': r'x \in (m, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {
            'mean': r"""
                \begin{cases}
                m + s \Gamma \left( 1 - \frac{1}{\alpha} \right) & \text{for } \alpha > 1 \\
                \infty & \text{otherwise} \\
                \end{cases}""",
            'mode': r'm + s \left( \frac{\alpha}{1+\alpha} \right)^{1/\alpha}',
            'variance': r"""
                \begin{cases}
                s^{2} \left( \Gamma \left( 1-\frac{2}{\alpha} \right) - 
                            \left( \Gamma \left( 1-\frac{1}{\alpha} \right) \right)^{2} 
                    \right) & \text{for } \alpha > 2 \\
                \infty & \text{otherwise}
                \end{cases}""",
        }

    def pdf(self, x, a, s):
        """InverseWeibull PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3919
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = (a / s) * np.power(x / s, -1.0 - a) * np.exp(-np.power(x / s, -a))
        return boundzero_numpy(fn, a > 0, s > 0, x > 0)

    def cdf(self, x, a, s):
        """InverseWeibull CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3926
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = np.exp(-np.power(x / s, -a))
        return boundzero_numpy(fn, a > 0, s > 0, x > 0)

    def invcdf(self, u, a, s):
        """InverseWeibull Inverse CDF aka PPF:
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3930
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = s * np.power(-np.log(u), -1.0 / a)
        return boundzero_numpy(fn, a > 0, s > 0, u >= 0, u <= 1)

    def logpdf(self, x, a, s):
        """InverseWeibull log PDF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = (
            np.log(a)
            - np.log(s)
            + logpow_numpy(x / s, -1.0 - a)
            - np.power(x / s, -a)  # this term grossly dominates if a >> 2
        )
        return boundlog_numpy(fn, a > 0, s > 0, x >= 0)

    def logcdf(self, x, a, s):
        """InverseWeibull log CDF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = -np.power(x / s, -a)
        return boundlog_numpy(fn, a > 0, s > 0, x >= 0)

    def loginvcdf(self, u, a, s):
        """InverseWeibull log Inverse CDF aka log PPF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = np.log(s) - (1.0 / a) * np.log(-np.log(u))
        return boundlog_numpy(fn, a > 0, s > 0, u >= 0, u <= 1)


class ZeroInflatedInverseWeibull(PositiveContinuous):
    r"""
    ZeroInflatedInvserseWeibull log-likelihood

    WIP! Mixture model to allow for observations dominated by zeros such as sev

    also see 
    + McElreath 2014, http://xcelab.net/rmpubs/Mcelreath%20Koster%202014.pdf, 
                      https://github.com/rmcelreath/mcelreath-koster-human-nature-2014
    + Jones 2013, https://royalsocietypublishing.org/doi/10.1098/rspb.2013.1210
    + https://stackoverflow.com/questions/42409761/pymc3-nuts-has-difficulty-sampling-from-a-hierarchical-zero-inflated-gamma-mode

    The pmf of this distribution is
    .. math::
 
        f(x \mid \psi, \alpha, s) = \left\{
            \begin{array}{l}
                (1 - \psi), & \text{if } x = 0 \\
                \psi \, \text{InverseWeibull}(\alpha, s), & \text{if } x > 0
            \end{array} 
            \right.

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi \, \text{InverseWeibull}(\mu, \sigma)`
    Variance  :math: TODO
    ========  ==========================
    
    Parameters
    ----------
    psi: float
        Expected proportion of InverseWeibull variates (0 <= psi <= 1)
    alpha: float
    s: float
    """

    def __init__(self, psi, alpha, s, *args, **kwargs):
        super().__init__(*args, defaults=("mode",), **kwargs)

        self.psi = psi = tt.as_tensor_variable(floatX(psi))
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.s = s = tt.as_tensor_variable(floatX(s))
        self.invweibull = InverseWeibull.dist(alpha=alpha, s=s)

        # TODO
        # self.mean = self.psi * self.invweibull.mean
        self.mode = self.psi * self.invweibull.mode

        assert_negative_support(alpha, "alpha", "ZeroInflatedInverseWeibull")
        assert_negative_support(s, "s", "ZeroInflatedInverseWeibull")

    # def _random(self, psi, size=None):
    #     """Note by definition any rvs_ from invweibull that are zero will
    #         correctly remain zero, covering the case x = 0"""
    #     rvs_ = self.invweibull.random(size=size)
    #     return rvs_ * psi

    def _random(self, psi, size=None):
        """Inputs are numpy arrays"""
        rvs_ = self.invweibull.random(size=size)
        pi = stats.binom(n=np.repeat([1], len(psi)), p=psi).rvs(len(psi))
        return rvs_ * pi

    def random(self, point=None, size=None):
        """
        Draw random values from ZeroInflatedInverseWeibull PDF distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        psi, alpha, s = draw_values(
            [self.psi, self.alpha, self.s], point=point, size=size
        )
        return generate_samples(self._random, psi, dist_shape=self.shape, size=size)

    def logp(self, value):
        """LogPDF"""
        psi = self.psi
        logp_ = tt.switch(
            tt.neq(value, 0),  # or use tt.gt(value, 0), dunno which faster
            tt.log(psi) + self.invweibull.logp(value),
            tt.log1p(-psi),
        )
        return bound(logp_, value >= 0, psi > 0, psi < 1)

    def cdf(self, value):
        """CDF"""
        psi = self.psi
        cdf_ = (1.0 - psi) * 1 + psi * self.invweibull.cdf(value)
        return boundzero_theano(cdf_, value >= 0, psi > 0, psi < 1)

    def invcdf(self, value):
        """InvCDF aka PPF"""
        psi = self.psi
        invcdf_ = self.invweibull.invcdf((value + psi - 1) / psi)
        return boundzero_theano(invcdf_, value >= 0, value <= 1, psi > 0, psi < 1)


class ZeroInflatedInverseWeibullNumpy:
    """Zero-inflated Inverse Weibull PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations potentially used if needed in pymc3 custom distributions
    Helpful to compare these ? seems rare
    NOTE: I'm lazy and have set m=0 throughout: this suits my usecase anyhow
    Ref: https://en.wikipedia.org/wiki/Fréchet_distribution
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html?highlight=inverse%20weibull
    Params: 0 < psi < 1 (prop invweibull), alpha (shape) > 0, s (scale) > 0, m (location of minimum) = 0
    Support: x > 0, u in [0, 1]
    """

    def __init__(self):
        self.name = 'InverseWeibull'
        self.notation = {'notation': r'x \sim InverseWeibull(\alpha, s, m=0)'}
        self.dist_natural = {
            'pdf': r"""f(x \mid \alpha, s, m=0) = \frac{\alpha}{s} \;
                                                  \left( \frac{x}{s} \right)^{-1-\alpha} \;
                                                  \exp \left( -\left( \frac{x}{s} \right)^{-\alpha} \right)""",
            'cdf': r'F(x \mid \alpha, s, m=0) = \exp \left( -\left( \frac{x}{s} \right)^{-\alpha} \right)',
            'invcdf': r"""F^{-1}(u \mid \alpha, s, m=0) = s \log(u)^{-\frac{1}{\alpha}}""",
        }
        self.dist_log = {
            'logpdf': r"""\log f(x \mid \alpha, s, m=0) = \log{\alpha} - (1+\alpha)\log{x} + 
                        \alpha \log{s} - \left( \frac{x}{s} \right)^{-\alpha}""",
            'logcdf': r'\log F(x \mid \alpha, s, m=0) = - \left( \frac{x}{s} \right)^{-\alpha}',
            'loginvcdf': r'\log F^{-1}(u \mid \alpha, s, m=0) = \log(s) - \frac{1}{\alpha} * \log(-\log(u))',
        }
        self.conditions = {
            'parameters': r"""\alpha > 0 \, \text{(shape)}, \; 
                            s > 0 \, \text{(scale, default } s=1 \text{)}, \; 
                            m \in (-\infty, \infty) \, \text{(location of minimum, default } m=0 \text{)}""",
            'support': r'x \in (m, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {
            'mean': r"""
                \begin{cases}
                m + s \Gamma \left( 1 - \frac{1}{\alpha} \right) & \text{for } \alpha > 1 \\
                \infty & \text{otherwise} \\
                \end{cases}""",
            'mode': r'm + s \left( \frac{\alpha}{1+\alpha} \right)^{1/\alpha}',
            'variance': r"""
                \begin{cases}
                s^{2} \left( \Gamma \left( 1-\frac{2}{\alpha} \right) - 
                            \left( \Gamma \left( 1-\frac{1}{\alpha} \right) \right)^{2} 
                    \right) & \text{for } \alpha > 2 \\
                \infty & \text{otherwise}
                \end{cases}""",
        }

    def pdf(self, x, a, s):
        """InverseWeibull PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3919
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = (a / s) * np.power(x / s, -1.0 - a) * np.exp(-np.power(x / s, -a))
        return boundzero_numpy(fn, a > 0, s > 0, x > 0)

    def cdf(self, x, a, s):
        """InverseWeibull CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3926
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = np.exp(-np.power(x / s, -a))
        return boundzero_numpy(fn, a > 0, s > 0, x > 0)

    def invcdf(self, u, a, s):
        """InverseWeibull Inverse CDF aka PPF:
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L3930
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = s * np.power(-np.log(u), -1.0 / a)
        return boundzero_numpy(fn, a > 0, s > 0, u >= 0, u <= 1)

    def logpdf(self, x, a, s):
        """InverseWeibull log PDF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = (
            np.log(a)
            - np.log(s)
            + logpow_numpy(x / s, -1.0 - a)
            - np.power(x / s, -a)  # this term grossly dominates if a >> 2
        )
        return boundlog_numpy(fn, a > 0, s > 0, x >= 0)

    def logcdf(self, x, a, s):
        """InverseWeibull log CDF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = -np.power(x / s, -a)
        return boundlog_numpy(fn, a > 0, s > 0, x >= 0)

    def loginvcdf(self, u, a, s):
        """InverseWeibull log Inverse CDF aka log PPF
        ref: ? manually calced and confirmed vs scipy
        """
        a = np.array(a).astype(np.float)  # , casting='no')
        s = np.array(s).astype(np.float)  # , casting='no')
        fn = np.log(s) - (1.0 / a) * np.log(-np.log(u))
        return boundlog_numpy(fn, a > 0, s > 0, u >= 0, u <= 1)


class Kumaraswamy(pm.Kumaraswamy):
    """Inherit the pymc class, add cdf, logcdf and invcdf, loginvcdf"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logcdf(self, value):
        """
        JS new function

        cdf = 1 - (1 - X**a)**b
        logcdf = log(1) + log(1 - ((1 - X**a)**b / 1)) = log(1 - (1 - X**a)**b)

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the
            log probabilities for multiple values are desired the values must
            be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        a = self.a
        b = self.b

        logcdf = tt.log(1 - (1 - value**a) ** b)

        return bound(logcdf, value >= 0, value <= 1, a > 0, b > 0)

    def loginvcdf(self, value):
        """
        JS new function

        invcdf = (1 - (1-u) ** (1/b)) ** (1/a)
        loginvcdf = (1/a) * np.log(1 - (1-u)**(1/b))

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the
            log probabilities for multiple values are desired the values must
            be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """

        a = self.a
        b = self.b

        loginvcdf = (1 / a) * tt.log(1 - (1 - value) ** (1 / b))

        return bound(loginvcdf, value >= 0, value <= 1, a > 0, b > 0)


class Lognormal(pm.Lognormal):
    """Inherit the pymc class, add cdf and invcdf"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cdf(self, value):
        """Lognormal CDF"""
        mu = self.mu
        sigma = self.sigma
        z = (tt.log(value) - mu) / sigma
        fn = 0.5 * tt.erfc(-z / tt.sqrt(2.0))
        # convenience alt use pymc3's invprobit: # fn = pm.math.invprobit(z)
        return boundzero_theano(fn, sigma > 0, value > 0)

    def invcdf(self, value):
        """Lognormal Inverse CDF aka PPF"""
        mu = self.mu
        sigma = self.sigma
        # value = tt.clip(value, CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS, 1-CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS)
        fn = tt.exp(mu - sigma * tt.sqrt(2) * tt.erfcinv(2 * value))
        return boundzero_theano(fn, sigma > 0, value >= 0, value <= 1)


class LognormalNumpy:
    """Lognormal PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations potentially used if needed in pymc3 custom distributions
    Helpful to compare these to scipy to confirm my correct implementation
    Ref: https://en.wikipedia.org/wiki/Log-normal_distribution
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html?highlight=lognorm#scipy.stats.lognorm
    Params: mu (location) > 0, sigma (variance) > 0
    Support: x > 0, u in [0, 1],
    """

    def __init__(self):
        self.name = 'Lognormal'
        self.notation = {'notation': r'x \sim Lognormal(\mu, \sigma)'}
        self.dist_natural = {
            'pdf': r"""f(x \mid \mu, \sigma) = \frac{1}{x \sigma \sqrt{2 \pi}} \exp \left( -{ \frac{(\log{x} - \mu)^{2}}{2 \sigma^{2}}} \right)
                                             = \frac{1}{x \sigma \sqrt{2 \pi}} \exp - \left(\frac{\log{x}-\mu}{\sigma \sqrt{2}} \right)^{2}""",
            'cdf': r"""F(x \mid \mu, \sigma) = \frac{1}{2} \left[ 1 + \text{erf} \left(\frac{\log{x}-\mu}{\sigma \sqrt{2}} \right) \right]
                                             = \frac{1}{2} \text{erfc} \left( \frac{-\log{x} -\mu}{\sigma \sqrt{2}} \right)""",
            'invcdf': r"""F^{-1}(u \mid \mu, \sigma) = \exp \left( \mu + \sigma * \text{normal_invcdf}(u) \right)
                                                     = \exp \left( \mu - \sigma \sqrt{2} \text{erfcinv}(2u) \right)""",
        }
        self.dist_log = {
            'logpdf': r'\log f(x \mid \mu, \sigma) = - \frac{1}{2 \sigma^2} \log{(x-\mu)^{2}} + \frac{1}{2} \log{\frac{1}{2 \pi \sigma^{2}}} -\log{x}',
            'logcdf': r'\log F(x \mid \mu, \sigma) = \log \left[\frac{1}{2} \text{erfc} \left( \frac{\log{(x)} -\mu}{\sigma \sqrt{2}} \right) \right]',
            'loginvcdf': r'\log F^{-1}(u \mid \mu, \sigma) = \mu - \sigma \sqrt{2} \text{erfcinv}(2u)',
        }
        self.conditions = {
            'parameters': r'\mu \in (-\infty, \infty) \, \text{(location)}, \; \sigma > 0 \, \text{(std. dev.)}',
            'support': r'x \in (0, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {
            'mean': r'\exp \left( \mu +\frac{\sigma^{2}}{2} \right)',
            'median': r'\exp ( \mu )',
            'mode': r'\exp ( \mu  - \sigma^{2} )',
            'variance': r'[\exp (\sigma^{2}) - 1] \exp (2 \mu + \sigma^{2})',
        }

    def pdf(self, x, mu, sigma):
        """Lognormal PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5050
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        fn = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
            -np.power((np.log(x) - mu) / (sigma * np.sqrt(2)), 2)
        )
        return boundzero_numpy(fn, sigma > 0, x > 0)

    def cdf(self, x, mu, sigma):
        """Lognormal CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5057
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        z = (np.log(x) - mu) / sigma
        fn = 0.5 * special.erfc(-z / np.sqrt(2.0))
        return boundzero_numpy(fn, sigma > 0, x > 0)

    def invcdf(self, u, mu, sigma):
        """Lognormal Inverse CDF aka PPF:
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5063
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        # u = np.maximum(np.minimum(u, 1-CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS), CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS)
        fn = np.exp(mu - sigma * np.sqrt(2) * special.erfcinv(2 * u))
        return boundzero_numpy(fn, sigma > 0, u >= 0, u <= 1)

    def logpdf(self, x, mu, sigma):
        """Lognormal log PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5054
        ref: https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L1887
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        fn = (
            -np.power(np.log(x) - mu, 2) / (2 * np.power(sigma, 2))
            + 0.5 * np.log(1 / (2 * np.pi * np.power(sigma, 2)))
            - np.log(x)
        )
        return boundlog_numpy(fn, sigma > 0, x > 0)

    def logcdf(self, x, mu, sigma):
        """Lognormal log CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5060
        ref: https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L1913
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        fn = np.log(self.cdf(x, mu, sigma))
        return boundlog_numpy(fn, sigma > 0, x > 0)

    def loginvcdf(self, u, mu, sigma):
        """Lognormal log Inverse CDF aka log PPF
        ref: ?
        """
        mu = np.array(mu).astype(np.float)  # , casting='no')
        sigma = np.array(sigma).astype(np.float)  # , casting='no')
        fn = mu - sigma * np.sqrt(2) * special.erfcinv(2 * u)
        return boundlog_numpy(fn, sigma > 0, u >= 0, u <= 1)


class ZeroInflatedLognormal(PositiveContinuous):
    r"""
    ZeroInflatedLognormal log-likelihood

    WIP! Mixture model to allow for observations dominated by zeros such as freq

    also see 
    + McElreath 2014, http://xcelab.net/rmpubs/Mcelreath%20Koster%202014.pdf, 
                      https://github.com/rmcelreath/mcelreath-koster-human-nature-2014
    + Jones 2013, https://royalsocietypublishing.org/doi/10.1098/rspb.2013.1210
    + https://stackoverflow.com/questions/42409761/pymc3-nuts-has-difficulty-sampling-from-a-hierarchical-zero-inflated-gamma-mode

    The pmf of this distribution is
    .. math::
 
        f(x \mid \psi, \mu, \sigma) = \left\{
            \begin{array}{l}
                (1 - \psi), & \text{if } x = 0 \\
                \psi \, \text{Lognormal}(\mu, \sigma), & \text{if } x > 0
            \end{array} 
            \right.

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi \text{Lognormal}(\mu, \sigma)`
    Variance  :math: TODO
    ========  ==========================
    
    Parameters
    ----------
    psi: float
        Expected proportion of Lognormal variates (0 <= psi <= 1)
    mu: float
    sigma: float
    """

    def __init__(self, psi, mu, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)  # defaults=("mode",)

        self.psi = psi = tt.as_tensor_variable(floatX(psi))
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.sigma = sigma = tt.as_tensor_variable(floatX(sigma))
        self.lognorm = Lognormal.dist(mu, sigma)
        # self.bernoulli = stats.binom()

        # TODO
        self.mean = self.psi * self.lognorm.mean  # lognorm.mean = exp(mu + sigma^2 / 2)
        # self.median = tt.exp(self.mu)
        # self.mode = 0 #self.psi * self.lognorm.mode

        assert_negative_support(sigma, "sigma", "ZeroInflatedLognormal")

    # def _random(self, psi, mu, sigma, size=None):
    #     """ Not sure 2021-02-21
    #         `Note by definition any rvs_ from lognorm that are zero will
    #         correctly remain zero, covering the case x = 0`
    #     """
    #     rvs_ = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
    #     return rvs_ * psi

    def _random(self, psi, mu, sigma, size=None):
        """Inputs are numpy arrays"""
        rvs_ = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
        pi = stats.binom(n=np.repeat([1], len(psi)), p=psi).rvs(len(psi))
        return rvs_ * pi

    def random(self, point=None, size=None):
        """
        Draw random values from InverseWeibull PDF distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        psi, mu, sigma = draw_values(
            [self.psi, self.mu, self.sigma], point=point, size=size
        )
        return generate_samples(
            self._random, psi, mu, sigma, dist_shape=self.shape, size=size
        )

    def logp(self, value):
        """LogPDF"""
        psi = self.psi
        logp_ = tt.switch(
            tt.gt(value, 0), tt.log(psi) + self.lognorm.logp(value), tt.log1p(-psi)
        )
        return bound(logp_, value >= 0, psi > 0, psi < 1)

    def cdf(self, value):
        """CDF"""
        psi = self.psi
        cdf_ = (1.0 - psi) * 1 + psi * self.lognorm.cdf(value)
        return boundzero_theano(cdf_, value >= 0, psi > 0, psi < 1)

    def invcdf(self, value):
        """InvCDF aka PPF"""
        psi = self.psi
        invcdf_ = self.lognorm.invcdf((value + psi - 1) / psi)
        return boundzero_theano(invcdf_, value >= 0, value <= 1, psi > 0, psi < 1)


class ZeroInflatedLognormalNumpy:
    """Zero-inflated Lognormal PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations potentially used if needed in pymc3 custom distributions
    Helpful to compare these to ? (seems to be quite rare)
    Ref: https://royalsocietypublishing.org/doi/10.1098/rspb.2013.1210
    Ref:
    Params: 0 < psi < 1 (prop lognormal), mu (location) > 0, sigma (variance) > 0
    Support: x > 0, u in [0, 1],
    """

    def __init__(self):
        self.name = 'ZeroInflatedLognormal'
        self.notation = {'notation': r'x \sim ZILognormal(\psi, \mu, \sigma)'}
        self.dist_natural = {
            'pdf': r"""f(x \mid \psi, \mu, \sigma) = \left\{ \begin{array}{l}
                    (1 - \psi), & \text{if } x = 0 \\
                    \psi \text{LognormalPDF}(\mu, \sigma, x), & \text{if } x > 0 \\
                    \end{array} \right.""",
            'cdf': r"""F(x \mid \psi, \mu, \sigma) = (1 - \psi) + \psi \text{LognormalCDF}(\mu, \sigma)""",
            'invcdf': r"""F^{-1}(u \mid \psi, \mu, \sigma) = \text{LognormalInvCDF} \left( \frac{u - 1}{\psi} + 1, \mu, \sigma \right)""",
        }
        self.dist_log = {
            'logpdf': r"""\log f(x \mid \psi, \mu, \sigma) = \left\{\begin{array}{l}
                            \log(1 - \psi), & \text{if } x = 0 \\
                            \log(\psi) + \text{LognormalLogPDF}(\mu, \sigma, x), & \text{if } x > 0 \\
                        \end{array} \right.""",
            'logcdf': r"""\log F(x \mid \psi, \mu, \sigma) = \log((1 - \psi) + \psi \text{LognormalLogCDF}(\mu, \sigma, x))""",
            'loginvcdf': r"""\log F^{-1}(u \mid \psi, \mu, \sigma) = \log(\text{LognormalLogInvCDF} \left( \frac{u + \psi - 1}{\psi}), \mu, \sigma) \right)""",
        }
        self.conditions = {
            'parameters': r"""\psi \in (0, 1)\, \text{(prop. lognormal)}, \;
                              \mu \in (-\infty, \infty) \, \text{(location)}, \; 
                              \sigma > 0 \, \text{(std. dev.)}""",
            'support': r'x \in [0, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {'mean': r'TODO', 'mode': r'TODO', 'variance': r'TODO'}
        self.lognorm = LognormalNumpy()

    def rvs(self, psi, mu, sigma):
        """ZILognormal random variates"""
        if len(psi) == len(mu):
            rvs_ = stats.lognorm(s=sigma, scale=np.exp(mu)).rvs()
            # pi = stats.binom(n=np.repeat([1], len(psi)), p=psi).rvs(len(psi))
            pi = stats.binom(n=1, p=psi).rvs()
        else:
            raise ValueError('psi and mu must have ssame length')

        return rvs_ * pi

    def pdf(self, x, psi, mu, sigma):
        """ZILognormal PDF"""
        psi = np.float(psi)
        mu = np.float(mu)
        sigma = np.float(sigma)
        pdf_ = np.where(x > 0, psi * self.lognorm.pdf(x, mu, sigma), 1.0 - psi)
        return boundzero_numpy(pdf_, psi > 0, psi < 1, sigma > 0, x >= 0)

    def cdf(self, x, psi, mu, sigma):
        """ZILognormal CDF"""
        psi = np.float(psi)
        mu = np.float(mu)
        sigma = np.float(sigma)
        cdf_ = (1.0 - psi) + psi * self.lognorm.cdf(x, mu, sigma)
        return boundzero_numpy(cdf_, psi > 0, psi < 1, sigma > 0, x >= 0)

    def invcdf(self, u, psi, mu, sigma):
        """ZILognormal Inverse CDF aka PPF:"""
        psi = np.float(psi)
        mu = np.float(mu)
        sigma = np.float(sigma)
        # z = (u + psi - 1.) / psi
        z = ((u - 1.0) / psi) + 1  # better formulation avoid computational issues
        invcdf_ = self.lognorm.invcdf(z, mu, sigma)
        # return invcdf_
        return boundzero_numpy(invcdf_, psi > 0, psi < 1, sigma > 0, u >= 0, u <= 1)


class Normal(pm.Normal):
    """Inherit the pymc class, add invcdf"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # defaults=("mode",)

    def invcdf(self, value):
        """Normal inverse cdf $F^{-1}(u | \mu,\sigma) -\sqrt{2} * \text{erfcinv}(2u)$"""
        mu = self.mu
        sigma = self.sigma
        value = tt.clip(
            value,
            CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS,
            1 - CLIP_U_AWAY_FROM_ZERO_ONE_FOR_INVCDFS,
        )
        fn = mu - sigma * tt.sqrt(2.0) * tt.erfcinv(2.0 * value)
        return boundzero_theano(fn, value >= 0.0, value <= 1.0)

    def loginvcdf(self, value):
        """Normal log Inverse CDF aka log PPF
        ref: ?
        """
        mu = self.mu
        sigma = self.sigma
        fn = np.log(mu - sigma * tt.sqrt(2.0) * tt.erfcinv(2.0 * value))
        # fn = np.log(mu - sigma * np.sqrt(2.) * special.erfcinv(2 * u))
        return bound(fn, value >= 0.0, value <= 1.0)


class NormalNumpy:
    """Normal PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
    Manual implementations potentially used if needed in pymc3 custom distributions
    Helpful to compare these to scipy to confirm my correct implementation
    Ref: https://en.wikipedia.org/wiki/Normal_distribution
    Ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L274
    Params: mu (location) > 0, sigma (variance) > 0
    Support: x > 0, u in [0, 1],
    """

    def __init__(self):
        self.name = 'Normal'
        self.notation = {'notation': r'x \sim Normal(\mu, \sigma)'}
        self.dist_natural = {
            'pdf': r'f(x \mid \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left( (x-\mu) / \sigma \right)^{2}}',
            'cdf': r"""F(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^{2}/2} dt 
                                             = \frac{1}{2} \big[1 + \text{erf} \big( \frac{x - \mu}{\sigma \sqrt{2}} \big) \big]
                                             = \frac{1}{2} \text{erfc} \big(- \frac{x - \mu}{\sigma \sqrt{2}} \big)""",
            'invcdf': r'F^{-1}(u \mid \mu, \sigma) = \mu - \sigma \sqrt{2} \text{erfcinv}(2u)',
        }
        self.dist_log = {
            'logpdf': r'\log f(x \mid \mu, \sigma) = - \log(\sigma \sqrt{2 \pi}) - \frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^{2}',
            'logcdf': r'\log F(x \mid \mu, \sigma) = \log{(\frac{1}{2})} + \log{\left[ \text{erfc} \left(- \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]}',
            'loginvcdf': r'\log F^{-1}(u \mid \mu, \sigma) = \log \left[ \mu - \sigma \sqrt{2} \text{erfcinv}(2u) \right]',
        }
        self.conditions = {
            'parameters': r'\mu \in (-\infty, \infty) \, \text{(location)}, \; \sigma > 0 \, \text{(std. dev.)}',
            'support': r'x \in (-\infty, \infty), \; u \sim \text{Uniform([0, 1])}',
        }
        self.summary_stats = {'mean': r'\mu', 'mode': r'\mu', 'variance': r'\sigma^{2}'}

    def pdf(self, x, mu, sigma):
        """Normal PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L300
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        z = (x - mu) / sigma
        fn = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(z, 2))
        return boundzero_numpy(fn, sigma > 0)

    def cdf(self, x, mu, sigma):
        """Normal CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L307
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        z = (x - mu) / sigma
        fn = 0.5 * special.erfc(
            -z / np.sqrt(2)
        )  # or equiv = .5 * (1 + special.erf( z / np.sqrt(2)))
        return boundzero_numpy(fn, sigma > 0)

    def invcdf(self, u, mu, sigma):
        """Normal Inverse CDF aka PPF:
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L319
        ref:
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = mu - sigma * np.sqrt(2.0) * special.erfcinv(2 * u)
        return boundzero_numpy(fn, sigma > 0, u >= 0, u <= 1)

    def logpdf(self, x, mu, sigma):
        """Normal log PDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L304
        ref:
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        z = (x - mu) / sigma
        fn = -np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * np.power(z, 2)
        return boundlog_numpy(fn, sigma > 0)

    def logcdf(self, x, mu, sigma):
        """Normal log CDF
        ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L310
        ref:
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        z = (x - mu) / sigma
        fn = np.log(0.5) + np.log(special.erfc(-z / np.sqrt(2)))
        return boundlog_numpy(fn, sigma > 0)

    def loginvcdf(self, u, mu, sigma):
        """Normal log Inverse CDF aka log PPF
        ref: ?
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = np.log(mu - sigma * np.sqrt(2.0) * special.erfcinv(2 * u))
        return boundlog_numpy(fn, sigma > 0, u >= 0, u <= 1)
