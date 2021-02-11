# model.distributions.py
# copyright 2021 Oreum OÜ
import numpy as np
import pymc3 as pm
from scipy import stats, special
import theano.tensor as tt

from pymc3.distributions import transforms
from pymc3.distributions.dist_math import bound, logpow
from pymc3.distributions.continuous import assert_negative_support, PositiveContinuous
from pymc3.distributions.distribution import Continuous, draw_values, generate_samples
from pymc3.theanof import floatX
from pymc3.util import get_variable_name

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


class NumpyCopiesOfPymcFns():
    """Just to keep namespace clean"""

    def __init__(self):
        pass

    def alltrue_elemwise(self, conditions):
        """Copy from pymc3
            pymc3.distributions.dist_math.py
        """
        ret = 1
        for c in conditions:
            ret = ret * (1 * c)
        return ret

    def bound(self, vals, *conditions):
        """Copy from pymc.bound
            pymc3.distributions.dist_math.py
        """
        return np.where(self.alltrue_elemwise(conditions), vals, 0)

    def logpow(self, x, m):
        """ Copy from pymc3
            Safe calc log(x**m) since m*log(x) will fail when m, x = 0.
        """
        return np.where(x == 0, np.where(m == 0, 0.0, -np.inf), m * np.log(x))


class Gamma(pm.Gamma):
    """Inherit the pymc class, clobber it and add logcdf and loginversecdf
    """
    # TODO: consider that invCDF is hard to calculate and scipy uses C functions
    # Likely use different dist in practice
    pass

class GammaNumpy():
    """Gamma PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
        Manual implementations used in pymc3 custom distributions
        Helpful to compare these to scipy to confirm my correct implementation
        Ref: https://en.wikipedia.org/wiki/Gamma_distribution
        Params: x > 0, u in [0, 1], a (shape) > 0, b (rate) > 0
    """
    def __init__(self):
        self.npc = NumpyCopiesOfPymcFns()
        self.name = 'Gamma'
        self.notation = {'notation': r'x \sim Gamma(\alpha, \beta)'}
        self.dist_natural = {
            'pdf': r'f(x \mid \alpha, \beta) = \frac{1}{\Gamma(\alpha)} \beta^{\alpha} x^{\alpha-1} e^{- \beta x}',
            'cdf': r'F(x \mid \alpha, \beta) = \frac{1}{\Gamma(\alpha)} \gamma(\alpha, \beta x)',
            'invcdf': r'F^{-1}(u \mid \alpha, \beta) = '}
        self.dist_log = {
            'logpdf': r'\log f(x \mid \alpha, \beta) = -\log \Gamma(\alpha) + \log \beta^{\alpha} + \log x^{\alpha-1} - \beta x',
            'logcdf': r'\log F(x \mid \alpha, \beta) = -\log \Gamma(\alpha) + \log \gamma(\alpha, \beta x)',
            'loginvcdf': r'\log F^{-1}(u \mid \alpha, \beta) = '}
        self.conditions = {
            'parameters': r'\alpha > 0 \, \text{(shape)}, \; \beta > 0 \, \text{(rate)}',
            'support': r'x \in (0, \infty), \; u \sim \text{Uniform([0, 1])}'}
        self.summary_stats = {
            'mean': r'\frac{\alpha}{\beta}',
            'mode': r'\frac{\alpha - 1}{\beta}, \; \text{for} \alpha \geq 1',
            'variance': r'\frac{\alpha}{\beta^{2}}'
        }

    def pdf(self, x, a, b):
        """Gamma PDF
        compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2595
        """
        fn = (1 / special.gamma(a)) * np.power(b, a) * np.power(x, a-1) * np.exp(-b * x)
        return self.npc.bound(fn, a > 0, b > 0, x >= 0)
    
    def cdf(self, x, a, b):
        """Gamma CDF: 
            where $\gamma(a, bx)$ is lower incomplete gamma function [0, lim)
            compare to https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L2602
        """
        # fn = (1 / special.gamma(a)) * special.gammainc(a, b * x)
        fn = special.gammainc(a, b * x)
        return self.npc.bound(fn, a > 0, b > 0, x >= 0)

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
        fn = -special.gammaln(a) + npc.logpow(b, a) + npc.logpow(x, a-1) - b * x
        return self.npc.bound(fn, a > 0, b > 0, x > 0)

    def logcdf(self, x, a, b):
        """Gamma log CDF: 
            where $\gamma(a, bx)$ is lower incomplete gamma function [0, lim)
            compare to https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L2614
        """
        return self.npc.bound((-special.gammaln(a)) + special.gammainc(a, b * x), 
                                a > 0, b > 0, x > 0)
        
    def loginvcdf(self, u, a, b):
        """Gamma log Inverse CDF aka log PPF:
            see sc.gammaincinv()
        """
        raise NotImplementedError('TODO gamma log inverse CDF')


class Gumbel(pm.Gumbel):
    """Inherit the pymc class, clobber it and add logcdf and loginversecdf
    """    

    def test(self):
        raise ValueError('Inside distributions.Gumbel')
    
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
        
        logcdf = -tt.exp(-(value - mu)/beta)

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

        loginvcdf = tt.log(mu) + tt.log(1 - (beta * tt.log(-tt.log(value))/mu))

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
           \frac{\alpha }{s}}\;\left({\frac  {x-m}{s}}\right)^{{-1-\alpha }}\;e^{{-({\frac  {x-m}{s}})^{{-\alpha }}}
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
    Variance  :math:`{\begin{cases}\ s^{2}\left(\Gamma \left(1-{\frac  {2}{\alpha }}\right)-\left(\Gamma \left(1-{\frac  {1}{\alpha }}\right)\right)^{2}\right)&{\text{for }}\alpha >2\\\ \infty &{\text{otherwise}}\end{cases}}`

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

    def __init__(self, alpha=None, s=1., *args, **kwargs):
        super().__init__(*args, defaults=("mode",), **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.s = s = tt.as_tensor_variable(floatX(s))

        # self.mode = s * tt.power(alpha / (floatX(1.) + alpha), floatX(1.) / alpha)
        self.mode = s * tt.power(alpha / (1. + alpha), 1. / alpha)
        
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
            stats.invweibull.rvs, c=alpha, scale=s, loc=0., dist_shape=self.shape, size=size
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
                tt.log(alpha) - 
                tt.log(s) +
                logpow(s / value, 1. + alpha) -
                tt.power(s / value, alpha) # this term grossly dominates if alpha >> 2
                ),
                value > 0.,
                alpha > 0.,
                s > 0. 
            )


class InverseWeibullNumpy():
    """ Inverse Weibull manual numpy
        Used to compare my formulations vs scipy
    """
    def __init__(self):
        pass

    def pdf(self, x, a, s):
        """Inverse Weibull PDF
            forcing m=1
        """
        i = a/s
        j = np.power(x/s, -1. - a)
        k = np.power(x/s, -a)
        k = np.exp(-k)
        return i * j * k

    def logpdf(self, x, a, s):
        """Inverse Weibull log PDF
            forcing m=1
        """
        ret = (np.log(a) + (-1. - a) * np.log(x) + a * np.log(s) - np.power(x/s, -a))
        return ret
    

class Kumaraswamy(pm.Kumaraswamy):
    """Inherit the pymc class, clobber it and add logcdf and loginversecdf
    """

    def test(self):
        raise ValueError('Inside distributions.Kumaraswamy')

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

        logcdf = tt.log(1 - (1 - value ** a) ** b)

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

        loginvcdf = (1/a) * tt.log(1 - (1-value)**(1/b))

        return bound(loginvcdf, value >= 0, value <= 1, a > 0, b > 0)


class LognormalNumpy():
    """Lognormal PDF, CDF, InvCDF and logPDF, logCDF, logInvCDF
        Manual implementations used in pymc3 custom distributions
        Helpful to compare these to scipy to confirm my correct implementation
        Ref: https://en.wikipedia.org/wiki/Log-normal_distribution
        Params: x > 0, u in [0, 1], mu (location) > 0, sigma (variance) > 0
    """
    def __init__(self):
        self.npc = NumpyCopiesOfPymcFns()
        self.name = 'Lognormal'
        self.notation = {'notation': r'x \sim Lognormal(\mu, \sigma)'}
        self.dist_natural = {
            'pdf': r"""f(x \mid \mu, \sigma) = \frac{1}{x \sigma \sqrt{2 \pi}} \exp \left( -{ \frac{(\log{x} - \mu)^{2}}{2 \sigma^{2}}} \right)
                                             = \frac{1}{x \sigma \sqrt{2 \pi}} \exp - \left(\frac{\log{x}-\mu}{\sigma \sqrt{2}} \right)^{2}""",
            'cdf': r"""F(x \mid \mu, \sigma) = \frac{1}{2} \left[ 1 + \text{erf} \left(\frac{\log{x}-\mu}{\sigma \sqrt{2}} \right) \right]
                                             = \frac{1}{2} \text{erfc} \left( \frac{-\log{x} -\mu}{\sigma \sqrt{2}} \right)""",
            'invcdf': r"""F^{-1}(u \mid \mu, \sigma) = \exp \left( \mu + \sigma * \text{normal_invcdf}(u) \right)
                                                     = \exp \left( \mu - \sigma \sqrt{2} \text{erfcinv}(2u) \right)"""}
        self.dist_log = {
            'logpdf': r'\log f(x \mid \mu, \sigma) = - \frac{1}{2 \sigma^2} \log{(x-\mu)^{2}} + \frac{1}{2} \log{\frac{1}{2 \pi \sigma^{2}}} -\log{x}',
            'logcdf': r'\log F(x \mid \mu, \sigma) = \log \left[\frac{1}{2} \text{erfc} \left( \frac{\log{(x)} -\mu}{\sigma \sqrt{2}} \right) \right]',
            'loginvcdf': r'\log F^{-1}(u \mid \mu, \sigma) = \mu - \sigma \sqrt{2} \text{erfcinv}(2u)'}
        self.conditions = {
            'parameters': r'\mu \in (-\infty, \infty) \, \text{(location)}, \; \sigma > 0 \, \text{(variance)}',
            'support': r'x \in (0, \infty), \; u \sim \text{Uniform([0, 1])}'}
        self.summary_stats = {
            'mean': r'\exp \left( \mu +\frac{\sigma^{2}}{2} \right)',
            'mode': r'\exp ( \mu +\sigma^{2} )',
            'variance': r'[\exp (\sigma^{2}) - 1] \exp (2 \mu + \sigma^{2})'
        }

    def pdf(self, x, mu, sigma):
        """Lognormal PDF
            ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5050
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp( -np.power( (np.log(x) - mu) / (sigma * np.sqrt(2)) ,2) )
        return self.npc.bound(fn, sigma > 0, x > 0)
    
    def cdf(self, x, mu, sigma):
        """Lognormal CDF
            ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5057
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        z = (np.log(x) - mu) / sigma
        fn = .5 * special.erfc( -z / np.sqrt(2))
        return self.npc.bound(fn, sigma > 0, x >= 0)

    def invcdf(self, u, mu, sigma):
        """Lognormal Inverse CDF aka PPF:
            ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5063
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = np.exp(mu - sigma * np.sqrt(2) * special.erfcinv(2 * u))
        return self.npc.bound(fn, sigma > 0, u >= 0, u <= 1)

    def logpdf(self, x, mu, sigma):
        """Lognormal log PDF
            ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5054
            ref: https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L1887
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = - np.power(np.log(x)-mu,2) / (2 * np.power(sigma, 2)) + .5 * np.log(1 / (2 * np.pi * np.power(sigma, 2))) - np.log(x)
        return self.npc.bound(fn, sigma > 0, x >= 0)

    def logcdf(self, x, mu, sigma):
        """Lognormal log CDF
            ref: https://github.com/scipy/scipy/blob/ab1c0907fe9255582397db04592d6066745018d3/scipy/stats/_continuous_distns.py#L5060
            ref: https://github.com/pymc-devs/pymc3/blob/41a25d561b3aa40c75039955bf071b9632064a66/pymc3/distributions/continuous.py#L1913
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = np.log(self.cdf(x, mu, sigma))
        return self.npc.bound(fn, sigma > 0, x >= 0)
        
    def loginvcdf(self, u, mu, sigma):
        """Lognormal log Inverse CDF aka log PPF
            ref: ?
        """
        mu = np.float(mu)
        sigma = np.float(sigma)
        fn = mu - sigma * np.sqrt(2) * special.erfcinv(2 * u)
        return self.npc.bound(fn, sigma > 0, u >= 0, u <= 1)


