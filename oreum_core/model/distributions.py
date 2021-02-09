# model.distributions.py
# copyright 2021 Oreum OÜ
import numpy as np
import pymc3 as pm
from scipy import stats
import theano.tensor as tt

from pymc3.distributions import transforms
from pymc3.distributions.dist_math import bound, logpow
from pymc3.distributions.continuous import assert_negative_support, PositiveContinuous
from pymc3.distributions.distribution import Continuous, draw_values, generate_samples
from pymc3.theanof import floatX
from pymc3.util import get_variable_name

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


class Gamma(pm.Gamma):
    """Inherit the pymc class, clobber it and add logcdf and loginversecdf
    """
    pass

class GammaNumpy():
    """Gamma manual numpy
       Used to compare my formulations vs scipy
    """
    def __init__(self):
        pass

    def pdf(x, a, b):
        """Gamma PDF"""
        
        return None

    def logpdf(x, a, b):
        """Gamma log PDF"""
        
        return None


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

    def pdf(x, a, s):
        """Inverse Weibull PDF
            forcing m=1
        """
        i = a/s
        j = np.power(x/s, -1. - a)
        k = np.power(x/s, -a)
        k = np.exp(-k)
        return i * j * k

    def logpdf(x, a, s):
        """Inverse Weibull log PDF
            forcing m=1
        """
        ret = np.log(a) + 
            (-1. - a) * np.log(x) + 
            a * np.log(s) - 
            np.power(x/s, -a)
        
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
        

