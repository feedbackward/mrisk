'''Setup: loss functions used for training and evaluation.'''

## External modules.
import numpy as np

## Internal modules.
from mml.losses import Loss
from mml.losses.absolute import Absolute
from mml.losses.classification import Zero_One
from mml.losses.logistic import Logistic
from mml.losses.quadratic import Quadratic


###############################################################################


## All the helper functions for our specialized losses.

def dev(u):
    '''
    This is the function denoted rho(u).
    '''
    return u * np.arctan(u) - np.log1p(u**2)/2.0


def dev_d1(u):
    '''
    First derivative of rho(u).
    '''
    return np.arctan(u)


def dev_d2(u):
    '''
    Second derivative of rho(u).
    '''
    return 1.0/(1.0+u**2)


_sigma_thres = 1.0
_epsilon_cushion = 0.05

def eta_setter(sigma, interpolator=False):
    '''
    Function for setting the eta > 0 value based on positive sigma.
    '''
    if sigma == 0:
        eta = 1+_epsilon_cushion
    elif sigma < _sigma_thres:
        if interpolator:
            ## Option 1:
            ## Ideal setting in terms of exact convergence to abs(u).
            eta = sigma/np.arctan(np.inf)
        else:
            ## Option 2:
            ## A practical setting which ensures the minimum exists.
            eta = (1+_epsilon_cushion)*sigma
    elif sigma < np.inf:
        eta = 2*sigma**2
    else:
        eta = 1.0

    return eta


def dev_ext(u, sigma, interpolator=False, eta_custom=None):
    '''
    This is the extended deviation function rho_sigma(u),
    with the eta > 0 multiplicative factor included in the
    computations, based on the sigma value for convenience.
    '''
    ## Weighting parameter eta.
    if eta_custom is None:
        eta = eta_setter(sigma=sigma, interpolator=interpolator)
    else:
        eta = eta_custom

    ## Sigma-dependent output.
    if sigma == 0:
        return eta * np.absolute(u)
    elif sigma == np.inf:
        return eta * u**2
    else:
        return eta * dev(u=u/sigma)


def dev_ext_d1(u, sigma, interpolator=False, eta_custom=None):
    '''
    This is the subgradient of extended deviation function.
    '''
    ## Weighting parameter eta.
    if eta_custom is None:
        eta = eta_setter(sigma=sigma, interpolator=interpolator)
    else:
        eta = eta_custom

    ## Sigma-dependent output.
    if sigma == 0:
        return eta * np.sign(u)
    elif sigma == np.inf:
        return eta * 2 * u
    elif sigma > 0:
        return eta * dev_d1(u=u/sigma) / sigma
    else:
        raise ValueError("The sigma value is not within expected interval.")
        

_sigma_lower = 0.0001
_sigma_upper = 1e4
def parse_sigma(sigma):
    '''
    A function which sets a lower/upper thresholds for sigma
    values. Anything below/above these thresholds is automatically
    set to 0.0/np.inf respectively.
    '''
    if sigma <= _sigma_lower:
        return 0.0
    elif sigma >= _sigma_upper:
        return np.inf
    else:
        return sigma
    

## The loss class derived from our proposed risk functions.

class M_Risk(Loss):
    '''
    A special loss class that takes a base loss
    object upon construction, and uses that to
    make an unbiased estimator of the "M-Risk",
    i.e., the sum of a shifted M-location and
    deviation term (about that location).
    - loss_base: the base loss object.
    - sigma: a value on the closed interval [0, np.inf].
    - eta: if None, then set automatically based on sigma.
    '''

    def __init__(self, loss_base, sigma, eta=None, name=None):
        loss_name = "M_Risk x {}".format(str(loss_base))
        super().__init__(name=loss_name)
        self.loss = loss_base
        self.sigma = sigma
        self.interpolator = False # set manually.
        self.eta = eta
        return None


    def func(self, model, X, y):
        '''
        '''
        theta = model.paras["theta"].item() # extract scalar.
        return theta + dev_ext(
            u=self.loss(model=model, X=X, y=y)-theta,
            sigma=self.sigma,
            interpolator=self.interpolator,
            eta_custom=self.eta
        )


    def grad(self, model, X, y):
        '''
        '''

        ## Initial computations.
        theta = model.paras["theta"].item() # extract scalar.
        loss_grads = self.loss.grad(model=model, X=X, y=y)
        dev_grads = dev_ext_d1(
            u=self.loss(model=model, X=X, y=y)-theta,
            sigma=self.sigma,
            interpolator=self.interpolator,
            eta_custom=self.eta
        )
        ddim = dev_grads.ndim
        tdim = model.paras["theta"].ndim

        ## Main sub-gradient computations.
        ## Note: since "theta" isn't returned by the model grad
        ##       computing function, we never need to worry about
        ##       "theta" getting mixed into the loop below.
        for pn, g in loss_grads.items():
            gdim = g.ndim
            if ddim > gdim:
                raise ValueError("Axis dimensions are wrong; ddim > gdim.")
            elif ddim < gdim:
                dev_grads_exp = np.expand_dims(
                    a=dev_grads,
                    axis=tuple(range(ddim,gdim))
                )
                g *= dev_grads_exp
            else:
                g *= dev_grads
        
        ## Finally, sub-gradient with respect to theta.
        loss_grads["theta"] = np.expand_dims(
            a=1.0-dev_grads,
            axis=tuple(range(ddim,1+tdim))
        )
        
        ## Return gradients for all parameters being optimized.
        return loss_grads
        

## A dictionary of instantiated losses.

dict_losses = {
    "absolute": Absolute(),
    "logistic": Logistic(),
    "quadratic": Quadratic(),
    "zero_one": Zero_One()
}

def get_loss(name, **kwargs):
    '''
    A simple parser that returns a loss instance.
    '''
    if kwargs["use_mrisk"]:
        return M_Risk(loss_base=dict_losses[name],
                      sigma=kwargs["sigma"])
    else:
        return dict_losses[name]


###############################################################################
