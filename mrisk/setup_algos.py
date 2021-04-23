'''Setup: algorithms.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos import Algorithm
from mml.algos.gd import GD_ERM
from mml.algos.rgd import RGD_Mest
from mml.utils.mest import est_loc_fixedpt, inf_tanh, est_scale_chi_fixedpt, chi_geman_quad


###############################################################################


class Weighted_Average(Algorithm):
    '''
    An Algorithm for sequentially computing
    the simplest possible sequential weighted
    average of parameter candidates.
    '''
    
    def __init__(self, model_main=None, model_ancillary=None):
        super().__init__(model=model_main, loss=None, name=None)
        self.model_ancillary = model_ancillary
        self.weight_sum = 1.0
        return None


    def update(self, X=None, y=None):
        for pn, p in self.paras.items():
            p *= self.weight_sum
            p += self.model_ancillary.paras[pn]
            p /= self.weight_sum + 1.0
        self.weight_sum += 1.0
        return None


## Detailed setup for algorithms.

_inf_fn = inf_tanh # influence function.
_est_loc = lambda X, s, thres, iters: est_loc_fixedpt(X=X, s=s,
                                                      inf_fn=_inf_fn,
                                                      thres=thres,
                                                      iters=iters)
_chi_fn = chi_geman_quad # chi function.
_est_scale = lambda X: est_scale_chi_fixedpt(X=X, chi_fn=_chi_fn)


## Simple parser for algorithm objects.

def get_algo(name, model, loss, name_main=None, model_main=None, **kwargs):

    if name == "SGD":
        algo = GD_ERM(step_coef=kwargs["step_size"],
                      model=model,
                      loss=loss)
    elif name == "RGD-M":
        algo = RGD_Mest(est_loc=_est_loc,
                        est_scale=_est_scale,
                        delta=0.01,
                        mest_thres=1e-03,
                        mest_iters=50,
                        step_coef=kwargs["step_size"],
                        model=model,
                        loss=loss)
    else:
        raise ValueError("Please pass a valid algorithm name.")
    
    if name_main is None or name_main == "":
        return (algo, None)
    else:
        if name_main == "Ave":
            algo_main = Weighted_Average(model_main=model_main,
                                         model_ancillary=model)
        else:
            raise ValueError("Please pass a valid main algorithm name.")
        return (algo, algo_main)

                      

###############################################################################
