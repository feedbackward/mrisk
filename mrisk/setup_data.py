'''Setup: preparation of data sets.'''

## External modules.
import numpy as np
import os
from pathlib import Path

## Internal modules.
from mml.data import dataset_dict, dataset_list, get_data_general
from mml.models.linreg import LinearRegression
from mml.utils.rgen import get_generator, get_stats


###############################################################################


## If benchmark data is to be used, specify the directory here.
dir_data_toread = os.path.join(str(Path.home()))

## First set dataset parameter dictionary with standard values
## for all the benchmark datasets in mml.
dataset_paras = dataset_dict


## Then add customized parameters for the simulations local to this project.
_n = 500
_n_test = 10**3
_d = 2
_init_range_sims = 5.0
dataset_paras_local = {
    "ds_lognormal": {"noise_dist": "lognormal",
                     "noise_paras": {"mean": 0.0, "sigma": 1.75}},
    "ds_normal": {"noise_dist": "normal",
                  "noise_paras": {"loc": 0.0, "scale": 2.2}},
    "ds_pareto": {"noise_dist": "pareto",
                  "noise_paras": {"shape": 2.1, "scale": 3.5}}
}
for key in dataset_paras_local:
    dataset_paras_local[key].update({"n_train": _n//2,
                                     "n_val": _n//2,
                                     "n_test": _n_test,
                                     "num_features": _d,
                                     "cov_X": np.eye(_d),
                                     "init_range_sims":_init_range_sims})
dataset_paras.update(dataset_paras_local)


## Data generation procedure.

def get_data(dataset, rg):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    if dataset in dataset_paras:
        paras = dataset_paras[dataset]
        if dataset in dataset_list:
            ## Benchmark dataset case.
            return get_data_general(dataset=dataset,
                                    paras=paras, rg=rg,
                                    directory=dir_data_toread)
        else:
            ## Local simulation case.
            return get_data_simulated(paras=paras, rg=rg)
    else:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )


def get_data_simulated(paras, rg):
    '''
    Data generation function.
    This particular implementation is a simple noisy
    linear model.
    '''

    n_train = paras["n_train"]
    n_val = paras["n_val"]
    n_test = paras["n_test"]
    n_total = n_train+n_val+n_test
    d = paras["num_features"]
    
    ## Specifying the true underlying model.
    w_star = np.ones(d).reshape((d,1))
    true_model = LinearRegression(num_features=d,
                                  paras_init={"w":w_star})
    paras.update({"w_star": w_star})
    
    ## Noise generator and stats.
    noise_gen = get_generator(name=paras["noise_dist"],
                              rg=rg,
                              **paras["noise_paras"])
    noise_stats = get_stats(name=paras["noise_dist"],
                            rg=rg,
                            **paras["noise_paras"])
    paras.update({"noise_mean": noise_stats["mean"],
                  "noise_var": noise_stats["var"]})

    ## Data generation.
    X = rg.multivariate_normal(mean=np.zeros(d),
                               cov=paras["cov_X"],
                               size=n_total)
    noise = noise_gen(n=n_total).reshape((n_total,1))-noise_stats["mean"]
    y = true_model(X=X) + noise
    
    ## Split into appropriate sub-views and return.
    X_train = X[0:n_train,:]
    y_train = y[0:n_train,:]
    X_val = X[n_train:(n_train+n_val),:]
    y_val = y[n_train:(n_train+n_val),:]
    X_test = X[(n_train+n_val):,:]
    y_test = y[(n_train+n_val):,:]
    return (X_train, y_train, X_val, y_val, X_test, y_test, paras)


###############################################################################
