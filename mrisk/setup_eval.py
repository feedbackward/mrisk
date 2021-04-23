'''Setup: post-training evaluation of performance.'''

## External modules.
import numpy as np

## Internal modules.
from setup_losses import get_loss


###############################################################################


## Evaluation metric parser.

def get_eval(loss_name=None, model_name=None, **kwargs):
    '''
    '''

    eval_dict = {}
    
    if loss_name is None:
        return eval_dict
    
    ## Just use the underlying loss used for evaluation,
    ## not the "mrisk" version of it.
    loss = get_loss(name=loss_name, **{"use_mrisk": False})
    loss_eval = lambda model, X, y: loss(model=model, X=X, y=y)
    eval_dict.update({loss_name: loss_eval})

    ## Now some additional context-specific prep.
    if loss_name == "logistic":
        loss_01 = get_loss(name="zero_one", **{"use_mrisk": False})
        loss_01_eval = lambda model, X, y: loss_01(model=model, X=X, y=y)
        eval_dict.update({"zero_one": loss_01_eval})
    
    return eval_dict


## Evaluation procedures.

def eval_model(epoch, model, storage, data, eval_dict):

    ## Unpack things.
    X_train, y_train, X_test, y_test = data
    store_train, store_test = storage

    ## Carry out relevant evaluations.
    ## Note: special to this "mrisk" project, we will just
    ##       store the mean values based on training data,
    ##       but store *all* the test values. Since the default
    ##       evaluator behaviour here is to just return everything,
    ##       this means we need to wrap it in np.mean() for the
    ##       training data case.
    for key in store_train.keys():
        evaluator = eval_dict[key]
        store_train[key][epoch,0] = np.mean(evaluator(model=model,
                                                      X=X_train,
                                                      y=y_train))
    for key in store_test.keys():
        evaluator = eval_dict[key]
        store_test[key][epoch,:] = evaluator(model=model,
                                             X=X_test,
                                             y=y_test).reshape(-1)
    return None


def eval_models(epoch, models, storage, data, eval_dict):
    '''
    Loops over the model list, assuming enumerated index
    matches the performance array index.
    '''
    for j, model in enumerate(models):
        eval_model(epoch=epoch, model=model, model_idx=j,
                   storage=storage, data=data,
                   eval_dict=eval_dict)
    return None


## Sub-routine for writing to disk.

def eval_write(fname, storage):

    ## Unpack.
    store_train, store_test = storage

    ## Write to disk as desired.
    if len(store_train) > 0:
        for key in store_train.keys():
            np.savetxt(fname=".".join([fname, key+"_train"]),
                       X=store_train[key],
                       fmt="%.7e", delimiter=",")
    if len(store_test) > 0:
        for key in store_test.keys():
            np.savetxt(fname=".".join([fname, key+"_test"]),
                       X=store_test[key],
                       fmt="%.7e", delimiter=",")
    return None


###############################################################################
