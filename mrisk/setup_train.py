'''Setup: training procedure for a single epoch (one pass over data).'''


###############################################################################


def train_epoch(algo, loss, X, y, batch_size=None,
                algo_main=None, verbose=False):

    if verbose:
        print(
            "Training... (num={}).".format(num)
        )
    
    n = len(X)
    idx_start = 0

    ## Cover the full-batch case.
    if batch_size is None or batch_size == 0:
        batch_size = n
    idx_stop = min(batch_size, n)
    
    ## Run the algorithm for one epoch.
    for onestep in algo:
        algo.update(X=X[idx_start:idx_stop,:],
                    y=y[idx_start:idx_stop,:])
        if algo_main is not None:
            algo_main.update()
        idx_start += batch_size
        idx_stop = min(idx_start+batch_size, n)
        to_stop = idx_start >= n
        algo.check(cond=to_stop)
    
    return None


###############################################################################
