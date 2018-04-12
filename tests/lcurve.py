from sklearn.datasets import make_regression
from kcsd.solvers import compute_elasticnet
from kcsd.cythonized import cd_fast

import numpy as np
import matplotlib.pyplot as plt
# % matplotlib
# inline

X, y = make_regression(n_features=1000, n_targets=3)

betas = compute_elasticnet(X, y, .5, .5, max_iters=1000)

betas.std(axis=1).max()


def lcurve_path(X, y, l1_ratio, alphas):
    assert y.ndim == 1
    N = X.shape[0]
    l2_regs = N * (1 - l1_ratio) * alphas
    l1_regs = N * l1_ratio * alphas

    rng = np.random.RandomState(seed=0)
    norms = []
    resids = []
    beta = np.zeros(X.shape[1])

    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    for l1, l2 in zip(l1_regs, l2_regs):
        beta = cd_fast.enet_coordinate_descent(beta, l1, l2, X, y, 1000, 1e-4, rng, True, False)[0]

        norms.append(np.linalg.norm(beta))
        resids.append(np.linalg.norm(np.matmul(X, beta) - y))

    return norms, resids


lcurve_path(X, y[:, 0], .2, np.logspace(-10, -1, 100))