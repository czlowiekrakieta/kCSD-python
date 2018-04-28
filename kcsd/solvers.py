import numpy as np
from copy import deepcopy
from random import sample
from scipy import linalg
try:
    from joblib.parallel import Parallel, delayed
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed
finally:
    import warnings
    warnings.warn("Could not import Parallel and delayed from joblib library.", ImportWarning)

from kcsd.cythonized import cd_fast
# ignore PyCharm screaming about unresolved reference, it works :) ŁM


def soft_threshold(x, a):
    absx = abs(x)
    if absx <= a:
        return 0
    if x > 0:
        return x - a
    if x < 0:
        return x + a


def compute_elasticnet(X, y, lasso_reg, ridge_reg, max_iters=100, tol=1e-3, selection='random', seed=0):
    """
    Human-friendly wrapper for Cython optimizer from sklearn.

    :param np.ndarray X:
    :param np.ndarray y:
    :param float lasso_reg:
    :param float ridge_reg:
    :param int max_iters:
    :param float tol:
    :param str selection:
    :param int seed:
    :return:
    """
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    coef = np.zeros(X.shape[1], dtype=X.dtype)
    rng = np.random.RandomState(seed=seed)
    if y.ndim == 2:
        T = y.shape[1]
        return np.vstack([cd_fast.enet_coordinate_descent(coef, lasso_reg, ridge_reg,
                                                          X, y[:, t], max_iters, tol, rng,
                                                          selection == 'random', False)[0]
                          for t in range(T)]).T
    else:
        return cd_fast.enet_coordinate_descent(coef, lasso_reg, ridge_reg, X, y, max_iters, tol,
                                               rng, selection == 'random', False)[0]


def slow_compute_elasticnet(X, y, lasso_reg, ridge_reg, max_iters, tol=1e-3, ret_loss=False, selection='random'):
    """
    Computes ElasticNet estimates for kCSD method

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    lasso_reg : float
    ridge_reg : float
    max_iters : int
    tol : float

    Returns
    -------
    beta : np.ndarray
        fitted coefficients
    """

    norms = np.square(X).sum(axis=0)
    norms = np.array(norms).flatten()

    beta = np.zeros(X.shape[1])
    prev_loss = np.inf
    losses = []

    if selection not in {'random', 'cyclic'}:
        raise NameError

    if selection == 'random':
        giver = lambda: sample(range(X.shape[1]), X.shape[1])
    else:
        giver = lambda: range(X.shape[1])

    for i in range(max_iters):
        for j in giver():
            temp_beta = deepcopy(beta)
            temp_beta[j] = 0.0

            resid = y - np.matmul(X, temp_beta)
            assert len(resid.shape) == 1 or (len(resid.shape) == 2 and resid.shape[1] == 1)
            numerator = soft_threshold(np.multiply(X[:, j], resid).sum(), X.shape[0]*lasso_reg)
            denominator = norms[j] + ridge_reg
            if numerator is None or np.isnan(resid).any():
                print(j,
                      X.shape,
                      np.nanmax(temp_beta),
                      temp_beta.shape,
                      np.isnan(resid).any(),
                      np.where(np.isnan(np.matmul(X, temp_beta))),
                      X.shape[0]*lasso_reg,
                      norms[j],
                      np.square(y - np.matmul(X, temp_beta)).sum())
                raise Exception

            beta[j] = numerator / denominator

        loss = np.square(y - np.matmul(X, beta)).sum()
        losses.append(loss)
        if prev_loss - loss < tol:
            break
        prev_loss = loss
    if ret_loss:
        return beta, losses

    return beta


def compute_lasso(X, y, lasso_reg, max_iters, tol=1e-3, selection='cyclic', *args, **kwargs):
    """
    Computes LASSO estimates for kCSD method, using ElasticNet with ridge parameter equal to zero.
    Nothing more than a wrapper function.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    alpha : float
    max_iters : int

    Returns
    -------
    beta : np.ndarray
        fitted coefficients
    """

    return compute_elasticnet(X, y, lasso_reg=lasso_reg, ridge_reg=0, max_iters=max_iters, tol=tol, selection=selection)


def compute_kernel_ridge(X, y, ridge_reg, *args, **kwargs):
    """
    Computes kernel ridge regression estimates in a closed-form solution.
    If you want to do it using coordinate descent algorithm, simply call
    compute_elasticnet with lasso_reg = 0

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    ridge : float

    Returns
    -------
    beta : np.ndarray
    """
    X = np.matrix(X)
    identity = np.identity(X.shape[0])
    B = X + ridge_reg*identity
    beta = np.dot(np.linalg.inv(B), y)
    return np.array(beta).reshape(-1, 1)


def compute_ridge(X, y, ridge_reg, *args, **kwargs):
    """
    Solves ridge regression (for real :) )
    :param X:
    :param y:
    :param alpha:
    :return:
    """
    REG_MAT = np.eye(X.shape[1])*ridge_reg
    A = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    return np.dot(linalg.inv(A+REG_MAT), Xy)


def kfold(N, k=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    permut = np.random.permutation(N)

    if k is None:
        k = N
    elif k > N:
        raise Exception("Can't have more folds than data samples.")
    elif k < 2:
        raise Exception("Must have at least 2 folds.")

    s = N // k
    for i in range(k):
        test_idx = permut[i*s:(i+1)*s]
        train_idx = np.setdiff1d(permut, test_idx)
        yield train_idx, test_idx


def cv_score(X, y, lasso_reg, ridge_reg, solver_fn, K=None, seed=None, max_iters=100):
    """
    TODO: decyzja, czy zakladamy, ze parametry regularyzacji sa juz przemnozone przez N, czy je mnozymy?
    chyba lepiej podawac przed przemnozeniem, bo tutaj jest zmienne N

    :param np.ndarray X:
    :param np.ndarray y:
    :param float lasso_reg:
    :param float ridge_reg:
    :param solver_fn:
    :param int K:
    :param seed:
    :param int max_iters:
    :return:
    """
    error = 0
    N = len(y)
    if K is None:
        K = N

    if y.ndim == 2:
        T = y.shape[1]
    else:
        y = y.reshape(-1, 1)
        T = 1
    for t in range(T):
        for train, test in kfold(N, K, seed=seed):
            n_f = train.size
            beta_ = solver_fn(X[train, :], y[train, t],
                              lasso_reg=lasso_reg * n_f,
                              ridge_reg=ridge_reg * n_f,
                              max_iters=max_iters)
            error += (np.matmul(X[test, :], beta_) - y[test]) ** 2

    return error.sum() / K


def train_and_test(X_train, y_train, X_test, y_test, lasso_reg, ridge_reg, solver_fn, tol=1e-3, max_iters=100, **kwargs):
    """

    :param np.ndarray X:
    :param np.ndarray y:
    :param np.ndarray train:
    :param np.ndarray test:
    :param float lasso_reg:
    :param float ridge_reg:
    :param solver_fn:
    :param kwargs:
    :return:
    """
    error = 0
    T = y_train.shape[1]
    for t in range(T):
        beta = solver_fn(X_train, y_train[:, t], lasso_reg=lasso_reg,
                         ridge_reg=ridge_reg, tol=tol, max_iters=max_iters, **kwargs)
        error += np.square(np.matmul(X_test, beta) - y_test[:, t]).sum()
    return error


def parallel_search(X, y, K, alphas, lambdas, solver, n_jobs=4, method='kcsd'):
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if method not in {'kcsd', 'regcsd'}:
        raise NameError

    if method == 'kcsd':
        solver_fn = solvers[solver]
        jobs = (delayed(train_and_test)(X[np.ix_(train, train)], y[train, :], X[np.ix_(test, train)], y[test, :],
                                        l, r, solver_fn)
                for r in lambdas for l in alphas
                for train, test in kfold(X.shape[0], k=K))
    else:
        jobs = (delayed(train_and_test)(X[train, :], y[train, :], X[test, :], y[test, :],
                                        l, r, compute_elasticnet)
                for l in alphas for r in lambdas
                for train, test in kfold(X.shape[0], k=K))

    errs = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    errs = np.array(errs)
    errs = errs.reshape(1, -1, lambdas.size, alphas.size)
    errs = errs.sum(axis=1)
    return errs


def lcurve_path(X, y, l1_ratio, alphas, gram_matrix):
    N = X.shape[0]
    A = alphas.shape[0]

    l2_regs = N * (1 - l1_ratio) * alphas
    l1_regs = N * l1_ratio * alphas

    rng = np.random.RandomState(seed=0)
    norms = np.zeros(A)
    resids = np.zeros(A)
    beta = np.zeros(X.shape[1])

    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    for i, (l1, l2) in enumerate(zip(l1_regs, l2_regs)):
        beta = cd_fast.enet_coordinate_descent(beta, l1, l2, X, y, 1000, 1e-4, rng, True, 0)[0]
        norms[i] = beta.T @ gram_matrix @ beta
        resids[i] = np.linalg.norm(np.matmul(X, beta) - y)

    norms = np.log(norms + np.finfo(np.float64).eps)
    resids = np.log(resids + np.finfo(np.float64).eps)

    areas = resids[0] * (norms - norms[-1]) + resids * (norms[-1] - norms[0]) + resids[-1] * (norms[0] - norms)

    return areas, norms, resids


def regularization_path(X, y, alphas=None, n_alphas=100, cv=False, method='lasso'):
    if alphas is None:
        alphas = np.logspace(-10, 1, num=n_alphas)

    if method not in ['lasso', 'ridge']:
        raise ValueError

    coefs = []
    N = X.shape[0]
    errors = []
    for alpha in alphas:
        if method == 'lasso':
            beta = compute_elasticnet(X, y, lasso_reg=alpha, ridge_reg=0, max_iters=100)
        else:
            beta = compute_elasticnet(X, y, lasso_reg=0, ridge_reg=alpha, max_iters=100)
        coefs.append(beta)

        if cv and method == 'lasso':
            errors.append(cv_score(X, y, lasso_reg=alpha))
        elif cv and method == 'ridge':
            errors.append(cv_score(X, y, ridge_reg=alpha))

    if cv:
        return alphas, np.hstack(coefs), errors

    return alphas, np.hstack(coefs)


solvers = {
    'kernel_ridge': compute_kernel_ridge,
    'ridge': compute_ridge,
    'elasticnet': compute_elasticnet,
    'lasso': compute_lasso
}
