import numpy as np
from copy import deepcopy


def soft_threshold(x, a):
    absx = abs(x)
    if absx <= a:
        return 0
    if x > 0:
        return x - a
    if x < 0:
        return x + a


def compute_elasticnet(X, y, lasso_reg, ridge_reg, max_iters, tol=1e-3, ret_loss=False):
    """
    Computes ElasticNet estimates for kCSD method

    TODO: przyspieszyc obliczenia przez wspieranie sie poprzednimi residuami
    TODO: przepisac na Cythona

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

    beta = np.zeros((X.shape[1], 1))
    prev_loss = np.inf
    losses = []
    for i in range(max_iters):
        for j in range(X.shape[1]):
            temp_beta = deepcopy(beta)
            temp_beta[j] = 0.0

            resid = y - np.matmul(X, temp_beta)
            numerator = soft_threshold(np.multiply(X[:, j], resid).sum(), X.shape[0]*lasso_reg)
            denominator = norms[j] + ridge_reg

            # print("num: ", numerator, "denom: ", denominator)

            beta[j] = numerator / denominator

        loss = np.square(y - np.matmul(X, beta)).sum()
        losses.append(loss)
        if prev_loss - loss < tol:
            break
        prev_loss = loss
    if ret_loss:
        return beta, losses

    return beta


def compute_lasso(X, y, lasso_reg, max_iters, tol=1e-3, *args, **kwargs):
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

    return compute_elasticnet(X, y, lasso_reg=lasso_reg, ridge_reg=0, max_iters=max_iters, tol=tol)


def compute_ridge(X, y, ridge_reg, *args, **kwargs):
    """
    Computes ridge regression estimates in a closed-form solution.
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

    identity = np.identity(X.shape[0])
    B = X + ridge_reg*identity
    return np.dot(B.I, np.matrix(y))


opt_zoo = {
    'ridge': compute_ridge,
    'elasticnet': compute_elasticnet,
    'lasso': compute_lasso
}