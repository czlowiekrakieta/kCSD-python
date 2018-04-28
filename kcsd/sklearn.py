import sys
import warnings

import numpy as np

from sklearn.linear_model.coordinate_descent import cd_fast


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


def _alpha_grid(X, y, Xy=None, l1_ratio=1.0,
                eps=1e-3, n_alphas=100):
    """ Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray, shape (n_samples,)
        Target values

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    """
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    if Xy is None:
        Xy = np.dot(X.T, y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)[::-1]


def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    """Compute elastic net path with coordinate descent

    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : bool or integer
        Amount of verbosity.

    return_n_iter : bool
        whether to return the number of iterations or not.

    positive : bool, default False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    check_input : bool, default True
        Skip input validation checks, including the Gram matrix when provided
        assuming there are handled by the caller when check_input=False.

    **params : kwargs
        keyword arguments passed to the coordinate descent solver.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).

    Notes
    -----
    For an example, see
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.

    See also
    --------
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    ElasticNet
    ElasticNetCV
    """
    # We expect X and y to be already Fortran ordered when bypassing
    # checks

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    if multi_output and positive:
        raise ValueError('positive=True is not allowed for multi-output'
                         ' (y.ndim != 1)')

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             eps=eps, n_alphas=n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    coefs = np.empty((n_features, n_alphas), dtype=X.dtype)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1], dtype=X.dtype))
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            model = cd_fast.enet_coordinate_descent_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
                tol, rng, random, positive)
        elif precompute is False:
            model = cd_fast.enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
                positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


def _path_residuals(X, y, train, test, path, path_params, alphas=None,
                    l1_ratio=1, X_order=None, dtype=None):
    """Returns the MSE for the models computed by 'path'
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values
    train : list of indices
        The indices of the train set
    test : list of indices
        The indices of the test set
    path : callable
        function returning a list of models on the path. See
        enet_path for an example of signature
    path_params : dictionary
        Parameters passed to the path function
    alphas : array-like, optional
        Array of float that is used for cross-validation. If not
        provided, computed using 'path'
    l1_ratio : float, optional
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2
    X_order : {'F', 'C', or None}, optional
        The order of the arrays expected by the path function to
        avoid memory copies
    dtype : a numpy dtype or None
        The dtype of the arrays expected by the path function to
        avoid memory copies
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    fit_intercept = path_params['fit_intercept']
    normalize = path_params['normalize']

    if y.ndim == 1:
        precompute = path_params['precompute']
    else:
        # No Gram variant of multi-task exists right now.
        # Fall back to default enet_multitask
        precompute = False

    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = \
        _pre_fit(X_train, y_train, None, precompute, normalize, fit_intercept,
                 copy=False)

    path_params = path_params.copy()
    path_params['Xy'] = Xy
    path_params['X_offset'] = X_offset
    path_params['X_scale'] = X_scale
    path_params['precompute'] = precompute
    path_params['copy_X'] = False
    path_params['alphas'] = alphas

    if 'l1_ratio' in path_params:
        path_params['l1_ratio'] = l1_ratio

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(X_train, 'csc', dtype=dtype, order=X_order)
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    del X_train, y_train

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    if normalize:
        nonzeros = np.flatnonzero(X_scale)
        coefs[:, nonzeros] /= X_scale[nonzeros][:, np.newaxis]

    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    X_test_coefs = np.dot(X_test, coefs)
    residues = X_test_coefs - y_test[:, :, np.newaxis]
    residues += intercepts
    this_mses = ((residues ** 2).mean(axis=0)).mean(axis=0)

    return this_mses