from abc import abstractmethod

import numpy as np

from kcsd.KCSD import KCSD, KCSD1D, KCSD2D, KCSD3D
from kcsd.solvers import compute_elasticnet, cv_score, kfold
import kcsd.utility_functions as utils

from sklearn.linear_model.coordinate_descent import ElasticNetCV, _path_residuals, enet_path, _alpha_grid
from joblib.parallel import Parallel, delayed


class REGCSD(KCSD):

    NAME = 'regcsd'

    @abstractmethod
    def estimate_at(self):
        raise NotImplementedError

    @abstractmethod
    def place_basis(self):
        raise NotImplementedError

    @abstractmethod
    def create_src_dist_tables(self):
        raise NotImplementedError

    def parameters(self, **kwargs):
        super(REGCSD, self).parameters(**kwargs)
        if self.k_fold_split is None:
            self.k_fold_split = 3

    def method(self):
        self.solver = 'elasticnet'
        self.create_lookup()
        self.k_pot = self.interpolate_pot_at(self.src_ele_dists).T
        self.b_src_estimation = self.basis(self.src_estm_dists, self.R).T
        self.pot_estimation = self.interpolate_pot_at(self.src_estm_dists).T

    def values(self, estimate='CSD', ret_beta=False):
        if estimate == 'CSD':
            estimation_table = self.b_src_estimation
        elif estimate == 'POT':
            estimation_table = self.pot_estimation
        else:
            raise NameError

        if not self.fitted:
            raise utils.NotFittedError

        estimation = np.zeros((self.n_estm, self.n_time))
        for t in range(self.n_time):
            beta = compute_elasticnet(X=self.k_pot,
                                      y=self.pots[:, t],
                                      lasso_reg=self.lambd,
                                      ridge_reg=self.alpha,
                                      max_iters=self.max_iters,
                                      tol=self.tol,
                                      selection=self.selection)
            # import ipdb; ipdb.set_trace()
            for i in range(self.n_src):
                estimation[:, t] += estimation_table[:, i] * beta[i]

        if ret_beta:  # debug
            return self.process_estimate(estimation), beta

        return self.process_estimate(estimation)

    def compute_cverror(self, lambd, alpha):
        error = 0
        for t in range(self.n_time):
            error += cv_score(self.k_pot, self.pots[:, t],
                              lasso_reg=alpha,
                              ridge_reg=lambd,
                              K=self.k_fold_split,
                              solver_fn=compute_elasticnet)
        return error

    def cross_validate(self, sentinel=None, Rs=None, alphas=None, l1_ratios=None):
        """
        Method for searching optimal hyper parameters for REGCSD class. Almost all of the heavy lifting
        is taken from scikit-learn's LinearModelCV and ElasticNetCV in order to fully use parallel
        fast computing. In order to comply with its requirements, redesigning of interface was needed,
        hence l1_ratio instead of lambdas.

        alphas are L1 regularization parameters
        if we denote lambda as L2 regularization, then l1_ratio is alpha/(alpha+lambda).

        if full regularization loss is R = a * L1 + b * L2 for some a and b, then we can write
        alpha = a + b
        l1_ratio = a/(a+b), hence:
        R = alpha * l1_ratio * L1 + alpha * (1 - l1_ratio) * L2

        In this notation it is easier to calculate full regularization path,
        since we hold l1_ratio as constant and fit models for whole sequence of alphas.

        This code is mostly shameless ripoff from sklearn's ElasticNetCV.

        :param None sentinel: raises error if not None - prevents positional arguments.
        :param Rs: actually, it's not used, although it's planned to fit hierarchical models, with different scale of Gauss
        :param np.ndarray|list[Float] alphas: L1 regularization parameters
        :param np.ndarray|list[Float] l1_ratios: l1/(l1+l2) regularization
        :return:
        """
        if sentinel is not None:
            raise TypeError("REGCSD's cross_validate method has different interface than KCSD's one. "
                            "In order to make sure that you are using this with full knowledge of its "
                            "consequences, using positional arguments is not allowed. Please read docstring "
                            "for further information. (Shift+Tab or write ??REGCSD.cross_validate "
                            "in Jupyter Notebook, Ctrl+Q in PyCharm.)")

        if l1_ratios is None and self.l1_ratios is None:
            l1_ratios = [.1, .2, .4, .5, .7, .8, .85, .9, .95, .99]
        elif self.l1_ratios is not None:
            l1_ratios = self.l1_ratios

        n_l1_ratio = len(l1_ratios)
        X, y = self.k_pot.copy(), self.pots[:, 0].copy()
        if alphas is None:
            alphas = []
            for l1_ratio in l1_ratios:
                alphas.append(_alpha_grid(
                    X, y, l1_ratio=l1_ratio,
                    fit_intercept=False,
                    eps=self.alpha_eps, n_alphas=self.n_alphas,
                    normalize=False,
                    copy_X=False))
        else:
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))

        path_params = {'copy_X': False,
                       'fit_intercept': False,
                       'normalize': False,
                       'precompute': False}

        jobs = (delayed(_path_residuals)(X, y, train, test, enet_path,
                                         path_params, alphas=this_alphas,
                                         l1_ratio=this_l1_ratio, X_order='F',
                                         dtype=X.dtype.type)
                for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
                for train, test in kfold(X.shape[0], self.k_fold_split))

        mse_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(jobs)
        mse_paths = np.reshape(mse_paths, (n_l1_ratio, self.k_fold_split, -1))
        mean_mse = np.mean(mse_paths, axis=1)
        self.mse_path_ = np.squeeze(np.rollaxis(mse_paths, 2, 1))
        best_mse = np.inf
        for l1_ratio, l1_alphas, mse_alphas in zip(l1_ratios, alphas,
                                                   mean_mse):
            i_best_alpha = np.argmin(mse_alphas)
            this_best_mse = mse_alphas[i_best_alpha]
            if this_best_mse < best_mse:
                best_alpha = l1_alphas[i_best_alpha]
                best_l1_ratio = l1_ratio
                best_mse = this_best_mse

        self.l1_ratio_ = best_l1_ratio
        self.alpha_ = best_alpha

        self.alpha = best_alpha * best_l1_ratio * self.n_ele
        self.lambd = best_alpha * (1 - best_l1_ratio) * self.n_ele
        self.fitted = True


class REGCSD1D(KCSD1D, REGCSD):
    def __init__(self, ele_pots, pots, **kwargs):
        super(REGCSD1D, self).__init__(ele_pots, pots, **kwargs)


class REGCSD2D(KCSD2D, REGCSD):
    def __init__(self, ele_pots, pots, **kwargs):
        super(REGCSD2D, self).__init__(ele_pots, pots, **kwargs)


class REGCSD3D(KCSD3D, REGCSD):
    pass
