from abc import abstractmethod

import numpy as np

from kcsd.KCSD import KCSD, KCSD1D, KCSD2D, KCSD3D
from kcsd.solvers import compute_elasticnet, cv_score, kfold, lcurve_path
import kcsd.utility_functions as utils

from sklearn.linear_model.coordinate_descent import _path_residuals, enet_path, _alpha_grid
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
        self.k_pot = np.vstack((self.interpolate_pot_at[i](self.src_ele_dists) for i in range(self.n_rs))).T
        self.b_src_estimation = np.hstack((self.basis(self.src_estm_dists, r).T for r in self.R))
        self.pot_estimation = np.hstack((self.interpolate_pot_at[i](self.src_estm_dists).T for i in range(self.n_rs)))

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
                                      lasso_reg=self.alpha,
                                      ridge_reg=self.lambd,
                                      max_iters=self.max_iters,
                                      tol=self.tol,
                                      selection=self.selection)
            for i in range(self.n_src):
                estimation[:, t] += estimation_table[:, i] * beta[i]

        if ret_beta:  # debug, will be removed
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

    def init_hyperparams(self, alphas=None, l1_ratios=None, Rs=None):

        if l1_ratios is None and self.l1_ratios is None:
            l1_ratios = [.1, .2, .4, .5, .7, .8, .85, .9, .95, .99]
        elif self.l1_ratios is not None:
            l1_ratios = self.l1_ratios

        n_l1_ratio = len(l1_ratios)
        if alphas is None:
            alphas = []
            for l1_ratio in l1_ratios:
                alphas.append(_alpha_grid(
                    self.k_pot, self.pots[:, 0],
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    eps=self.alpha_eps, n_alphas=self.n_alphas,
                    normalize=False,
                    copy_X=False))
        else:
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))

        if Rs is None:
            Rs = self.all_R

        return alphas, l1_ratios, Rs

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
        :param Rs: list of R values. Can be nested, if you want hierarchical basis.
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

        alphas, l1_ratios, Rs = self.init_hyperparams(alphas=alphas, l1_ratios=l1_ratios, Rs=Rs)

        n_l1_ratio = len(l1_ratios)

        path_params = {'copy_X': False,
                       'fit_intercept': False,
                       'normalize': False,
                       'precompute': False}

        mse_paths = []

        for r, R in enumerate(Rs):
            self.update_R(R)
            print("Cross validating R: ", R)
            X, y = self.k_pot.copy(), self.pots[:, 0].copy()
            jobs = (delayed(_path_residuals)(X, y, train, test, enet_path,
                                             path_params, alphas=this_alphas,
                                             l1_ratio=this_l1_ratio, X_order='F',
                                             dtype=X.dtype.type)
                    for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
                    for train, test in kfold(X.shape[0], self.k_fold_split))

            curr_mse_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                      backend="threading")(jobs)
            mse_paths.append(curr_mse_paths)

        mse_paths = np.reshape(mse_paths, (len(self.all_R), n_l1_ratio, self.k_fold_split, -1))
        mse_paths = mse_paths.mean(axis=2)
        best_idx = np.where(mse_paths == mse_paths.min())
        #
        # for l1_ratio, l1_alphas, mse_alphas in zip(l1_ratios, alphas,
        #                                            mean_mse):
        #     i_best_alpha = np.argmin(mse_alphas)
        #     this_best_mse = mse_alphas[i_best_alpha]
        #     if this_best_mse < best_mse:
        #         best_alpha = l1_alphas[i_best_alpha]
        #         best_l1_ratio = l1_ratio
        #         best_mse = this_best_mse
        self.update_R(self.all_R[best_idx[0][0]])
        self.l1_ratio_ = l1_ratios[best_idx[1][0]]
        self.alpha_ = alphas[best_idx[1][0]][best_idx[2][0]]

        self.alpha = self.alpha_ * self.l1_ratio_ * self.n_ele
        self.lambd = self.alpha_ * (1 - self.l1_ratio_) * self.n_ele
        self.fitted = True

        return self

    def fit_lcurve(self, sentinel=None, Rs=None, alphas=None, l1_ratios=None):

        if sentinel is not None:
            raise TypeError

        alphas, l1_ratios, Rs = self.init_hyperparams(alphas=alphas, l1_ratios=l1_ratios, Rs=Rs)

        cv_scores = []
        cv_params = []

        all_norms = []
        all_resids = []
        for r, R in enumerate(Rs):
            self.update_R(R)
            print("Cross validating R: ", R)
            X, y = self.k_pot.copy(), self.pots[:, 0]
            gram = self.b_src_estimation.T @ self.b_src_estimation
            gram *= (self.xmax - self.xmin)*(self.ymax - self.ymin)/(self.n_estm)
            jobs = (delayed(lcurve_path)(X, y, this_l1_ratio, this_alphas, gram)
                    for this_l1_ratio, this_alphas in zip(l1_ratios, alphas))

            paral_result = Parallel(n_jobs=4, backend='threading', verbose=self.verbose)(jobs)
            areas, norms, resids = zip(*paral_result)
            best_alphas = [this_alphas[ar.argmax()] for this_alphas, ar in zip(alphas, areas)]
            best_alphas = np.array(best_alphas)

            all_norms.append(norms)
            all_resids.append(resids)

            jobs = (delayed(cv_score)(X, y, l1 * alph, (1 - l1) * alph, compute_elasticnet, self.k_fold_split)
                    for l1, alph in zip(l1_ratios, best_alphas))
            scores = Parallel(n_jobs=4, backend='threading', verbose=self.verbose)(jobs)

            idx = np.argmin(scores)
            cv_scores.append(scores[idx])
            cv_params.append((l1_ratios[idx], best_alphas[idx]))

        best_score_idx = np.argmin(cv_scores)
        self.alpha_ = cv_params[best_score_idx][1]
        self.l1_ratio_ = cv_params[best_score_idx][0]

        self.update_alpha(self.n_ele * self.alpha_ * self.l1_ratio_)
        self.update_lambda(self.n_ele * self.alpha_ * (1 - self.l1_ratio_))
        self.update_R(self.all_R[best_score_idx])

        self.fitted = True
        self.all_norms = all_norms
        self.all_resids = all_resids
        return self

    def plot_lcurve(self, R_nr=0, l1_ratio_nr=0):
        import matplotlib.pyplot as plt

        plt.plot(np.exp(self.all_resids[R_nr][l1_ratio_nr]), np.exp(self.all_norms[R_nr][l1_ratio_nr]), 'ro')
        plt.xlabel("RESIDS")
        plt.ylabel("NORMS")
        plt.show()

class REGCSD1D(KCSD1D, REGCSD):
    def __init__(self, ele_pos, pots, **kwargs):
        super(REGCSD1D, self).__init__(ele_pos, pots, **kwargs)


class REGCSD2D(KCSD2D, REGCSD):
    def __init__(self, ele_pos, pots, **kwargs):
        super(REGCSD2D, self).__init__(ele_pos, pots, **kwargs)


class REGCSD3D(KCSD3D, REGCSD):
    def __init__(self, ele_pots, pots, **kwargs):
        super(REGCSD3D, self).__init__(ele_pots, pots, **kwargs)
