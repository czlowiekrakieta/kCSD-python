from abc import abstractmethod

import numpy as np

from kcsd.KCSD import KCSD, KCSD1D, KCSD2D, KCSD3D
from kcsd.solvers import compute_elasticnet, cv_score
import kcsd.utility_functions as utils

from sklearn.linear_model import ElasticNetCV


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


class REGCSD1D(KCSD1D, REGCSD):
    def __init__(self, ele_pots, pots, **kwargs):
        super(REGCSD1D, self).__init__(ele_pots, pots, **kwargs)


class REGCSD2D(KCSD2D, REGCSD):
    def __init__(self, ele_pots, pots, **kwargs):
        super(REGCSD2D, self).__init__(ele_pots, pots, **kwargs)


class REGCSD3D(KCSD3D, REGCSD):
    pass
