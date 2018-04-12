import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time as tm
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

from kcsd.sources import generate_sources_along_the_curve, CovData, generate_sources_on_grid
from kcsd.potentials import build_kernel_matrix, calculate_potential, add_2d_gaussians, integrate_2D, add_1d_gaussians
from kcsd.solvers import compute_elasticnet, regularization_path, cv_score
from kcsd.utility_functions import distribute_srcs_2D
from kcsd.KCSD import KCSD1D, KCSD2D
from kcsd.REGCSD import REGCSD2D
from kcsd.basis_functions import gauss_1D, gauss
from kcsd.evaluation import compute_mag, compute_rdm, compute_relative_error
from kcsd.csd_profile import gauss_1d_dipole, gauss_1d_dipole_f

from sklearn.linear_model import LarsCV, ElasticNetCV

L1_RATIO_SEQUENCE = np.linspace(.1, 1, num=10, endpoint=True)


def build_potentials(xlin, ylin, xx, yy, true_csd, R=1, num_levels=30):
    
    timedict = {}
    measure_locations = np.meshgrid(xlin, ylin)
    measure_locations = np.dstack(measure_locations).reshape(-1, 2)
    pots = calculate_potential(np.stack((xx, yy)), true_csd, measure_locations, h=R, sigma=R)
    return measure_locations, pots

xlin = np.linspace(-2, 2, num=100)
ylin = np.linspace(-2, 2, num=90)
xx, yy = np.meshgrid(xlin, ylin)
xx = xx.T
yy = yy.T
s1 = np.random.uniform(size=5)
s1[2] = 1
s2 = np.random.uniform(size=5)
s2[2] = -1
states = [((-1, 1), CovData(s1)), ((1, -1), CovData(s2))]
true_csd = add_2d_gaussians(xx, yy, states)
R = 1


def plot(xx, yy, est, measure_locations=None):
    if est.ndim == 3:
        est = est[:, :, 0]
    t_max = np.abs(est).max()
    levels = np.linspace(-t_max, t_max, num=25)
    plt.contourf(xx, yy, est, levels=levels, cmap=cm.bwr_r)
    
    if measure_locations is not None:
        plt.scatter(measure_locations[:, 0], measure_locations[:, 1], alpha=.1)

ele_pots, pots = build_potentials(np.linspace(-2, 2, num=7), np.linspace(-2, 2, num=8), 
                                  xx, yy, true_csd)

R = np.array([[.23], [.1, .15, .2, .25, .3], np.linspace(.15, .25, num=10)])

R = np.array([np.array(x) for x in R])

kcsd = REGCSD2D(ele_pos=ele_pots, pots=pots, Rs=0.23)

# import ipdb; ipdb.set_trace()
kcsd.cross_validate()
