from kcsd.sources import generate_sources_along_the_curve, generate_sources_on_grid
from kcsd.solvers import compute_elasticnet, cv_score
from kcsd.csd_profile import add_2d_gaussians
from kcsd.potentials import build_kernel_matrix, calculate_potential
from kcsd.utility_functions import distribute_srcs_2D, csd_tuple
from kcsd.KCSD import KCSD2D

import sys
import numpy as np
import pickle

neuron_coordinates = [
    [(5, 0), (2, -2)], [(7, 1), (2, -2)],
    [(-5, 0), (-2, -2)], [(-8, 2), (-2, -2)],
    [(2, -2), (0, -5)], [(-2, -2), (0, -5)],
    [(0, -5), (1, -10)],
    [(1, -10), (4, -20)], [(1, -10), (-4, -20)]
]


def rotation_matrix(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


variance = np.matrix([[100, 0], [0, 1]])
all_states = []
n = np.array(neuron_coordinates)

for i, nodes in enumerate(neuron_coordinates):
    ampl = np.random.uniform(size=20)/10
    ampl += 1 if i > n.shape[0] // 2 else -1
    angle = (nodes[1][1] - nodes[0][1])/(nodes[1][0] - nodes[0][0])
    rot = rotation_matrix(angle)
    states = generate_sources_along_the_curve(np.array(nodes), spread_cov=rot @ variance @ rot.T, with_fakes=False,
                                              nr_between_points=20, ampl=ampl)
    all_states += states

xn = n[:, :, 0].flatten()
yn = n[:, :, 1].flatten()
xlin = np.linspace(xn.min(), xn.max(), num=100)
ylin = np.linspace(yn.min(), yn.max(), num=100)
xx_mg, yy_mg = np.meshgrid(xlin, ylin)
xx, yy, R = distribute_srcs_2D(xx_mg, yy_mg, n_src=1000, ext_x=1, ext_y=1, R_init=1)
measure_locations = np.dstack(np.meshgrid(xx[:, 0][::10], yy[0, :][::10])).reshape(-1, 2)
grid_states = generate_sources_on_grid(xx[:, 0], yy[0, :])
K_grid = build_kernel_matrix(measure_locations, states=grid_states, xx=xx, yy=yy, xlin=xx[:, 0], ylin=yy[0, :],
                             verbose=True)
csd_prof = add_2d_gaussians(xx, yy, all_states)
potentials = calculate_potential(np.stack((xx, yy)), csd=csd_prof, measure_locations=measure_locations, h=1)

print(measure_locations)

kcsd = KCSD2D(ele_pos=measure_locations, pots=potentials)
kcsd.cross_validate()


L_cnt = 10
R_cnt = 10
errors = np.zeros((L_cnt, R_cnt)) - 1
thresh = .05
N = L_cnt*R_cnt
import time
t0 = time.time()
for i, lasso_reg in enumerate(np.logspace(-10, 1, num=L_cnt)):
    for j, ridge_reg in enumerate(np.logspace(-10, 1, num=R_cnt)):
        try:
            cv = cv_score(K_grid, potentials, lasso_reg=lasso_reg, ridge_reg=ridge_reg)
            errors[i, j] = cv
        except TypeError:
            continue

        cnt = (i * R_cnt + j) + 1
        tc = time.time() - t0
        dt = tc / cnt
        if cnt / N > thresh:
            print("Progress: {}. Time elapsed: {}. Time estimated: {}".format(int(100 * thresh),
                                                                              tc, (N - cnt) * dt))
            thresh += .05

from datetime import datetime
import os
import kcsd as kmod
dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
path = os.path.join(kmod.__path__[0], 'results_pickles/neuron/{}'.format(dt))

if not os.path.exists(path):
    os.makedirs(path)

with open(path + '/data.pkl', 'wb') as f:
    t = csd_tuple(kcsd=kcsd, k_matrix=K_grid, errors=errors, real_states=all_states,
                  potentials=potentials, real_csd=csd_prof, fitting_states=grid_states,
                  xx=xx, yy=yy)
    pickle.dump(t, f)
