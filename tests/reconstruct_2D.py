from kcsd.sources import generate_sources_along_the_curve, generate_sources_on_grid
from kcsd.solvers import cv_score
from kcsd.csd_profile import add_2d_gaussians
from kcsd.potentials import build_kernel_matrix, calculate_potential
from kcsd.utility_functions import distribute_srcs_2D, csd_tuple
from kcsd.KCSD import KCSD2D

import sys
import numpy as np
import pickle
from kcsd import __path__

arg = 'sin'
if len(sys.argv) > 1:
    arg = sys.argv[1]


print('CALCULATING EQUATION: ', arg.upper())
from datetime import datetime
import os

dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
path = os.path.join(__path__[0], 'results_pickles/{}/{}'.format(arg, dt))

if not os.path.exists(path):
    os.makedirs(path)

N_SRC = 1000
x = np.linspace(-2, 2, num=5)
zoo = {
    'sin': np.sin(x ** 2),
    'straight': x * .5 + .5,
    'parabole': x ** 2,
    'cubic': x ** 3,
    'gauss': np.exp(-x ** 2),
}

y = zoo[arg]
real_states, _ = generate_sources_along_the_curve(coordinates=np.vstack((x, y)).T,
                                                  nr_between_points=10,
                                                  with_fakes=True,
                                                  fake_squares=10)
print("GENERATED SOURCES")
xlin = np.linspace(min(x), max(x), num=32)
ylin = np.linspace(min(y), max(y), num=32)
xx_mg, yy_mg = np.meshgrid(xlin, ylin)
grid_states = generate_sources_on_grid(xlin=xlin, ylin=ylin)
xx, yy, R = distribute_srcs_2D(xx_mg, yy_mg, n_src=N_SRC, ext_x=1, ext_y=1, R_init=1)
print("DISTRIBUTED SOURCES")
measure_locations = np.dstack(np.meshgrid(xx[:, 0][::10], yy[0, :][::10])).reshape(-1, 2)
K = build_kernel_matrix(measure_locations=measure_locations, xx=xx, yy=yy,
                        xlin=xx[:, 0], ylin=yy[0, :], dim=2, states=grid_states)
csd_prof = add_2d_gaussians(xx, yy, real_states)
potentials = calculate_potential(np.stack((xx, yy)), csd=csd_prof, measure_locations=measure_locations, h=1)
L_cnt = 10
R_cnt = 10
errors = np.zeros((L_cnt, R_cnt)) - 1

with open(path + '/data.pkl', 'wb') as f:
    t = csd_tuple(kcsd='not_yet_calculated', k_matrix=K, errors=errors, fitting_states=grid_states,
                  potentials=potentials, real_csd=csd_prof, real_states=real_states,
                  xx=xx, yy=yy)
    pickle.dump(t, f)

kcsd = KCSD2D(ele_pos=measure_locations, pots=potentials, csd_at=np.stack((xx, yy)))
kcsd.cross_validate()

with open(path + '/data.pkl', 'wb') as f:
    t = csd_tuple(kcsd=kcsd, k_matrix=K, errors=errors, fitting_states=grid_states,
                  potentials=potentials, real_csd=csd_prof, real_states=real_states,
                  xx=xx, yy=yy)
    pickle.dump(t, f)

N = L_cnt * R_cnt

thresh = 0.05
import time

t0 = time.time()
for i, lasso_reg in enumerate(np.logspace(-10, 1, num=L_cnt)):
    for j, ridge_reg in enumerate(np.logspace(-10, 1, num=R_cnt)):
        try:
            cv = cv_score(K, potentials, lasso_reg=lasso_reg, ridge_reg=ridge_reg)
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


with open(path + '/data.pkl', 'wb') as f:
    t = csd_tuple(kcsd=kcsd, k_matrix=K, errors=errors, fitting_states=grid_states,
                  potentials=potentials, real_csd=csd_prof, real_states=real_states,
                  xx=xx, yy=yy)
    pickle.dump(t, f)