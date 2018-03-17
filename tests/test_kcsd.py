import numpy as np
from kcsd.potentials import add_2d_gaussians, calculate_potential
from kcsd.sources import CovData
from kcsd import KCSD2D

xlin = np.linspace(-2, 2, num=100)
xx, yy = np.meshgrid(xlin, xlin)
states = [((0, 0), CovData(np.random.uniform(size=5)))]
true_csd = add_2d_gaussians(xx, yy, states)
R = 1

lxlin = np.linspace(-2, 2, num=2)
measure_locations = np.meshgrid(lxlin, lxlin)
measure_locations = np.dstack(measure_locations).reshape(-1, 2)
pots = calculate_potential(np.stack((xx, yy)), true_csd, measure_locations, h=R, sigma=R)

fails = 0
try:
    kcsd = KCSD2D(measure_locations, pots, parallel=True, solver='ridge', n_jobs=4)
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Parallel Ridge failed.")

try:
    kcsd = KCSD2D(measure_locations, pots, parallel=True, solver='elasticnet', n_jobs=4)
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Parallel Elasticnet failed")

try:
    kcsd = KCSD2D(measure_locations, pots, parallel=True, solver='kernel_ridge', n_jobs=4)
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Parallel Kernel Ridge failed")

try:
    kcsd = KCSD2D(measure_locations, pots, parallel=False, solver='ridge')
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Sequential Ridge failed.")

try:
    kcsd = KCSD2D(measure_locations, pots, parallel=False, solver='elasticnet')
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Sequential Elasticnet failed")

try:
    kcsd = KCSD2D(measure_locations, pots, parallel=False, solver='kernel_ridge')
    kcsd.cross_validate()
    kcsd.values()
except:
    fails += 1
    print("Sequential Kernel Ridge failed")

print("\n\nFAILURES: {}".format(fails))