import numpy as np
from kcsd.potentials import add_2d_gaussians, calculate_potential
from kcsd.sources import CovData
from kcsd.REGCSD import REGCSD2D

xlin = np.linspace(-2, 2, num=100)
xx, yy = np.meshgrid(xlin, xlin)
states = [((0, 0), CovData(np.random.uniform(size=5)))]
true_csd = add_2d_gaussians(xx, yy, states)
R = 1

lxlin = np.linspace(-2, 2, num=2)
measure_locations = np.meshgrid(lxlin, lxlin)
measure_locations = np.dstack(measure_locations).reshape(-1, 2)
pots = calculate_potential(np.stack((xx, yy)), true_csd, measure_locations, h=R, sigma=R)

f = 0
try:
    reg = REGCSD2D(measure_locations, pots, parallel=True)
    reg.cross_validate()
    reg.values()
except:
    f += 1
    print("Parallel REGCSD failed")

try:
    reg = REGCSD2D(measure_locations, pots, parallel=False)
    reg.cross_validate()
    reg.values()
except:
    f += 1
    print("Sequential REGCSD failed")

print("\n\nFAILURES: {}".format(f))
