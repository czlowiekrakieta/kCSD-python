import numpy as np

from kcsd.potentials import generate_csd, calculate_potential
from kcsd.csd_profile import csd_available_dict

csd_at, csd_profile = generate_csd(csd_available_dict[1][0])

measure_locations = np.linspace(.3, .8, num=25)

potentials = calculate_potential(csd_at, csd_profile, measure_locations, 1)

from kcsd import KCSD1D

kcsd = KCSD1D(measure_locations.reshape(-1, 1), potentials, reg_method='elasticnet')

kcsd.cross_validate()

kcsd.goodness_of_fit(csd_profile)