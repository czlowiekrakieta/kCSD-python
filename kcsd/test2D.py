import numpy as np
from kcsd.utility_functions import CovData
from kcsd.csd_profile import add_2d_gaussians
from kcsd.sources import generate_sources_along_the_curve

xlin = np.linspace(-2, 2, num=100)
xx, yy = np.mgrid[xlin.min():xlin.max():np.complex(0, xlin.shape[0]),
                  xlin.min():xlin.max():np.complex(0, xlin.shape[0])]
xxp, yyp = np.mgrid[-2:2:np.complex(0, 100),
                    0:4:np.complex(0, 100)]

def rotation_matrix(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def gaussian_monopole(ampl=None, angle=None, varx=None, vary=None):
    for z in [ampl, angle, varx, vary]:
        if z is not None and not 0 < z < 1:
            raise ValueError

    s = np.random.uniform(size=5)
    if ampl is not None:
        s[2] = ampl
    if angle is not None:
        s[0] = angle
    if varx is not None:
        s[3] = varx
    if vary is not None:
        s[4] = vary

    states = [((0, 0), CovData(s))]
    true_csd = add_2d_gaussians(xx, yy, states)
    return true_csd, (xx, yy)


def gaussian_dipole(ampl=None, angle=None, varx=None, vary=None):
    for z in [ampl, angle, varx, vary]:
        if z is not None and not (isinstance(z, (list, np.ndarray)) and len(z) == 2 and all([0 < x < 1 for x in z])):
            raise ValueError

    s = np.random.uniform(size=(2, 5))
    if ampl is not None:
        s[:, 2] = ampl
    if angle is not None:
        s[:, 0] = angle
    if varx is not None:
        s[:, 3] = varx
    if vary is not None:
        s[:, 4] = vary

    if s[0, 2]*s[1, 2] > 0:
        s[0, 2] *= -1

    states = [((-1, 1), CovData(s[0, :])),
              ((1, -1), CovData(s[1, :]))]
    true_csd = add_2d_gaussians(xx, yy, states)
    return true_csd, (xx, yy)


def gaussian_quadrupole(ampl=None, angle=None, varx=None, vary=None):
    for z in [ampl, angle, varx, vary]:
        if z is not None and not (isinstance(z, (list, np.ndarray)) and len(z) == 4 and all([0 < x < 1 for x in z])):
            raise ValueError

    s = np.random.uniform(size=(4, 5))
    if ampl is not None:
        s[:, 2] = ampl
    if angle is not None:
        s[:, 0] = angle
    if varx is not None:
        s[:, 3] = varx
    if vary is not None:
        s[:, 4] = vary

    states = [((-1, -1), CovData(s[0, :])),
              ((1, -1), CovData(s[1, :])),
              ((-1, 1), CovData(s[2, :])),
              ((1, 1), CovData(s[3, :]))]
    return add_2d_gaussians(xx, yy, states), (xx, yy)


def parabole_rough():
    y = xlin**2
    xxp, yyp = np.mgrid[-2:2:np.complex(0, 100),
               0:4:np.complex(0, 100)]
    states = generate_sources_along_the_curve(np.vstack((xlin, y)).T, 10, with_fakes=False)
    return add_2d_gaussians(xxp, yyp, states), (xxp, yyp)


def parabole_smooth(k=5):
    y = xlin ** 2
    xxp, yyp = np.mgrid[-2:2:np.complex(0, 100),
               0:4:np.complex(0, 100)]

    variance = np.matrix([[1, 0], [0, 30]])
    states_new_parab = []
    coords = np.vstack((xlin, y)).T

    for i in range(y.shape[0] - 1):
        ampl = np.random.uniform(size=k)
        angle = np.arctan((y[i + 1] - y[i]) / (xlin[i + 1] - xlin[i]))
        rot = rotation_matrix(angle)
        ts = generate_sources_along_the_curve(coords[i:i + 2, :],
                                              nr_between_points=k, spread_cov=rot @ variance @ rot.T, ampl=ampl)
        states_new_parab += ts

    # pstates_new = generate_sources_along_the_curve(np.vstack((xlin, y)).T, 10, with_fakes=False)
    true_csd = add_2d_gaussians(xxp, yyp, states_new_parab)
    return true_csd, (xxp, yyp)


def straight_line():
    y = xlin * .5 + .5
    xxs, yys = np.mgrid[-2:2:np.complex(0, 100),
                        -0.5:1.5:np.complex(0, 100)]
    states = generate_sources_along_the_curve(np.vstack((xlin, y)).T, 10, with_fakes=False)
    true_csd_S = add_2d_gaussians(xxs, yys, states)
    return true_csd_S, (xxs, yys)


def sin_rough():
    pass


def sin_smooth():
    pass


def neuron():
    neuron_coordinates = [
        [(5, 0), (2, -2)], [(7, 1), (2, -2)],
        [(-5, 0), (-2, -2)], [(-8, 2), (-2, -2)],
        [(2, -2), (0, -5)], [(-2, -2), (0, -5)],
        [(0, -5), (1, -10)],
        [(1, -10), (4, -20)], [(1, -10), (-4, -20)]
    ]

    variance = np.matrix([[1, 0], [0, 10]])
    states = []
    n = np.array(neuron_coordinates)

    for i, nodes in enumerate(neuron_coordinates):
        ampl = np.random.uniform(size=20) / 10
        ampl += 1 if i > n.shape[0] // 2 else -1
        angle = np.arctan((nodes[1][1] - nodes[0][1]) / (nodes[1][0] - nodes[0][0]))
        rot = rotation_matrix(angle)
        ts = generate_sources_along_the_curve(np.array(nodes), spread_cov=rot @ variance @ rot.T, with_fakes=False,
                                              nr_between_points=20, ampl=ampl)
        states += ts

    xn = n[:, :, 0].flatten()
    yn = n[:, :, 1].flatten()
    xxn, yyn = np.mgrid[xn.min():xn.max():np.complex(0, 100),
                        yn.min():yn.max():np.complex(0, 100)]
    true_csd_neur = add_2d_gaussians(xxn, yyn, states)
    return true_csd_neur, (xxn, yyn)


zoo = {
    'neuron': neuron,
    'straight': straight_line,
    'dipole': gaussian_dipole,
    'monopole': gaussian_monopole,
    'quadrupole': gaussian_quadrupole,
}