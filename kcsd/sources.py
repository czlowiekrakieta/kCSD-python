import numpy as np
from random import sample
import warnings


def rstat(seed):
    return np.random.RandomState(seed=seed)


def add_uniform_noise(x, r, seed=0, bounds=None):
    if r > 0:
        np.add(x, np.random.RandomState(seed=seed).uniform(-r, r, size=x.shape), x)
    if bounds is not None:
        np.clip(x, bounds[0], bounds[1], x)


def add_noise_to_covariance(cov, r, seed=0):
    cov[(0, 1), (0, 1)] += rstat(seed).uniform(-r, r, size=2)
    cov[(1, 1), (0, 0)] += rstat(seed//10 + 3*seed).uniform(-r, r, size=1)
    while np.linalg.det(cov) < 0:
        cov[(0, 1), (0, 1)] += rstat(seed).uniform(0, r, size=2)
    return cov


def array_check(x, randomness, shape=None, seed=0):
    if isinstance(x, float):
        x = np.ones(shape=shape)*x
        add_uniform_noise(x, randomness, seed=seed)
        return x
    elif isinstance(x, np.ndarray):
        add_uniform_noise(x, randomness, seed=seed)
        return x
    else:
        raise TypeError

# 1D
def generate_blob_1D(start, stop, nr, randomness, seed=0):
    """

    :param start:
    :param stop:
    :param nr:
    :param np.ndarray randomness:
    :param seed:
    :return:
    """
    L = np.linspace(start=start, stop=stop, num=nr)
    L += randomness
    return L


def generate_states(centers, std, scaling, randomness, seed):
    sh = centers.shape
    std = np.clip(std+randomness, 1e-1, max(std+randomness)) #array_check(std, randomness, sh, seed)
    scaling = scaling+randomness #array_check(scaling, randomness, sh, seed)
    return np.vstack((scaling, centers, std))

#
# def get_dipole(pos_center, neg_center, nr, randomness=0, seed=0, width=.4, std=.5, scaling=1):
#     """
#     function that takes center of positive and negative part of dipole
#     and construct states for gaussian sources around them.
#
#     :param float pos_center:
#     :param float neg_center:
#     :param int nr:
#     :param float randomness:
#     :param int seed:
#     :param float width:
#     :return:
#     """
#     randomness = rstat(seed).uniform(-r, r, size=nr)
#     pos_blob = generate_blob_1D(pos_center - width, pos_center + width,
#                                 nr=nr, randomness=randomness, seed=seed)
#     neg_blob = pos_blob.copy() + (neg_center - pos_center)
#     add_uniform_noise(neg_blob, randomness, seed)
#     centers = np.hstack((pos_blob, neg_blob))
#     sh = centers.shape
#     std = array_check(std, randomness, sh, seed)
#     scaling = array_check(scaling, randomness, sh, seed)
#     scaling[sh[0]//2:] *= -1
#     return np.vstack((scaling, centers, std))
#


def get_dipoles(pm_table, std, scaling, left, right, nr_per_blob=10, real=None, seed=0, randomness=0):
    """

    :param list[int] pm_table: list of 1 and -1, determining sign of source in this place
    :param std:
    :param scaling:
    :param float left: left margin of dipoles array
    :param float right: right margin of dipoles array
    :param list[int] | float real: if list o ints, must be the same size as pm and determines, which dipoles
                                    are real, and which ones are noise. if float, determines proportion of real dipoles.
                                    they will be randomly sampled
    :return:
    """

    # TODO: zmodyfikowac to tak, zeby generowalo od razu kilka/kilkanascie wartosci losowych, ktore
    # dopiero potem beda uzywane do symulowania losowosci

    N = len(pm_table)
    if isinstance(real, float):
        real = np.random.RandomState(seed=seed).uniform(size=N) < real
        real = real.astype(bool)
    else:
        if not (isinstance(real, list) and all([isinstance(x, int) for x in real])):
            raise TypeError

    centers = np.linspace(left, right, num=N)
    width = np.ones(N)*(right - left)/(2*N)
    random_bank = (2*rstat(seed).uniform(size=2*N+N*nr_per_blob*2)-1)*randomness
    centers += random_bank[:N]
    width += random_bank[N:2*N]
    np.clip(width, min(width)**2 if min(width) < 1 else 1e-3, max(width), width)
    true_currents = []
    fake_currents = []
    subbank = random_bank[2*N:]
    for i in range(N):
        blob = generate_blob_1D(centers[i]-width[i], centers[i]+width[i],
                                nr=nr_per_blob,
                                randomness=subbank[i*2*nr_per_blob:i*2*nr_per_blob+nr_per_blob],
                                seed=seed)
        if pm_table[i] == 1:
            states = generate_states(blob, std, np.abs(scaling), subbank[i*2*nr_per_blob + nr_per_blob:(i+1)*2*nr_per_blob], seed)
        else:
            states = generate_states(blob, std, -np.abs(scaling), subbank[i*2*nr_per_blob + nr_per_blob:(i+1)*2*nr_per_blob], seed)

        if real[i]:
            true_currents.append(states)
        else:
            fake_currents.append(states)

    return true_currents, fake_currents


def get_dipoles_with_fakes_outside():
    pass

def get_random_blobs_1D(nr, fake_to_real_ratio):
    pass

# 2D
def posdefcheck(a, raise_ex=False):
    eigs = np.linalg.eigvals(a)
    if np.any(eigs<0):
        if raise_ex:
            raise np.linalg.LinAlgError("Matrix not positive definite: {}. Eigenvalues: {}".format(
                        a, eigs)
                    )
        return False
    return True

class CovData:
    def __init__(self, arr, resample=True):
        if len(arr.shape) == 1:
            retry = True
            self.trials = 0
            while retry:
                self.init(arr)
                retry = not posdefcheck(self.cov)
                arr = np.random.uniform(size=5)

        elif arr.shape == (2, 2):
            posdefcheck(arr, raise_ex=True)
            self.cov = arr
            self.amplitude = 2*np.random.uniform() - 1
        else:
            raise np.linalg.LinAlgError

    def init(self, arr):
        if isinstance(arr, np.ndarray) and len(arr) == 5:
            self.angle = 2*np.pi*arr[0]
            self.rmin = arr[1]/10 + 0.1
            self.rmax = 2*self.rmin
            self.amplitude = 2*arr[2]-1
            self.sigma_x = self.rmin + arr[3]*self.rmin
            self.sigma_y = self.rmin + arr[4]*self.rmin

            sine = np.sin(self.angle)
            cos = np.cos(self.angle)
            dsine = np.sin(2*self.angle)

            var = sine**2/(2*self.sigma_x**2) + cos**2/(2*self.sigma_y**2)
            cov = -dsine/(4*self.sigma_x**2) + dsine/(2*self.sigma_y**2)

            self.cov = np.array([[var, cov], [cov, var]])
            # if not np.all(np.linalg.eigvals(self.cov) > 0):
            #     raise np.linalg.LinAlgError("matrix not positive definite: {}".format(self.cov))

        else:
            if hasattr(arr, '__iter__'):
                raise TypeError("arr need to be numpy ndarray of length 5. It is: {}, size: {}".format(arr.__class__, len(arr)))
            else:
                raise TypeError("arr need to be numpy ndarray of length 5. It is: {}".format(type(arr)))

    def __repr__(self):
        return "CovData object. Angle: {:.2f}, Var: {:.2f}, Cov: {:.2f}".format(self.angle, self.cov[0, 0], self.cov[0, 1])


def get_chessboard():
    pass


def get_tree_with_fakes_outside():
    pass


def get_sinusoid():
    pass


def generate_sources_along_the_curve(coordinates, nr_between_points, seed=0, n_divisions=10, fake_squares=3,
                                     ret_clear=False, with_fakes=False):
    """

    :param np.ndarray coordinates:
    :param int nr_between_points:
    :param int seed:
    :param int n_divisions:
    :param int fake_squares:
    :return:
    """
    N = len(coordinates)
    np.random.seed(seed)
    xmin, xmax = min(coordinates[:, 0]), max(coordinates[:, 0])
    ymin, ymax = min(coordinates[:, 1]), max(coordinates[:, 1])
    xdivs = np.linspace(xmin, xmax, num=n_divisions)
    ydivs = np.linspace(ymin, ymax, num=n_divisions)
    clear_squares = np.ones((n_divisions, n_divisions)).astype(bool)

    states = []
    for i in range(N-1):
        prev = coordinates[i]
        curr = coordinates[i+1]
        # current_randoms = random_bank[i*nr_between_points*10:(i+1)*nr_between_points*10]
        x = prev[0] + np.random.uniform(size=nr_between_points)*(curr[0]-prev[0])
        y = prev[1] + np.random.uniform(size=nr_between_points)*(curr[1]-prev[1])

        small_x, large_x = min(prev[0], curr[0]), max(prev[0], curr[0])
        small_y, large_y = min(prev[1], curr[1]), max(prev[1], curr[1])
        taken_x = np.where((xdivs >= small_x) & (xdivs <= large_x))[0]
        taken_y = np.where((ydivs >= small_y) & (ydivs <= large_y))[0]

        clear_squares[np.ix_(taken_x, taken_y)] = False

        for j in range(nr_between_points):
            cov = CovData(np.random.uniform(size=5))
            states.append((np.array((x[j], y[j])), cov))

    # fake_randoms = random_bank[N*nr_between_points*7:]
    # fakes = generate_blobs_on_2D(centers=fake_centers,
    #                              bounds=[min(xmin, ymin), max(xmax, ymax)],
    #                              real=[0]*fake_squares,
    #                              seed=seed,
    #                              random_bank=np.random.uniform(size=fake_squares*nr_between_points*5 + fake_squares*5),
    #                              nr_per_blob=nr_between_points)

    if with_fakes:
        clear = clear_squares.nonzero()
        clear = list(zip(*clear))
        clear = sample(clear, fake_squares)
        clear = list(zip(*clear))
        clear[0] = np.array(clear[0])
        clear[1] = np.array(clear[1])
        fake_centers = np.vstack((xdivs[clear[0]] + (xmax - xmin)/n_divisions,
                                  ydivs[clear[1]] + (ymax - ymax)/n_divisions)).T
        fakes = generate_blobs_within_boundaries(centers=fake_centers,
                                                 square_idx=clear,
                                                 xdivs=xdivs,
                                                 ydivs=ydivs,
                                                 nr_per_blob=nr_between_points,
                                                 random_bank=np.random.uniform(size=fake_squares*nr_between_points*5 + fake_squares*5))
        return states, fakes
    # else:
    #     return states, fakes, (clear_squares, fake_centers, clear, xdivs, ydivs)
    else:
        return states


def generate_blobs_within_boundaries(centers, square_idx, xdivs, ydivs, nr_per_blob, random_bank=None, seed=0):
    N = len(centers)

    if random_bank is None:
        random_bank = rstat(seed).uniform(size=N*nr_per_blob*5 + N*5)
    elif len(random_bank) != 5*N*nr_per_blob + 5*N:
        raise TypeError("{}, {}, {}".format(len(random_bank), 5*N*nr_per_blob + 5*N, N))

    st = N*nr_per_blob*5
    sources = []
    xdivs = np.hstack((xdivs, [xdivs[-1] + (xdivs[1] - xdivs[0])]))
    ydivs = np.hstack((ydivs, [ydivs[-1] + (ydivs[1] - ydivs[0])]))

    for i in range(N):
        spread = CovData(random_bank[st+5*i:st+5*(i+1)])
        blob_centers = rstat(seed).multivariate_normal(centers[i], spread.cov, size=nr_per_blob)
        np.clip(blob_centers[:, 0], xdivs[square_idx[0][i]], xdivs[square_idx[0][i]+1], blob_centers[:, 0])
        np.clip(blob_centers[:, 1], ydivs[square_idx[1][i]], ydivs[square_idx[1][i]+1], blob_centers[:, 1])
        sm_st = i*nr_per_blob*5
        for j in range(nr_per_blob):
            cd = CovData(random_bank[sm_st + j*5:sm_st + (j+1)*5])
            sources.append((blob_centers[j], cd))

    return sources


def generate_blobs_on_2D(centers, bounds, real, nr_per_blob, seed=0, random_bank=None):
    N = len(centers)

    if random_bank is None:
        random_bank = rstat(seed).uniform(size=N*nr_per_blob*5 + N*5)
    elif len(random_bank) != 5*N*nr_per_blob + 5*N:
        raise TypeError("{}, {}, {}".format(len(random_bank), 5*N*nr_per_blob + 5*N, N))

    # 5 stems from: angle, sigma_x, sigma_y, amplitude, r_min. They describe covariance of single source
    # second part in this sum describes covariance of spread

    st = N*nr_per_blob*5
    true_sources = []
    fake_sources = []
    for i in range(N):
        spread = CovData(random_bank[st+5*i:st+5*(i+1)])
        blob_centers = rstat(seed).multivariate_normal(centers[i], spread.cov, size=nr_per_blob)
        np.clip(blob_centers, bounds[0], bounds[1], blob_centers)
        sm_st = i*nr_per_blob*5
        for j in range(nr_per_blob):
            cd = CovData(random_bank[sm_st + j*5:sm_st + (j+1)*5])
            if real[i]:
                true_sources.append((blob_centers[j], cd))
            else:
                fake_sources.append((blob_centers[j], cd))

    return true_sources, fake_sources


def generate_sources_on_grid(xlin, ylin, seed=0, cov=None):
    np.random.seed(seed)
    xx, yy = np.meshgrid(xlin, ylin)
    n_per_x_axis = xlin.shape[0]
    n_per_y_axis = ylin.shape[0]
    centers = np.dstack([xx, yy]).reshape(-1, 2)
    cds = []
    for i in range(n_per_x_axis):
        for j in range(n_per_y_axis):
            if cov is None:
                cd = np.random.uniform(size=5)
                cd = CovData(cd)
            else:
                cd = CovData(cov)
            cds.append(cd)
    return list(zip(centers, cds))