"""
This script is used to generate basis sources for the 
kCSD method Jan et.al (2012) for 3D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Michal Czerwinski, Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
from __future__ import division

import numpy as np
from scipy import interpolate
from collections import namedtuple
import matplotlib.pyplot as plt
try:
    from joblib.parallel import Parallel, delayed
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed


class NotFittedError(Exception):
    pass

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def check_for_duplicated_electrodes(elec_pos):
    """Checks for duplicate electrodes

    Parameters
    ----------
    elec_pos : np.array

    Returns
    -------
    has_duplicated_elec : Boolean
    """
    unique_elec_pos = np.vstack({tuple(row) for row in elec_pos})
    has_duplicated_elec = unique_elec_pos.shape == elec_pos.shape
    return has_duplicated_elec


def distribute_srcs_1D(X, n_src, ext_x, R_init):
    """Distribute sources in 1D equally spaced

    Parameters
    ----------
    X : np.arrays
        points at which CSD will be estimated
    n_src : int
        number of sources to be included in the model
    ext_x : floats
        how much should the sources extend the area X
    R_init : float
        Same as R in 1D case

    Returns
    -------
    X_src : np.arrays
        positions of the sources
    R : float
        effective radius of the basis element
    """
    X_src = np.mgrid[(np.min(X)-ext_x):(np.max(X)+ext_x):np.complex(0,n_src)]
    R = R_init
    return X_src, R


def distribute_srcs_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """Distribute n_src's in the given area evenly

    Parameters
    ----------
    X, Y : np.arrays
        points at which CSD will be estimated
    n_src : int
        demanded number of sources to be included in the model
    ext_x, ext_y : floats
        how should the sources extend the area X, Y
    R_init : float
        demanded radius of the basis element

    Returns
    -------
    X_src, Y_src : np.arrays
        positions of the sources
    nx, ny : ints
        number of sources in directions x,y
        new n_src = nx * ny may not be equal to the demanded number of sources
    R : float
        effective radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lx_n = Lx + 2*ext_x
    Ly_n = Ly + 2*ext_y
    [nx, ny, Lx_nn, Ly_nn, ds] = get_src_params_2D(Lx_n, Ly_n, n_src)
    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2
    X_src, Y_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):np.complex(0,nx),
                            (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):np.complex(0,ny)]
    # d = round(R_init/ds)
    #R = d * ds
    R = R_init
    return X_src, Y_src, R


def get_src_params_2D(Lx, Ly, n_src):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly

    Parameters
    ----------
    Lx, Ly : floats
        lengths in the directions x, y of the area,
        the sources should be placed
    n_src : int
        demanded number of sources
    
    Returns
    -------
    nx, ny : ints
        number of sources in directions x, y
        new n_src = nx * ny may not be equal to the demanded number of sources
    Lx_n, Ly_n : floats
        updated lengths in the directions x, y
    ds : float
        spacing between the sources
    """
    coeff = [Ly, Lx - Ly, -Lx * n_src]
    rts = np.roots(coeff)
    r = [r for r in rts if type(r) is not complex and r > 0]
    nx = r[0]
    ny = n_src/nx
    ds = Lx/(nx-1)
    nx = np.floor(nx) + 1
    ny = np.floor(ny) + 1
    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds
    return (nx, ny, Lx_n, Ly_n, ds)


def distribute_srcs_3D(X, Y, Z, n_src, ext_x, ext_y, ext_z, R_init):
    """Distribute n_src sources evenly in a rectangle of size Lx * Ly * Lz

    Parameters
    ----------
    X, Y, Z : np.arrays
        points at which CSD will be estimated
    n_src : int
        desired number of sources we want to include in the model
    ext_x, ext_y, ext_z : floats
        how should the sources extend over the area X,Y,Z
    R_init : float
        demanded radius of the basis element
    
    Returns
    -------
    X_src, Y_src, Z_src : np.arrays
        positions of the sources in 3D space
    nx, ny, nz : ints
        number of sources in directions x,y,z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources
        
    R : float
        updated radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lz = np.max(Z) - np.min(Z)
    Lx_n = Lx + 2*ext_x
    Ly_n = Ly + 2*ext_y
    Lz_n = Lz + 2*ext_z
    (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n, 
                                                              Ly_n, 
                                                              Lz_n,
                                                              n_src)
    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2
    ext_z_n = (Lz_nn - Lz)/2
    X_src, Y_src, Z_src = np.mgrid[(np.min(X) - ext_x_n):(np.max(X) + ext_x_n):np.complex(0,nx),
                                   (np.min(Y) - ext_y_n):(np.max(Y) + ext_y_n):np.complex(0,ny),
                                   (np.min(Z) - ext_z_n):(np.max(Z) + ext_z_n):np.complex(0,nz)]
    # d = np.round(R_init/ds)
    R = R_init
    return (X_src, Y_src, Z_src, R)


def get_src_params_3D(Lx, Ly, Lz, n_src):
    """Helps to evenly distribute n_src sources in a cuboid of size Lx * Ly * Lz

    Parameters
    ----------
    Lx, Ly, Lz : floats
        lengths in the directions x, y, z of the area,
        the sources should be placed
    n_src : int
        demanded number of sources to be included in the model

    Returns
    -------
    nx, ny, nz : ints
        number of sources in directions x, y, z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources
    Lx_n, Ly_n, Lz_n : floats
        updated lengths in the directions x, y, z
    ds : float
        spacing between the sources (grid nodes)
    """
    V = Lx*Ly*Lz
    V_unit = V / n_src
    L_unit = V_unit**(1./3.)
    nx = np.ceil(Lx / L_unit)
    ny = np.ceil(Ly / L_unit)
    nz = np.ceil(Lz / L_unit)
    ds = Lx / (nx-1)
    Lx_n = (nx-1) * ds
    Ly_n = (ny-1) * ds
    Lz_n = (nz-1) * ds
    return (nx, ny, nz, Lx_n, Ly_n, Lz_n, ds)


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
    """
    ŁM

    Wrapper used for building covariance matrix from set of random numbers.

    """
    def __init__(self, arr, resample=True, print_init=True):
        """
        arr = [angle, rmin, amplitude, sigma_x, sigma_y]

        :param arr:
        :param resample:
        """
        self.cov = np.eye(2)
        self.pi = print_init
        if len(arr.shape) == 1:
            retry = True
            self.trials = 0
            while retry:
                self.init(arr)
                retry = not posdefcheck(self.cov)
                arr = np.random.uniform(size=5)
                self.trials += 1

        elif arr.shape == (2, 2):
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

            sine_sq = np.sin(self.angle)**2
            cos_sq = np.cos(self.angle)**2
            dsine = np.sin(2*self.angle)

            vary = sine_sq/(2*self.sigma_x**2) + cos_sq/(2*self.sigma_y**2)
            varx = sine_sq/(2*self.sigma_y**2) + cos_sq/(2*self.sigma_x**2)
            cov = -dsine/(4*self.sigma_x**2) + dsine/(4*self.sigma_y**2)

            self.cov = np.array([[varx, cov], [cov, vary]])
            # if not np.all(np.linalg.eigvals(self.cov) > 0):
            #     raise np.linalg.LinAlgError("matrix not positive definite: {}".format(self.cov))

        else:
            if hasattr(arr, '__iter__'):
                raise TypeError("arr need to be numpy ndarray of length 5. It is: {}, size: {}".format(arr.__class__, len(arr)))
            else:
                raise TypeError("arr need to be numpy ndarray of length 5. It is: {}".format(type(arr)))

    def __repr__(self):
        msg = "CovData object. Amplitude: {:.2f}, Angle: {:.2f}, VarX: {:.2f}, VarY: {:.2f}"
        return msg.format(self.amplitude, self.angle, self.cov[0, 0], self.cov[1, 1])

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, val):
        """

        :param np.ndarray val:
        :return:
        """
        posdefcheck(val, raise_ex=True)
        self._cov = val


csd_tuple = namedtuple('csd_tuple', ['real_states', 'fitting_states', 'kcsd', 'k_matrix', 'errors', 'potentials', 'real_csd', 'xx', 'yy'])


def plot_csd_2D(xx, yy, true_csd, levels=20, cmap='csd'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if true_csd.ndim == 3:
        true_csd = true_csd[:, :, 0]

    if isinstance(levels, int):
        t_max = np.max(np.abs(true_csd))
        t_max += t_max/1000
        levels = np.linspace(-t_max, t_max, num=levels)
    elif isinstance(levels, np.ndarray):
        levels = levels.ravel()
    else:
        raise TypeError

    if cmap.lower() == 'csd':
        cmap = cm.bwr_r
    elif cmap.lower() == 'pot':
        cmap = cm.viridis
    plt.contourf(xx, yy, true_csd, levels=levels, cmap=cm.bwr_r)


def L_model_fast(k_pot, pots, lamb, i):
    k_inv = np.linalg.inv(k_pot + lamb *
                          np.identity(k_pot.shape[0]))

    beta_new = np.dot(k_inv, pots)
    V_est = np.dot(k_pot, beta_new)
    modelnorm = np.einsum('ij,ji->i', beta_new.T, V_est)
    residual = np.linalg.norm(V_est - pots)
    modelnorm = np.linalg.norm(modelnorm)
    return modelnorm, residual


def parallel_search(k_pot, pots, lambdas, n_jobs=4):
    jobs = (delayed(L_model_fast)(k_pot, pots, lamb, i)
            for i, lamb in enumerate(lambdas))

    modelvsres = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    modelnormseq, residualseq = zip(*modelvsres)
    return modelnormseq, residualseq


def plot_lcurve(residualseq, modelnormseq, imax, curveseq, lambdas, R):
    '''Method for plotting L-curve and triangle areas
    Parameters
    ----------
    residualseq: from L_fit
    modelnormseq: from L_fit
    imax: point index for maximum triangle area
    curveseq: from L_fit - triangle areas
    Lambdas: lambda vector
    R: Radius of basis source

    Shows
    ----------
    Two Plots
    '''
    fig_L = plt.figure()
    ax_L = fig_L.add_subplot(121)
    plt.title('ind_max :' + str(np.round(lambdas[imax], 8)) + ' R: ' + str(R))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Norm of Model", fontsize=20)
    plt.xlabel("Norm of Prediction Error", fontsize=20)
    ax_L.plot(residualseq, modelnormseq, marker=".", c="green")
    ax_L.plot([residualseq[imax]], [modelnormseq[imax]], marker="o", c="red")

    x = [residualseq[0], residualseq[imax], residualseq[-1]]
    y = [modelnormseq[0], modelnormseq[imax], modelnormseq[-1]]
    ax_L.fill(x, y, alpha=0.2)

    ax2_L = fig_L.add_subplot(122)
    plt.xscale('log')
    plt.ylabel("Curvature", fontsize=20)
    plt.xlabel("Norm of Prediction Error", fontsize=20)
    ax2_L.plot(residualseq, curveseq, marker=".", c="green")
    ax2_L.plot([residualseq[imax]], [curveseq[imax]], marker="o", c="red")

    return