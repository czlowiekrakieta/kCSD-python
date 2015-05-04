import numpy as np

def step_rescale_2D(xp, yp, mu, R):
    """
    Returns normalized 2D step function.
    **Parameters**
    xp, yp : floats or np.arrays
        point or set of points where function should be calculated

    mu : float
        origin of the function

    R : float
    cutoff range
    """
    s = ((xp-mu[0])**2 + (yp-mu[1])**2 <= R**2)
    s = s / (np.pi*R**2)
    return s        

def gauss_rescale_2D(x, y, mu, three_stdev):
    """
    Returns normalized gaussian 2D scale function
    **Parameters**
    x, y : floats or np.arrays
        coordinates of a point/points at which we calculate the density

    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution
    """
    h = 1./(2*np.pi)
    stdev = three_stdev/3.0
    h_n = h * stdev
    Z = h_n * np.exp(-0.5 * stdev**(-2) * ((x - mu[0])**2 + (y - mu[1])**2))
    return Z


def gauss_rescale_lim_2D(x, y, mu, three_stdev):
    """
    Returns gausian 2D function cut off after 3 standard deviations.
    """
    Z = gauss_rescale_2D(x, y, mu, three_stdev)
    Z *= ((x - mu[0])**2 + (y - mu[1])**2 < three_stdev**2)
    return Z


def make_src_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """
    **Parameters**
    
    X, Y : np.arrays
        points at which CSD will be estimated
    
    n_src : int
        demanded number of sources to be included in the model
    
    ext_x, ext_y : floats
        how should the sources extend the area X, Y
    
    R_init : float
        demanded radius of the basis element
    **Returns**
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

    d = round(R_init/ds)
    R = d * ds

    return X_src, Y_src, R

# def get_src_params_2D_new(Lx, Ly, n_src):
#     V = Lx*Ly
#     V_unit = V/n_src
#     L_unit = V_unit**(0.5)

#     nx = np.ceil(Lx/L_unit)
#     ny = np.ceil(Ly/L_unit)

#     ds = Lx/(nx-1)

#     Lx_n = (nx-1)*ds
#     Ly_n = (ny-1)*ds

#     return (nx, ny, Lx_n, Ly_n, ds)


def get_src_params_2D(Lx, Ly, n_src):
    """
    Helps to distribute n_src sources evenly in a rectangle of size Lx * Ly
    **Parameters**
    Lx, Ly : floats
        lengths in the directions x, y of the area,
        the sources should be placed
    
    n_src : int
        demanded number of sources
    **Returns**
    
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


basis_types = {
    "step": step_rescale_2D,
    "gauss": gauss_rescale_2D,
    "gauss_lim": gauss_rescale_lim_2D,
}

KCSD2D_params = {
    'sigma': 1.0,
    'n_srcs_init': 300,
    'lambd': 0.0,
    'R_init': 0.23,
    'ext_x': 0.0,
    'ext_y': 0.0,
    'h': 1.0,
}
