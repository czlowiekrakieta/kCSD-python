from numpy.linalg import norm
from kcsd.csd_profile import add_2d_gaussians


def compute_relative_error(true_csd, reconstructed_csd):
    """
    Computes relative error between true CSD and reconstructed one.

    Let T true CSD, R reconstructed

    In continuous case:
    RE = \frac{\int( T - R ) dS}{\int T dS}

    In our, discrete:
    RE = \frac{\norm{T-R}}{\norm{T}}
    :param numpy.ndarray true_csd:
    :param numpy.ndarray reconstructed_csd:
    :return:
    """
    if reconstructed_csd.ndim > true_csd.ndim and reconstructed_csd.shape[-1] == 1:
        reconstructed_csd = reconstructed_csd[:, ..., 0]

    numerator = norm(true_csd - reconstructed_csd)
    denominator = norm(true_csd)
    return numerator / denominator


def compute_rdm(true_csd, reconstructed_csd):
    true_norm = norm(true_csd)
    recon_norm = norm(reconstructed_csd)
    return norm(true_csd/true_norm - reconstructed_csd/recon_norm)


def compute_mag(true_csd, reconstructed_csd):
    return norm(true_csd / reconstructed_csd)


def reconstruct_2D_density(xx, yy, states, regression_coefs):
    N = len(states)
    if len(states) != len(regression_coefs):
        raise Exception
    for i in range(N):
        states[i][1].amplitude *= regression_coefs[i]
    return add_2d_gaussians(xx, yy, states)