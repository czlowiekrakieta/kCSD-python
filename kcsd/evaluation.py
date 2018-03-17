from numpy.linalg import norm
from kcsd.csd_profile import add_2d_gaussians


def compute_relative_error(true_csd, reconstructed_csd):
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