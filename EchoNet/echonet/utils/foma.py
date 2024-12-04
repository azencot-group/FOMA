import torch
import numpy as np
from intrinsic_dimension import intrinsic_dimension


def get_batch_foma(args, X, Y, latent=False, lam=None):
    """Preprocesses and rescales the input batch using singular value decomposition (SVD) and scaling."""
    X_re, Y_re = reshape_inputs(X, Y)
    m = X_re.shape[-1]

    # Concatenate X and Y for SVD
    Z = torch.concatenate((X_re, Y_re), axis=1)

    # Scale down the concatenated matrix
    Z = scale_down(args, Z, latent, lam)

    # Separate the scaled Z back into X and Y
    X_re, Y_re = Z[:, :m], Z[:, m:]
    X = X_re.reshape(X.shape[0], *X.shape[1:])  # Reshape X back to its original shape

    return X, Y


def reshape_inputs(X, Y):
    """Reshapes inputs X and Y if needed, ensuring they have the correct dimensionality."""
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)  # Flatten all dimensions except the first
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)  # Reshape Y to be a 2D array
    return X, Y


def scale_down(args, Z, latent, lam):
    """Performs singular value decomposition and scales the matrix based on the lambda value and args configuration."""
    U, s, Vt = torch.linalg.svd(Z, full_matrices=False)

    lam = get_lambda(args, lam)

    if args.estimate_id == 0:
        lam_mult = calculate_lam_mult_by_rho(args, s, lam)
    else:
        id_est = estimate_intrinsic_dimension(args, Z, latent)
        lam_mult = apply_lambda_after_id_estimation(s, id_est, lam, args)

    # Rescale the singular values and reconstruct Z
    s = s * lam_mult
    Z = U @ torch.diag(s) @ Vt

    return Z


def get_lambda(args, lam):
    """Generates or converts lambda value based on the args and input."""
    if lam is not None and not torch.is_tensor(lam):
        lam = torch.tensor(lam)
    elif lam is None:
        lam = torch.distributions.beta.Beta(concentration1=args.alpha, concentration0=args.alpha).sample()
    return lam


def calculate_lam_mult_by_rho(args, s, lam):
    """Calculates lambda multiplier based on cumulative percentage of singular values and threshold (rho)."""
    cumperc = torch.cumsum(s, dim=0) / torch.sum(s)
    condition = cumperc > args.rho if args.small_singular else cumperc < args.rho
    return torch.where(condition, lam, 1.0)


def estimate_intrinsic_dimension(args, Z, latent):
    """Estimates intrinsic dimension based on args and whether latent space is used."""
    if args.estimate_id == 1 and not latent:
        return min(args.id, Z.shape[-1] - 1)
    else:
        id_est = intrinsic_dimension(Z)
        return min(int(np.ceil(id_est)), Z.shape[-1] - 1)


def apply_lambda_after_id_estimation(s, id_est, lam, args):
    """Applies lambda scaling after estimating intrinsic dimension."""
    lam_mult = torch.ones(s.shape[-1], device=s.device)
    lam_mult[id_est:] = lam if args.small_singular else lam_mult[id_est:] * lam
    return lam_mult


