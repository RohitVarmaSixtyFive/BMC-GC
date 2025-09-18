"""
Math utilility functions for Augumented Lagrangian Method for Low-Rank Euclidean Distance Matrix Completion
"Exact Reconstruction of Euclidean Distance Geometry Problem Using Low-rank Matrix Completion" by Abiy Tasissa, Rongjie Lai et al.
Original Code in MATLAB : https://github.com/abiy-tasissa/Nonconvex-Euclidean-Distance-Geometry-Problem-Solver
"""
import numpy as np
import torch
import warnings

def recon_from_points(P, squared=True, eps=1e-12):
    """
    Reconstruct distances from point coordinates P (shape [n, d] or (n, r)).
    Accepts numpy arrays (or things convertible to numpy).
    Returns squared distances if squared=True, else Euclidean distances.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # If the above are True, you can see where they are:
    if np.isnan(P).any() or np.isinf(P).any():
        print("Locations of NaN values:", np.argwhere(np.isnan(P)))
    
    P = np.asarray(P, dtype=float)
    sq_norms = np.sum(P * P, axis=1)
    G = P @ P.T
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * G
    # numeric cleanup
    if eps is not None and eps > 0:
        # keep tiny negatives as zero (use logical_and to avoid bitwise on arrays)
        D_sq = np.where(np.logical_and(D_sq < 0, D_sq > -eps), 0.0, D_sq)
        D_sq = np.maximum(D_sq, 0.0)
    if squared:
        return D_sq
    else:
        return np.sqrt(D_sq)


def A_operator(X, edgeind, I, J, dim):
    """Linear operator A for distance matrix completion."""
    X_diag = torch.sum(X**2, dim=1)
    X_offdiag = torch.sum(X[I, :] * X[J, :], dim=1)
    return X_diag[I] + X_diag[J] - 2 * X_offdiag


def At_operator(y, edgeind, dim):
    """Adjoint operator A^T for distance matrix completion."""
    device = y.device
    X_flat = torch.zeros(dim**2, device=device)
    X_flat[edgeind] = -2 * y
    X = X_flat.view(dim, dim)
    X_diag = -torch.sum(X, dim=1)
    X = X.clone()
    X[torch.arange(dim), torch.arange(dim)] = X_diag
    return X


def gradient(P, edgeind, I, J, dim, b, D1, r):
    """Compute gradient for the augmented Lagrangian formulation."""
    tmp = A_operator(P, edgeind, I, J, dim) - b + D1
    F = torch.sum(P**2) + 0.5 * r * torch.norm(tmp)**2
    G = 2 * P + 2 * r * At_operator(tmp, edgeind, dim) @ P
    return F, G


def create_mask(n, rate):
    """Create a symmetric mask matrix with given observation rate."""
    mask = (np.random.rand(n, n) < rate).astype(float)
    np.fill_diagonal(mask, 1)
    mask = np.tril(mask) + np.triu(mask.T, 1)
    return mask


def compute_ipm_from_P(P, center=True):
    """
    Compute Inner Product Matrix from factor matrix P.
    """
    


    if np.isnan(P).any() or np.isinf(P).any():
        print("Locations of NaN values:", np.argwhere(np.isnan(P)))
    
    IPM = P @ P.T
    if center:
        IPM = IPM - np.mean(IPM, axis=1, keepdims=True) - np.mean(IPM, axis=0) + np.mean(IPM)
        IPM = (IPM + IPM.T) / 2  # Ensure symmetry
    return IPM


def compute_distances_from_ipm(IPM):
    """
    Compute distance matrix from Inner Product Matrix using standard formula.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    diag = np.diag(IPM)
    return diag[:, None] + diag[None, :] - 2 * IPM


def compute_ground_truth_ipm(Dist):
    """
    Compute ground truth IPM using double-centering formula.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    Dist = np.asarray(Dist, dtype=float)
    row_mean = np.mean(Dist, axis=1, keepdims=True)
    col_mean = np.mean(Dist, axis=0, keepdims=True)
    total_mean = np.mean(Dist)
    return -0.5 * (Dist - row_mean - col_mean + total_mean)


def P_from_IPM(IPM, rank):
    """
    Extract factor matrix P from Inner Product Matrix using eigendecomposition.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    eigenvalues, eigenvectors = np.linalg.eigh(IPM)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top 'rank' components and form P matrix
    top_eigenvalues = np.maximum(eigenvalues[:rank], 0)  # Ensure non-negative
    P = eigenvectors[:, :rank] @ np.diag(np.sqrt(top_eigenvalues))
    
    return P
