"""
Error analysis and evaluation metrics for matrix completion methods.
"""
import numpy as np
from math_utils import (
    recon_from_points, compute_ipm_from_P, compute_distances_from_ipm,
    compute_ground_truth_ipm, P_from_IPM
)


def compute_all_errors(P, Dist_original, Weight, verbose=True):
    """
    Compute all error metrics from the factor matrix P.
    
    Args:
        P: Factor matrix
        Dist_original: Original distance matrix
        Weight: Binary mask indicating observed entries
        verbose: Whether to print detailed error information
    
    Returns:
        dict: Dictionary containing various error metrics
    """
    # Convert inputs to numpy
    Dist_original = np.asarray(Dist_original, dtype=float)
    Weight = np.asarray(Weight, dtype=float)
    P = np.asarray(P, dtype=float)
    
    # Compute reconstructions
    IPM_Truth = compute_ground_truth_ipm(Dist_original)
    IPM_Recon = compute_ipm_from_P(P, center=True)
    
    # Distance matrices
    Dist_Truth_from_IPM = compute_distances_from_ipm(IPM_Truth)
    Dist_Recon_from_IPM = compute_distances_from_ipm(IPM_Recon)
    
    # Additional distance computations for comparison
    P_centered = P - np.mean(P, axis=0, keepdims=True)
    Dist_from_P_centered = recon_from_points(P_centered, squared=True)
    Dist_from_P_uncentered = recon_from_points(P, squared=True)
    
    # Error calculations
    eps = 1e-16
    
    # IPM errors
    ipm_recon_err = np.linalg.norm(IPM_Truth - IPM_Recon, 'fro') / (np.linalg.norm(IPM_Truth, 'fro') + eps)
    ipm_masked_err = np.linalg.norm(Weight * (IPM_Truth - IPM_Recon), 'fro') / (np.linalg.norm(Weight * IPM_Truth, 'fro') + eps)
    
    # Distance errors
    dist_err = np.linalg.norm(Dist_Truth_from_IPM - Dist_Recon_from_IPM, 'fro') / (np.linalg.norm(Dist_Truth_from_IPM, 'fro') + eps)
    dist_masked_err = np.linalg.norm(Weight * (Dist_Truth_from_IPM - Dist_Recon_from_IPM), 'fro') / (np.linalg.norm(Weight * Dist_Truth_from_IPM, 'fro') + eps)
    
    # Raw distance errors (vs original input)
    dist_raw_err = np.linalg.norm(Dist_original - Dist_Recon_from_IPM, 'fro') / (np.linalg.norm(Dist_original, 'fro') + eps)
    dist_raw_masked_err = np.linalg.norm(Weight * (Dist_original - Dist_Recon_from_IPM), 'fro') / (np.linalg.norm(Weight * Dist_original, 'fro') + eps)
    
    # Centering consistency checks
    diff_centered = Dist_Recon_from_IPM - Dist_from_P_centered
    center_diff_norm_rel = np.linalg.norm(diff_centered, 'fro') / (np.linalg.norm(Dist_Recon_from_IPM, 'fro') + eps)
    max_abs_diff = np.max(np.abs(diff_centered))
    
    diff_uncentered = Dist_Recon_from_IPM - Dist_from_P_uncentered
    center_diff_unc_norm_rel = np.linalg.norm(diff_uncentered, 'fro') / (np.linalg.norm(Dist_Recon_from_IPM, 'fro') + eps)
    max_abs_diff_unc = np.max(np.abs(diff_uncentered))
    
    # Success rate
    shape = Dist_original.shape
    success_rate = (np.sum(np.abs(IPM_Recon - IPM_Truth) < 1e-5)) / (shape[0] * shape[1])
    
    errors = {
        'IPM_ReconError': ipm_recon_err,
        'IPM_ReconError_Masked': ipm_masked_err,
        'Dist_ReconError': dist_err,
        'Dist_ReconError_Masked': dist_masked_err,
        'Dist_ReconError_Raw': dist_raw_err,
        'Dist_ReconError_Raw_Masked': dist_raw_masked_err,
        'centered_distance_diff_rel': center_diff_norm_rel,
        'centered_distance_max_abs_diff': max_abs_diff,
        'uncentered_distance_diff_rel': center_diff_unc_norm_rel,
        'uncentered_distance_max_abs_diff': max_abs_diff_unc,
        'success_rate_close_IPM': success_rate
    }
    
    if verbose:
        print("=== Error Analysis ===")
        print(f"IPM reconstruction error (rel, fro): {errors['IPM_ReconError']:.6e}")
        print(f"IPM reconstruction error (masked, rel): {errors['IPM_ReconError_Masked']:.6e}")
        print(f"Distance reconstruction error (from IPM, rel): {errors['Dist_ReconError']:.6e}")
        print(f"Distance reconstruction error (masked, rel): {errors['Dist_ReconError_Masked']:.6e}")
        print(f"Distance reconstruction vs original (rel): {errors['Dist_ReconError_Raw']:.6e}")
        print(f"Distance reconstruction vs original (masked, rel): {errors['Dist_ReconError_Raw_Masked']:.6e}")
        print(f"Centered-P vs IPM distance consistency (rel): {errors['centered_distance_diff_rel']:.3e}")
        print(f"Success rate (IPM elements < 1e-5 error): {errors['success_rate_close_IPM']:.3f}")
    
    return errors
