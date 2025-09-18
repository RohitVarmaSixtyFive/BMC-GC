import os
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal, gamma as np_gamma
from scipy.stats import wishart
from optimization import alternating_completion
from evaluation import compute_all_errors
from math_utils import recon_from_points
from tqdm import tqdm

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def create_mask(n, rate, seed=None):

    rng = np.random.RandomState(seed) if seed is not None else np.random
    base = rng.rand(n, n)
    mask = (base < rate)
    mask = np.triu(mask, 1)  # upper triangle random
    mask = mask | mask.T     # symmetrize
    np.fill_diagonal(mask, True)
    return mask.astype(bool)

# -------------------------
# Bayesian sampler
# -------------------------
def _log_posterior_P_i_vectorized(P_i, i, P_all, D_obs, neighbors_i, mu_p, Delta_p, alpha):
    """
    Vectorized log-posterior for a single point P_i.
    neighbors_i: 1D array of indices j != i where D_ij is observed (mask True).
    """
    # prior: -1/2 (P_i - mu_p)^T Delta_p (P_i - mu_p)
    diff_prior = P_i - mu_p
    log_prior = -0.5 * float(diff_prior.T @ Delta_p @ diff_prior)

    if neighbors_i.size == 0:
        return log_prior

    P_j = P_all[neighbors_i, :]               # (m, d)
    diffs = P_i[None, :] - P_j                # (m, d)
    dist_sq = np.sum(diffs**2, axis=1)        # (m,)
    r_ij = D_obs[i, neighbors_i] - dist_sq
    log_likelihood = -0.5 * alpha * np.sum(r_ij**2)

    return log_prior + log_likelihood

def Normal_Wishart_sample(mu_0, beta, W, nu, seed=None):
    """
    Sample (mu_p, Delta_p) from Normal-Wishart prior/posterior.
    """
    d = mu_0.shape[0]
    if nu < d:
        nu = d
    if seed is None:
        Delta_p = wishart(df=nu, scale=W).rvs()
    else:
        Delta_p = wishart(df=nu, scale=W).rvs()
    cov_mu = np.linalg.inv(beta * Delta_p)
    mu_p = multivariate_normal(mu_0, cov_mu)
    return mu_p, Delta_p

def BayesianEDMCompletion(D_obs, Mask, T, rank, output_file=None,
                          P_initial=None, init_alpha=1.0, proposal_scale=0.05,
                          mu_0=None, Beta_0=None, W_0=None, nu_0=None,
                          alpha_a0=1e-6, alpha_b0=1e-6, burn_in=1000,
                          save_file=False, seed=None):
    """
    Bayesian EDM completion (Gibbs with per-point MH).
    """
    rng = np.random.RandomState(seed)
    N = D_obs.shape[0]
    d = rank

    # Preprocess mask as boolean and neighbors
    Mask_bool = Mask.astype(bool)
    neighbors = [np.where(Mask_bool[i, :] & (np.arange(N) != i))[0] for i in range(N)]

    # Number of observed unordered off-diagonal pairs:
    N_obs_pairs = np.sum(np.triu(Mask_bool, 1))
    if N_obs_pairs <= 0:
        raise ValueError("No observed off-diagonal entries in Mask.")

    # Initialize P
    if P_initial is not None and P_initial.shape == (N, d):
        print("Initializing P with warm-start.")
        P_old = P_initial.copy()
    else:
        print("Initializing P randomly.")
        P_old = rng.randn(N, d)

    # Hyperpriors defaults
    if mu_0 is None: mu_0 = np.zeros(d)
    if nu_0 is None: nu_0 = float(d + 2)   
    if Beta_0 is None: Beta_0 = 2.0
    if W_0 is None: W_0 = np.eye(d)
    W_0_inv = np.linalg.inv(W_0)

    alpha = float(init_alpha)

    # bookkeeping
    results_list = []
    D_predict = np.zeros_like(D_obs, dtype=float)
    total_iterations = burn_in + int(T)

    print(f"Bayesian MCMC: {total_iterations} iters ({burn_in} burn-in + {T} samples). N={N}, d={d}")

    proposal_cov = np.eye(d) * (proposal_scale ** 2)

    with tqdm(total=total_iterations, desc="Bayesian EDM", ncols=120) as pbar:

        for t in range(total_iterations):
            
# Sample Normal-Wishart hyperparams given P
            P_bar = np.mean(P_old, axis=0)
            # S = sum_i (P_i - mean)(P_i - mean)^T  (sum-of-squares matrix)
            S = np.cov(P_old, rowvar=False, bias=True) * N

            beta_star = Beta_0 + N
            nu_star = nu_0 + N
            mu_star = (Beta_0 * mu_0 + N * P_bar) / beta_star

            diff_mu = P_bar - mu_0
            W_star_inv = W_0_inv + S + (Beta_0 * N / beta_star) * np.outer(diff_mu, diff_mu)
            W_star = np.linalg.inv(W_star_inv)

            mu_p, Delta_p = Normal_Wishart_sample(mu_star, beta_star, W_star, nu_star, seed=seed)
            
# Sample P_i using Metropolis-Hastings (vectorized posterior eval)
            P_new = P_old.copy()
            acceptance_count = 0

            for i in range(N):
                P_i_current = P_old[i, :].copy()
                P_i_candidate = rng.multivariate_normal(P_i_current, proposal_cov)

                log_post_current = _log_posterior_P_i_vectorized(P_i_current, i, P_old,
                                                                D_obs, neighbors[i], mu_p, Delta_p, alpha)
                log_post_candidate = _log_posterior_P_i_vectorized(P_i_candidate, i, P_old,
                                                                D_obs, neighbors[i], mu_p, Delta_p, alpha)

                log_acceptance_ratio = log_post_candidate - log_post_current
                if np.log(rng.rand()) < log_acceptance_ratio:
                    P_new[i, :] = P_i_candidate
                    acceptance_count += 1

            P_old = P_new
            acceptance_rate = acceptance_count / N

# Sample precision alpha (treat unordered pairs once)
            D_recon = recon_from_points(P_old)   # should produce squared distances matrix
            diff_sq = (D_obs - D_recon) ** 2
            # sum over i<j only (observed)
            sse_pairs = np.sum(np.triu(Mask_bool.astype(float) * diff_sq, 1))  # sum_{i<j} (diff^2)
            a_N = alpha_a0 + 0.5 * N_obs_pairs
            b_N = alpha_b0 + 0.5 * sse_pairs  # rate
            # numpy's gamma uses shape, scale (=1/rate)
            alpha = np_gamma(shape=a_N, scale=1.0 / float(b_N))

# Collect & average predictions after burn-in
            recon_error_current_sample = np.linalg.norm((Mask_bool * (D_recon - D_obs)), 'fro') / max(np.linalg.norm(Mask_bool * D_obs, 'fro'), 1e-16)


            if t >= burn_in:
                sample_index = t - burn_in
                D_predict = (D_predict * sample_index + D_recon) / (sample_index + 1)

            # update tqdm progress bar
            pbar.set_postfix({
                "ReconErr": f"{recon_error_current_sample:.3e}",
                "alpha": f"{alpha:.2e}",
                "acc": f"{acceptance_rate:.2f}"
            })
            pbar.update(1)
            
            results_list.append({
                'step': t + 1,
                'recon_error': recon_error_current_sample,
                'alpha': alpha,
                'acceptance_rate': acceptance_rate
            })

            # adaptive step-size tuning in burn-in
            if (t + 1) % 10 == 0 and t < burn_in:
                if acceptance_rate < 0.2:
                    proposal_scale *= 0.9
                elif acceptance_rate > 0.4:
                    proposal_scale *= 1.1
                proposal_cov = np.eye(d) * (proposal_scale ** 2)

        results_df = pd.DataFrame(results_list)

    return D_predict, results_df

# -------------------------
# High-level pipeline
# -------------------------
def run_integrated_completion(D_true, mask, rank, alternating_opts, lsopts,
                              bayesian_T=1000,
                              init_alpha=1.0, burn_in=1000, save_file=True, seed=None):
    """
    Alternating completion (warm start) followed by Bayesian completion.
    Returns P_alt (from alternating), D_pred_bayes (averaged posterior recon), alt_errors, bayes_results_df.
    """
    print("=== Alternating Completion (warm start) ===")
    P_alt = alternating_completion(D_true, mask, alternating_opts, lsopts)

    print("\n--- Alternating Completion Results ---")
    alt_errors = compute_all_errors(P_alt, D_true, mask, verbose=True)

    D_obs = D_true * mask.astype(float)
    D_pred_bayes, results_df = BayesianEDMCompletion(D_obs=D_obs,
                                                    Mask=mask,
                                                    T=bayesian_T,
                                                    rank=rank,
                                                    P_initial=P_alt,
                                                    init_alpha=init_alpha,
                                                    burn_in=burn_in,
                                                    save_file=save_file,
                                                    seed=seed)

    # Final evaluation on unobserved entries
    unobserved_mask = (~mask).astype(float)
    np.fill_diagonal(unobserved_mask, 0.0)
    denom = np.linalg.norm(unobserved_mask * D_true, 'fro')
    final_error = np.linalg.norm((D_pred_bayes - D_true) * unobserved_mask, 'fro') / (denom if denom > 0 else 1.0)

    print(f"\n=== Final Results ===")
    print(f"Alternating Completion Error (masked): {alt_errors.get('Dist_ReconError_Raw', np.nan):.6e}")
    print(f"Bayesian Method Error (unobserved): {final_error:.6e}")

    return P_alt, D_pred_bayes, alt_errors, results_df



# -------------------------
# Example main (customize mask_ratios and alpha_values)
# -------------------------
if __name__ == "__main__":
    # Problem size & ground truth (example)
    n_points = 1000
    n_dimensions = 3
    rng = np.random.RandomState(0)
    pts = rng.rand(n_points, n_dimensions)
    dot_pt = np.sum(pts**2, axis=1)
    D_true = dot_pt[:, None] + dot_pt[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(D_true, 0.0)

    mask_ratios = [0.03]       
    alpha_values = [1]      

    opts = {
        'r': 1.0,
        'rank': n_dimensions,
        'maxit': 30,
        'tol': 1e-5,
        'printenergy': 0,
        'printerror': 0
    }
    lsopts = {
        'maxit': 20,
        'xtol': 1e-8,
        'gtol': 1e-8,
        'ftol': 1e-10,
        'alpha': 1e-3,
        'rho': 1e-4,
        'sigma': 0.1,
        'eta': 0.8,
    }


    for mask_ratio in mask_ratios:
        mask = create_mask(n_points, mask_ratio, seed=42)
        #verify mask percentage
        actual_ratio = np.sum(mask) / (n_points * n_points)
        print(f"\nMask ratio: {actual_ratio:.4f} (target was {mask_ratio})")
        for alpha0 in alpha_values:
            print(f"\n--- mask_ratio={mask_ratio}, init_alpha={alpha0} ---")
            P_alt, D_pred_bayes, alt_errors, results_df = run_integrated_completion(
                D_true=D_true,
                mask=mask,
                rank=n_dimensions,
                alternating_opts=opts,
                lsopts=lsopts,
                bayesian_T=500,
                init_alpha=alpha0,
                burn_in=1000,   # reduce burn-in for example runs; increase in real experiments
                save_file=True,
                seed=0
            )

            total_error = np.linalg.norm(D_pred_bayes - D_true, 'fro') / max(np.linalg.norm(D_true, 'fro'), 1e-16)
            print(f"Total Reconstruction Error (full matrix): {total_error:.6e}")