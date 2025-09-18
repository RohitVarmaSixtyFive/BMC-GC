#!/usr/bin/env python3
"""
Run Bayesian EDM completion in PARALLEL by loading pre-generated matrices.
This version suppresses runtime warnings for cleaner output, shows a
running average error in the progress bar, and is structured to run
targeted experiments.
"""
import os
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal, gamma as np_gamma
from scipy.stats import wishart
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing
from collections import defaultdict
import itertools
import warnings

# ---- Assume this function exists in your local implementation ----
def recon_from_points(points):
    """
    Dummy implementation: must return a squared-distance matrix from points.
    Your actual implementation should be used here.
    """
    n_points, n_dims = points.shape
    dot_pt = np.sum(points**2, axis=1)
    D_recon_sq = dot_pt[:, None] + dot_pt[None, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(D_recon_sq, 0.0)
    return D_recon_sq
# ----------------------------------------------------------------

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _log_posterior_P_i_vectorized(P_i, i, P_all, D_obs, neighbors_i, mu_p, Delta_p, alpha):
    diff_prior = P_i - mu_p
    log_prior = -0.5 * float(diff_prior.T @ Delta_p @ diff_prior)
    if neighbors_i.size == 0:
        return log_prior
    P_j = P_all[neighbors_i, :]
    diffs = P_i[None, :] - P_j
    dist_sq = np.sum(diffs**2, axis=1)
    r_ij = D_obs[i, neighbors_i] - dist_sq
    log_likelihood = -0.5 * alpha * np.sum(r_ij**2)
    return log_prior + log_likelihood

def Normal_Wishart_sample(mu_0, beta, W, nu, seed=None):
    d = mu_0.shape[0]
    if nu < d:
        nu = d
    try:
        Delta_p = wishart(df=nu, scale=W).rvs()
        cov_mu = np.linalg.inv(beta * Delta_p)
        mu_p = multivariate_normal(mu_0, cov_mu)
    except np.linalg.LinAlgError:
        return np.linalg.inv(W), mu_0
    return mu_p, Delta_p

def BayesianEDMCompletion(D_obs, Mask, T, rank,
                          P_initial=None, init_alpha=1.0, proposal_scale=0.05,
                          mu_0=None, Beta_0=None, W_0=None, nu_0=None,
                          alpha_a0=1e-6, alpha_b0=1e-6, burn_in=1000,
                          seed=None, verbose=False):
    rng = np.random.RandomState(seed)
    N = D_obs.shape[0]
    d = rank

    Mask_bool = Mask.astype(bool)
    neighbors = [np.where(Mask_bool[i, :] & (np.arange(N) != i))[0] for i in range(N)]
    N_obs_pairs = np.sum(np.triu(Mask_bool, 1))
    if N_obs_pairs <= 0:
        return np.zeros_like(D_obs), pd.DataFrame()

    if P_initial is not None and P_initial.shape == (N, d):
        P_old = P_initial.copy()
    else:
        P_old = rng.randn(N, d)

    if mu_0 is None: mu_0 = np.zeros(d)
    if nu_0 is None: nu_0 = float(d + 2)
    if Beta_0 is None: Beta_0 = 2.0
    if W_0 is None: W_0 = np.eye(d)
    
    try:
        W_0_inv = np.linalg.inv(W_0)
    except np.linalg.LinAlgError:
        W_0 = W_0 + 1e-6 * np.eye(d)
        W_0_inv = np.linalg.inv(W_0)

    alpha = float(init_alpha)
    D_predict = np.zeros_like(D_obs, dtype=float)
    total_iterations = burn_in + int(T)
    proposal_cov = np.eye(d) * (proposal_scale ** 2)
    iterator = range(total_iterations)
    
    # <<< CHANGE 1: Initialize a counter for thinned samples >>>
    samples_collected = 0

    for t in iterator:
        P_bar = np.mean(P_old, axis=0)
        S = np.cov(P_old, rowvar=False, bias=True) * N
        beta_star = Beta_0 + N
        nu_star = nu_0 + N
        mu_star = (Beta_0 * mu_0 + N * P_bar) / beta_star
        diff_mu = P_bar - mu_0
        W_star_inv = W_0_inv + S + (Beta_0 * N / beta_star) * np.outer(diff_mu, diff_mu)
        
        try:
            W_star = np.linalg.inv(W_star_inv)
        except np.linalg.LinAlgError:
            W_star_inv += 1e-6 * np.eye(d)
            W_star = np.linalg.inv(W_star_inv)

        mu_p, Delta_p = Normal_Wishart_sample(mu_star, beta_star, W_star, nu_star, seed=seed)
        P_new = P_old.copy()
        acceptance_count = 0

        for i in range(N):
            P_i_current = P_old[i, :].copy()
            P_i_candidate = rng.multivariate_normal(P_i_current, proposal_cov)
            log_post_current = _log_posterior_P_i_vectorized(P_i_current, i, P_old, D_obs, neighbors[i], mu_p, Delta_p, alpha)
            log_post_candidate = _log_posterior_P_i_vectorized(P_i_candidate, i, P_old, D_obs, neighbors[i], mu_p, Delta_p, alpha)
            log_acceptance_ratio = log_post_candidate - log_post_current
            if np.log(rng.rand()) < log_acceptance_ratio:
                P_new[i, :] = P_i_candidate
                acceptance_count += 1

        P_old = P_new
        acceptance_rate = acceptance_count / N
        D_recon = recon_from_points(P_old)
        diff_sq = (D_obs - D_recon) ** 2
        sse_pairs = np.sum(np.triu(Mask_bool.astype(float) * diff_sq, 1))
        a_N = alpha_a0 + 0.5 * N_obs_pairs
        b_N = alpha_b0 + 0.5 * sse_pairs
        alpha = np_gamma(shape=a_N, scale=1.0 / float(b_N))

        # <<< CHANGE 2: Modified logic to average every 10th sample >>>
        # After burn-in, check if the current sample is the 10th, 20th, etc.
        if t >= burn_in and (t - burn_in) % 10 == 0:
            # Update the running average using the new sample and the count of collected samples
            D_predict = (D_predict * samples_collected + D_recon) / (samples_collected + 1)
            samples_collected += 1
        
        if (t + 1) % 10 == 0 and t < burn_in:
            if acceptance_rate < 0.2:
                proposal_scale *= 0.9
            elif acceptance_rate > 0.4:
                proposal_scale *= 1.1
            proposal_cov = np.eye(d) * (proposal_scale ** 2)
            
    return D_predict, pd.DataFrame()

def run_bayesian_only(D_true, D_noisy, mask, rank, bayesian_T=500, burn_in=500, seed=None, verbose=False):
    D_obs = D_noisy * mask.astype(float)
    D_pred_bayes, _ = BayesianEDMCompletion(D_obs=D_obs, Mask=mask, T=bayesian_T, rank=rank, burn_in=burn_in, seed=seed, verbose=verbose)
    denom = np.linalg.norm(D_true, 'fro')
    final_error = np.linalg.norm((D_pred_bayes - D_true), 'fro') / (denom if denom > 0 else 1.0)
    return final_error

# --- Worker Function for Parallel Processing ---
def run_single_trial(params):
    mr, n, snr, t, n_dims, bayesian_T, burn_in, seed_base, base_dir = params
    config_path = Path(base_dir) / f"n_{n}" / f"mask_{mr:.2f}"
    filename = f"n{n}_mask{mr:.2f}_snr{snr}_trial{t}.npz"
    filepath = config_path / filename

    try:
        data = np.load(filepath)
        D_true = data['D_true']
        D_noisy = data['D_noisy']
        mask = data['mask']
    except FileNotFoundError:
        return None

    seed = seed_base + t
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        error = run_bayesian_only(D_true, D_noisy, mask,
                                  rank=n_dims,
                                  bayesian_T=bayesian_T,
                                  burn_in=burn_in,
                                  seed=seed,
                                  verbose=False)
    
    return (mr, n, snr, error)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def results_to_dataframe(results_by_mask):
    rows = []
    for mr, result_dict in results_by_mask.items():
        for n, (snr_list, means, stds) in result_dict.items():
            for snr, mean, std in zip(snr_list, means, stds):
                rows.append({
                    "mask_ratio": mr,
                    "n": n,
                    "snr_db": snr,
                    "mean_error": mean,
                    "std_error": std,
                })
    return pd.DataFrame(rows)

def plot_results_across_masks(results_by_mask, exp_name: str, y_log=True):
    for mr, result_dict in results_by_mask.items():
        plt.figure(figsize=(8, 5))
        for n, (snr_list, means, stds) in sorted(result_dict.items()):
            plt.errorbar(snr_list, means, yerr=stds, marker="o", label=f"n={n}", capsize=3)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Relative Reconstruction Error (Frobenius)")
        plt.title(f"Error vs SNR — Bayesian EDM (mask_ratio={mr})")
        if y_log:
            plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        out_path = RESULTS_DIR / f"{exp_name}_mask{mr}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

def plot_results_by_mask_ratio(results_by_mask, n_fixed: int, exp_name: str, y_log=True):
    plt.figure(figsize=(8, 6))
    for mr, result_dict in sorted(results_by_mask.items()):
        if n_fixed in result_dict:
            snr_list, means, stds = result_dict[n_fixed]
            plt.errorbar(snr_list, means, yerr=stds, marker="o", label=f"mask_ratio={mr}", capsize=3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Relative Reconstruction Error (Frobenius)")
    plt.title(f"Error vs. SNR for n={n_fixed}")
    if y_log:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{exp_name}_n{n_fixed}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_experiment(n_list, snr_db_list, mask_ratio_list, n_trials, common_params):
    """ A generic function to run any experimental setup. """
    tasks = []
    param_combinations = itertools.product(mask_ratio_list, n_list, snr_db_list, range(n_trials))
    for mr, n, snr, t in param_combinations:
        tasks.append((mr, n, snr, t) + common_params)

    print(f"Created {len(tasks)} tasks to run in parallel...")

    results_flat = []
    error_sum = 0.0
    completed_count = 0
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(tasks), desc="Running MCMC Trials") as pbar:
            for result in pool.imap_unordered(run_single_trial, tasks):
                if result is not None:
                    results_flat.append(result)
                    mr, n, snr, error = result
                    completed_count += 1
                    error_sum += error
                    running_avg_error = error_sum / completed_count
                    pbar.set_postfix(avg_error=f'{running_avg_error:.4f}')
                pbar.update(1)

    trial_errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in results_flat:
        if result is not None:
            mr, n, snr, error = result
            trial_errors[mr][n][snr].append(error)

    results_by_mask = defaultdict(dict)
    for mr, n_dict in trial_errors.items():
        for n, snr_dict in n_dict.items():
            snrs = sorted(snr_dict.keys())
            mean_errors = [np.mean(snr_dict[snr]) for snr in snrs]
            std_errors = [np.std(snr_dict[snr]) for snr in snrs]
            results_by_mask[mr][n] = (snrs, np.array(mean_errors), np.array(std_errors))
            
    return results_by_mask

    
RESULTS_DIR = Path("noise_results") / "run1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Common Parameters
    COMMON_PARAMS = (
        3,    # n_dimensions
        300,  # bayesian_T
        1200, # burn_in
        0,    # seed_base
        Path("..") / "generated_data"  # DATA_DIRECTORY
    )

    # # --- Experiment 1 ---
    # print("--- Running Experiment 1: Scaling with Problem Size (n) ---")
    # n_list_exp1 = [100, 250, 500, 1000]
    # snr_db_list_exp1 = [5, 10, 15, 20, 25, 30]
    # mask_ratio_list_exp1 = [0.5]
    # n_trials_exp1 = 20

    # results_exp1 = run_experiment(n_list_exp1, snr_db_list_exp1, mask_ratio_list_exp1, n_trials_exp1, COMMON_PARAMS)
    # df1 = results_to_dataframe(results_exp1)
    # csv1 = RESULTS_DIR / "exp1_scaling_n_robustness.csv"
    # df1.to_csv(csv1, index=False)
    # print(f"Results saved: {csv1}")
    # plot_results_across_masks(results_exp1, exp_name="exp1_scaling_n_robustness")

    # # --- Experiment 2 ---
    # print("\n--- Running Experiment 2: Robustness to Sparsity ---")
    # n_list_exp2 = [250]
    # snr_db_list_exp2 = [20]
    # mask_ratio_list_exp2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # n_trials_exp2 = 20
    
    # results_exp2 = run_experiment(n_list_exp2, snr_db_list_exp2, mask_ratio_list_exp2, n_trials_exp2, COMMON_PARAMS)
    # df2 = results_to_dataframe(results_exp2)
    # csv2 = RESULTS_DIR / "exp2_sparsity_robustness.csv"
    # df2.to_csv(csv2, index=False)
    # print(f"Results saved: {csv2}")
    # plot_results_by_mask_ratio(results_exp2, n_fixed=250, exp_name="exp2_sparsity_robustness")
    
    # --- Experiment 3 ---
    print("\n--- Running Experiment 3: Robustness to Sparsity ---")
    n_list_exp2 = [250]
    snr_db_list_exp2 = [20]
    mask_ratio_list_exp2 = [0.01, 0.05]
    n_trials_exp2 = 20

    results_exp2 = run_experiment(n_list_exp2, snr_db_list_exp2, mask_ratio_list_exp2, n_trials_exp2, COMMON_PARAMS)
    df2 = results_to_dataframe(results_exp2)
    csv2 = RESULTS_DIR / "exp2_sparsity_robustness_temp.csv"
    df2.to_csv(csv2, index=False)
    print(f"Results saved: {csv2}")
    plot_results_by_mask_ratio(results_exp2, n_fixed=250, exp_name="exp2_sparsity_robustness_temp")
