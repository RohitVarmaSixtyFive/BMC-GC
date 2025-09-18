import os
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal, gamma as np_gamma
from scipy.stats import wishart
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing
import matplotlib.pyplot as plt

# --- Local modules (assumed present) ---
from optimization import alternating_completion
from evaluation import compute_all_errors
from math_utils import recon_from_points
import warnings 
from bpmc import BayesianEDMCompletion


def ensure_dir_for_file(path: Path):
    """Ensure parent directory exists for a given file path."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: Path):
    """Ensure directory exists."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def write_text_file(path, text):
    """Write text to file, ensuring parent directory exists."""
    p = Path(path)
    ensure_dir_for_file(p)
    p.write_text(text, encoding="utf-8")

def write_dataframe_csv(path, df, mode="w"):
    """
    Save DataFrame to CSV.
    mode = "w" (overwrite) or "a" (append). If append and file doesn't exist, header is written.
    """
    p = Path(path)
    ensure_dir_for_file(p)
    header = True if (mode == "w" or not p.exists()) else False
    df.to_csv(p, index=False, mode=mode, header=header)

def append_row_to_csv(path, row_dict):
    """Append a single-row dict to CSV, creating file with header if needed."""
    p = Path(path)
    ensure_dir_for_file(p)
    write_header = not p.exists()
    df = pd.DataFrame([row_dict])
    df.to_csv(p, index=False, mode="a", header=write_header)

def create_mask(n, rate, seed=None):
    """Create a symmetric boolean mask with diagonal True. 'rate' is fraction observed per entry (probability)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    base = rng.rand(n, n)
    mask = (base < rate)
    mask = np.triu(mask, 1)  # upper triangle random
    mask = mask | mask.T     # symmetrize
    np.fill_diagonal(mask, True)
    return mask.astype(bool)

def generate_ground_truth_edm(n_points, n_dimensions, seed=None):

    rng = np.random.RandomState(seed)
    pts = rng.rand(n_points, n_dimensions)
    dot_pt = np.sum(pts**2, axis=1)
    D_true = dot_pt[:, None] + dot_pt[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(D_true, 0.0)
    return D_true

def _log_posterior_P_i_vectorized(P_i, i, P_all, D_obs, neighbors_i, mu_p, Delta_p, alpha):
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
    d = mu_0.shape[0]
    if nu < d:
        nu = d
    # scipy's wishart.rvs doesn't accept RNG, so reproducibility only by setting global np.random.seed before call
    Delta_p = wishart(df=nu, scale=W).rvs()
    cov_mu = np.linalg.inv(beta * Delta_p)
    mu_p = multivariate_normal(mu_0, cov_mu)
    return mu_p, Delta_p

def BayesianEDMCompletion(D_obs, Mask, T, rank, output_file=None,
                          P_initial=None, init_alpha=1.0, proposal_scale=0.05,
                          mu_0=None, Beta_0=None, W_0=None, nu_0=None,
                          alpha_a0=1e-6, alpha_b0=1e-6, burn_in=1000,
                          save_file=False, seed=None, quiet=False):
    """
    Bayesian EDM completion (Gibbs with per-point MH).
    Returns the posterior predictive distance matrix averaged over collected samples.
    """
    rng = np.random.RandomState(seed)
    N = D_obs.shape[0]
    d = rank

    Mask_bool = Mask.astype(bool)
    neighbors = [np.where(Mask_bool[i, :] & (np.arange(N) != i))[0] for i in range(N)]

    N_obs_pairs = np.sum(np.triu(Mask_bool, 1))
    if N_obs_pairs <= 0:
        raise ValueError("No observed off-diagonal entries in Mask.")

    if P_initial is not None and P_initial.shape == (N, d):
        if not quiet: print("Initializing P with warm-start.")
        P_old = P_initial.copy()
    else:
        if not quiet: print("Initializing P randomly.")
        P_old = rng.randn(N, d)

    if mu_0 is None: mu_0 = np.zeros(d)
    if nu_0 is None: nu_0 = float(d + 2)
    if Beta_0 is None: Beta_0 = 2.0
    if W_0 is None: W_0 = np.eye(d)
    W_0_inv = np.linalg.inv(W_0)
    alpha = float(init_alpha)

    D_predict = np.zeros_like(D_obs, dtype=float)
    total_iterations = burn_in + int(T)
    if not quiet: print(f"Bayesian MCMC: {total_iterations} iters ({burn_in} burn-in + {T} samples). N={N}, d={d}")
    proposal_cov = np.eye(d) * (proposal_scale ** 2)

    pbar = tqdm(range(total_iterations), desc="Bayesian EDM", ncols=120, disable=quiet)
    samples_collected = 0
    for t in pbar:
        P_bar = np.mean(P_old, axis=0)
        S = np.cov(P_old, rowvar=False, bias=True) * N
        beta_star = Beta_0 + N
        nu_star = nu_0 + N
        mu_star = (Beta_0 * mu_0 + N * P_bar) / beta_star
        diff_mu = P_bar - mu_0
        W_star_inv = W_0_inv + S + (Beta_0 * N / beta_star) * np.outer(diff_mu, diff_mu)
        W_star = np.linalg.inv(W_star_inv)
        # For reproducibility, set global seed if provided before wishart sampling
        if seed is not None:
            np.random.seed(seed + t)  # vary per-iteration
        mu_p, Delta_p = Normal_Wishart_sample(mu_star, beta_star, W_star, nu_star, seed=seed)

        P_new = P_old.copy()
        acceptance_count = 0
        for i in range(N):
            P_i_current = P_old[i, :].copy()
            P_i_candidate = rng.multivariate_normal(P_i_current, proposal_cov)
            log_post_current = _log_posterior_P_i_vectorized(P_i_current, i, P_old, D_obs, neighbors[i], mu_p, Delta_p, alpha)
            log_post_candidate = _log_posterior_P_i_vectorized(P_i_candidate, i, P_old, D_obs, neighbors[i], mu_p, Delta_p, alpha)
            if np.log(rng.rand()) < log_post_candidate - log_post_current:
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

        # Collect samples after burn-in (thinning every 10)
        if t >= burn_in and (t - burn_in) % 10 == 0:
            D_predict = (D_predict * samples_collected + D_recon) / (samples_collected + 1)
            samples_collected += 1

        recon_error_current_sample = np.linalg.norm((Mask_bool * (D_recon - D_obs)), 'fro') / max(np.linalg.norm(Mask_bool * D_obs, 'fro'), 1e-16)
        pbar.set_postfix({"ReconErr": f"{recon_error_current_sample:.3e}", "alpha": f"{alpha:.2e}", "acc": f"{acceptance_rate:.2f}"})

        # adaptive proposal during burn-in
        if (t + 1) % 10 == 0 and t < burn_in:
            if acceptance_rate < 0.2: proposal_scale *= 0.9
            elif acceptance_rate > 0.4: proposal_scale *= 1.1
            proposal_cov = np.eye(d) * (proposal_scale ** 2)

    return D_predict

def run_integrated_completion_single(D_true, mask, rank, alternating_opts, lsopts,
                                     bayesian_T=1000, init_alpha=1.0, burn_in=1000, seed=None, quiet=True):
    """
    Single-run pipeline returning (row_dict)
    This function is used by worker processes.
    """
    try:
        P_alt = alternating_completion(D_true, mask, alternating_opts, lsopts)
    except Exception as e:
        # If warm-start fails, fall back to random init but continue
        P_alt = None

    if P_alt is not None:
        D_alt_recon = recon_from_points(P_alt)
    else:
        # fallback: random
        rng = np.random.RandomState(seed)
        P_alt = rng.randn(D_true.shape[0], rank)
        D_alt_recon = recon_from_points(P_alt)

    denom = np.linalg.norm(D_true, 'fro')
    denom = denom if denom > 0 else 1.0
    alternating_error = np.linalg.norm((D_alt_recon - D_true), 'fro') / denom

    D_obs = D_true * mask.astype(float)
    D_pred_bayes = BayesianEDMCompletion(D_obs=D_obs, Mask=mask, T=bayesian_T,
                                         rank=rank, P_initial=P_alt,
                                         init_alpha=init_alpha, burn_in=burn_in,
                                         seed=seed, quiet=True)

    bayesian_error = np.linalg.norm((D_pred_bayes - D_true), 'fro') / denom

    row = {
        'timestamp': datetime.now().isoformat(),
        'n_points': int(D_true.shape[0]),
        'mask_ratio': float(np.mean(mask.astype(float)) - (1.0 / D_true.shape[0]) ),  # approximate mask ratio (excluding diag bias)
        'n_dimensions': int(rank),
        'alternating_error': float(alternating_error),
        'bayesian_error': float(bayesian_error),
        'seed': int(seed) if seed is not None else -1
    }
    return row

# -------------------------
# Worker wrappers for different experiments
# -------------------------
def _task_mask_ratio(args):
    """Worker for mask-ratio experiment. args is a tuple with (n_points, n_dimensions, mask_ratio, mc_iter, alternating_opts, lsopts, bayesian_T, burn_in, base_seed)"""
    (n_points, n_dimensions, mask_ratio, mc_iter, alternating_opts, lsopts, bayesian_T, burn_in, base_seed) = args
    seed = (base_seed or 0) + mc_iter
    D_true = generate_ground_truth_edm(n_points, n_dimensions, seed=seed)
    mask = create_mask(n_points, mask_ratio, seed=seed)
    # return row plus meta such as which mc_iter and mask_ratio
    row = run_integrated_completion_single(D_true, mask, n_dimensions, alternating_opts, lsopts,
                                          bayesian_T=bayesian_T, burn_in=burn_in, seed=seed, quiet=True)
    row['mask_ratio'] = float(mask_ratio)
    row['mc_iteration'] = int(mc_iter)
    row['n_points'] = int(n_points)
    return row

def _task_n_values(args):
    """Worker for n-values experiment. args is a tuple with (n_points, n_dimensions, mask_ratio, mc_iter, alternating_opts, lsopts, bayesian_T, burn_in, base_seed)"""
    (n_points, n_dimensions, mask_ratio, mc_iter, alternating_opts, lsopts, bayesian_T, burn_in, base_seed) = args
    seed = (base_seed or 0) + mc_iter
    D_true = generate_ground_truth_edm(n_points, n_dimensions, seed=seed)
    mask = create_mask(n_points, mask_ratio, seed=seed)
    row = run_integrated_completion_single(D_true, mask, n_dimensions, alternating_opts, lsopts,
                                          bayesian_T=bayesian_T, burn_in=burn_in, seed=seed, quiet=True)
    row['mask_ratio'] = float(mask_ratio)
    row['mc_iteration'] = int(mc_iter)
    row['n_points'] = int(n_points)
    return row

def run_experiments_on_mask_ratios(
    mask_ratios,
    n_points,
    n_dimensions,
    n_monte_carlo,
    alternating_opts,
    lsopts,
    bayesian_T=500,
    burn_in=1000,
    results_dir="noiseless_results/run1",
    per_iteration_csv_name="mask_ratio_results.csv",
    summary_txt_name="mask_ratio_summary.txt",
    n_workers=None,
    base_seed=0
    ):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    per_iter_csv = results_dir / per_iteration_csv_name
    # backup existing CSV (if any)
    if per_iter_csv.exists():
        backup = per_iter_csv.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        per_iter_csv.rename(backup)
        print(f"Existing results CSV moved to backup: {backup}")

    n_workers = n_workers or max(1, cpu_count() - 1)
    # Build task list
    tasks = []
    for ratio in mask_ratios:
        for i in range(n_monte_carlo):
            tasks.append((n_points, n_dimensions, ratio, i, alternating_opts, lsopts, bayesian_T, burn_in, base_seed))

    print(f"Starting Mask Ratio experiments with {len(tasks)} tasks on {n_workers} workers...")
    # Use imap_unordered to receive completed rows as they finish
    with Pool(processes=n_workers) as pool:
        for row in tqdm(pool.imap_unordered(_task_mask_ratio, tasks), total=len(tasks), desc="MaskRatio tasks"):
            # Append each completed row to CSV immediately (safe single-process writing)
            # Extend row with timestamp if missing
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().isoformat()
            append_row_to_csv(per_iter_csv, row)

    # After all runs, read the CSV to create summary
    results_df = pd.read_csv(per_iter_csv)
    summary = results_df.groupby('mask_ratio')[['alternating_error', 'bayesian_error']].agg(['mean', 'std'])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_txt_path = results_dir / f"{Path(summary_txt_name).stem}_{ts}.txt"
    write_text_file(summary_txt_path, f"Mask Ratio Experiment Summary ({ts})\n\n{summary.to_string()}\n")

    print(f"\nSaved per-iteration CSV -> {per_iter_csv}")
    print(f"Saved summary -> {summary_txt_path}")

    return results_df

def run_experiments_on_n(
    n_values,
    mask_ratio,
    n_dimensions,
    n_monte_carlo,
    alternating_opts,
    lsopts,
    bayesian_T=500,
    burn_in=1000,
    results_dir="noiseless_results/run1",
    per_iteration_csv_name="n_values_results.csv",
    summary_txt_name="n_values_summary.txt",
    n_workers=None,
    base_seed=0
    ):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    per_iter_csv = results_dir / per_iteration_csv_name
    if per_iter_csv.exists():
        backup = per_iter_csv.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        per_iter_csv.rename(backup)
        print(f"Existing results CSV moved to backup: {backup}")

    n_workers = n_workers or max(1, cpu_count() - 1)
    tasks = []
    for n in n_values:
        for i in range(n_monte_carlo):
            tasks.append((n, n_dimensions, mask_ratio, i, alternating_opts, lsopts, bayesian_T, burn_in, base_seed))

    print(f"Starting N-values experiments with {len(tasks)} tasks on {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        for row in tqdm(pool.imap_unordered(_task_n_values, tasks), total=len(tasks), desc="NValue tasks"):
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().isoformat()
            append_row_to_csv(per_iter_csv, row)

    results_df = pd.read_csv(per_iter_csv)
    summary = results_df.groupby('n_points')[['alternating_error', 'bayesian_error']].agg(['mean', 'std'])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_txt_path = results_dir / f"{Path(summary_txt_name).stem}_{ts}.txt"
    write_text_file(summary_txt_path, f"N Values Experiment Summary ({ts})\n\n{summary.to_string()}\n")

    print(f"\nSaved per-iteration CSV -> {per_iter_csv}")
    print(f"Saved summary -> {summary_txt_path}")

    return results_df

def plot_error_vs_mask_ratio(results_df, results_dir="noiseless_results/run1", filename=None):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = results_dir / f"error_vs_mask_ratio_{ts}.png"
    else:
        filename = Path(filename)

    grouped = results_df.groupby('mask_ratio').agg({
        'alternating_error': ['mean', 'std'],
        'bayesian_error': ['mean', 'std']
    }).sort_index()
    x = grouped.index.values.astype(float)

    a_mean = grouped[('alternating_error','mean')].values
    a_std  = grouped[('alternating_error','std')].values
    b_mean = grouped[('bayesian_error','mean')].values
    b_std  = grouped[('bayesian_error','std')].values

    plt.figure(figsize=(7,5))
    plt.errorbar(x, a_mean, marker='o', linestyle='-', linewidth=1.5, markersize=6, label='Alternating', capsize=3)
    plt.errorbar(x, b_mean, marker='s', linestyle='-', linewidth=1.5, markersize=6, label='Bayesian', capsize=3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mask ratio (fraction observed)')
    plt.ylabel('Relative Frobenius Error (mean ± std)')
    plt.title('Error vs Mask Ratio (log-log)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_error_vs_n_points(results_df, results_dir="noiseless_results/run1", filename=None):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = results_dir / f"error_vs_n_points_{ts}.png"
    else:
        filename = Path(filename)

    grouped = results_df.groupby('n_points').agg({
        'alternating_error': ['mean', 'std'],
        'bayesian_error': ['mean', 'std']
    }).sort_index()
    x = grouped.index.values.astype(float)

    a_mean = grouped[('alternating_error','mean')].values
    a_std  = grouped[('alternating_error','std')].values
    b_mean = grouped[('bayesian_error','mean')].values
    b_std  = grouped[('bayesian_error','std')].values

    plt.figure(figsize=(7,5))
    plt.errorbar(x, a_mean,  marker='o', linestyle='-', linewidth=1.5, markersize=6, label='Alternating', capsize=3)
    plt.errorbar(x, b_mean,  marker='s', linestyle='-', linewidth=1.5, markersize=6, label='Bayesian', capsize=3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of points (n)')
    plt.ylabel('Relative Frobenius Error (mean ± std)')
    plt.title('Error vs Number of Points (log-log)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

if __name__ == "__main__":

    N_DIMENSIONS = 3
    N_MONTE_CARLO = 20
    BAYESIAN_T = 500
    BURN_IN = 1000
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    opts = {'r': 1.0, 'rank': N_DIMENSIONS, 'maxit': 30, 'tol': 1e-5, 'printenergy': 0, 'printerror': 0}
    lsopts = {'maxit': 20, 'xtol': 1e-8, 'gtol': 1e-8, 'ftol': 1e-10, 'alpha': 1e-3,
              'rho': 1e-4, 'sigma': 0.1, 'eta': 0.8}

    results_dir = "paper_noiseless_results"
    ensure_dir(Path(results_dir))

    # How many workers? default: use all cores - 2 (leave 2 cores free)
    n_workers = max(1, cpu_count() - 2)
    print(f"Using {n_workers} worker processes (cpu_count={cpu_count()})")

    # Experiment 1: Mask ratios
    print("="*60)
    print("🚀 Running Experiment 1: Varying Mask Ratios")
    print("="*60)
    mask_ratio_results = run_experiments_on_mask_ratios(
        mask_ratios=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        n_points=500,
        n_dimensions=N_DIMENSIONS,
        n_monte_carlo=N_MONTE_CARLO,
        alternating_opts=opts, lsopts=lsopts,
        bayesian_T=BAYESIAN_T, burn_in=BURN_IN,
        results_dir=results_dir,
        n_workers=n_workers,
        base_seed=2025
    )

    # Experiment 2: Varying N
    print("\n" + "="*60)
    print("🚀 Running Experiment 2: Varying N (Number of Points)")
    print("="*60)
    n_values_results = run_experiments_on_n(
        n_values=[50, 100, 250, 500],
        mask_ratio=0.05,
        n_dimensions=N_DIMENSIONS,
        n_monte_carlo=N_MONTE_CARLO,
        alternating_opts=opts, lsopts=lsopts,
        bayesian_T=BAYESIAN_T, burn_in=BURN_IN,
        results_dir=results_dir,
        n_workers=n_workers,
        base_seed=2025
    )
    
    # Reload CSVs (consistent paths)
    mask_ratio_csv = Path(results_dir) / "mask_ratio_results.csv"
    n_values_csv = Path(results_dir) / "n_values_results.csv"

    if mask_ratio_csv.exists():
        mask_ratio_results = pd.read_csv(mask_ratio_csv)
        plot_error_vs_mask_ratio(mask_ratio_results, results_dir=results_dir)
    else:
        print(f"Warning: expected CSV not found: {mask_ratio_csv}")

    if n_values_csv.exists():
        n_values_results = pd.read_csv(n_values_csv)
        plot_error_vs_n_points(n_values_results, results_dir=results_dir)
    else:
        print(f"Warning: expected CSV not found: {n_values_csv}")

    print("\nAll done.")
