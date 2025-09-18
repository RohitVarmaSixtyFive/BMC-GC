import numpy as np
from pathlib import Path

# --- Helper Functions (placeholders as provided) ---

def create_mask(n, rate, seed=None):

    rng = np.random.RandomState(seed) if seed is not None else np.random
    base = rng.rand(n, n)
    mask = (base < rate)
    mask = np.triu(mask, 1)  # upper triangle random
    mask = mask | mask.T     # symmetrize
    np.fill_diagonal(mask, True)
    return mask.astype(bool)

def add_noise_to_D_snr(D_true, snr_db, seed=None):

    if snr_db is None or np.isinf(snr_db):
        return D_true.copy()
    rng = np.random.RandomState(seed)

    # use off-diagonal entries only to compute power (diagonal is zero)
    N = D_true.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)
    signal_vals = D_true[mask_offdiag]
    # use mean square as signal power
    signal_power = np.mean(signal_vals ** 2)
    if signal_power <= 0:
        # fallback: small epsilon
        signal_power = 1e-16

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_sigma = np.sqrt(noise_power)

    # generate symmetric gaussian noise with zero diagonal
    # create upper triangular noise then mirror
    upper = rng.randn(N, N)
    # zero lower triangle and diagonal
    upper = np.triu(upper, k=1)
    noise = upper + upper.T
    noise *= noise_sigma
    # print(f"Noise sigma is {noise_sigma:.6e} to achieve SNR {snr_db} dB (signal power {signal_power:.6e})")
    D_noisy = D_true + noise
    # enforce symmetry
    D_noisy = 0.5 * (D_noisy + D_noisy.T)
    # diagonal zero and non-negative
    np.fill_diagonal(D_noisy, 0.0)
    D_noisy = np.clip(D_noisy, a_min=0.0, a_max=None)
    return D_noisy

def generate_and_save_data(n_list, mask_ratio_list, snr_list, n_trials,
                           n_dimensions=3, seed_base=0, base_dir="data"):
    """
    Generates and saves matrices, masks, and noisy data for experiments.
    
    For each trial, it saves a single .npz file containing:
    - D_true: The ground truth distance matrix.
    - D_noisy: The noisy distance matrix.
    - mask: The observation mask.
    """
    print("--- Starting Data Generation ---")
    base_path = Path(base_dir)

    for n in n_list:
        for snr in snr_list:
            for mask_ratio in mask_ratio_list:
                # Create the specific directory for this configuration
                output_dir = base_path / f"n_{n}" / f"mask_{mask_ratio:.2f}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for t in range(n_trials):
                    print(f"Generating: n={n}, mask={mask_ratio:.2f}, trial={t+1}")
                    
                    # Consistent seeding for reproducibility
                    seed = seed_base + t
                    rng = np.random.RandomState(seed)

                    # Generate ground truth distance matrix
                    pts = rng.rand(n, n_dimensions)
                    dot_pt = np.sum(pts**2, axis=1)
                    D_true = dot_pt[:, None] + dot_pt[None, :] - 2.0 * (pts @ pts.T)
                    np.fill_diagonal(D_true, 0.0)
                    
                    # Create mask and noisy matrix
                    mask = create_mask(n, mask_ratio, seed=seed + 1000)

                    D_noisy = add_noise_to_D_snr(D_true, snr_db=snr, seed=seed + 2000)

                    # Define the filename and path for saving
                    filename = f"n{n}_mask{mask_ratio:.2f}_snr{snr}_trial{t}.npz"
                    print(filename)
                    filepath = output_dir / filename
                    
                    # Use np.savez_compressed for efficient storage
                    np.savez_compressed(
                        filepath,
                        D_true=D_true,
                        D_noisy=D_noisy,
                        mask=mask
                    )

    print(f"\n--- Data Generation Complete. Files saved in: {base_path.resolve()} ---")

def generate_and_save_clean_data(n_list, mask_ratio_list, n_trials,
                           n_dimensions=3, seed_base=0, base_dir="data"):
    """
    Generates and saves matrices, masks, and noisy data for experiments.
    
    For each trial, it saves a single .npz file containing:
    - D_true: The ground truth distance matrix.
    - mask: The observation mask.
    """
    base_path = Path(base_dir)

    for n in n_list:
        for mask_ratio in mask_ratio_list:
            # Create the specific directory for this configuration
            output_dir = base_path / f"n_{n}" / f"mask_{mask_ratio:.2f}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for t in range(n_trials):
                print(f"Generating: n={n}, mask={mask_ratio:.2f}, trial={t+1}")
                
                # Consistent seeding for reproducibility
                seed = seed_base + t
                rng = np.random.RandomState(seed)

                # Generate ground truth distance matrix
                pts = rng.rand(n, n_dimensions)
                dot_pt = np.sum(pts**2, axis=1)
                D_true = dot_pt[:, None] + dot_pt[None, :] - 2.0 * (pts @ pts.T)
                np.fill_diagonal(D_true, 0.0)
                
                # Create mask and noisy matrix
                mask = create_mask(n, mask_ratio, seed=seed + 1000)

                # Define the filename and path for saving
                filename = f"n{n}_mask{mask_ratio:.2f}_trial{t}.npz"
                filepath = output_dir / filename
                
                np.savez_compressed(
                    filepath,
                    D_true=D_true,
                    mask=mask
                )

    print(f"\n--- Data Generation Complete. Files saved in: {base_path.resolve()} ---")


if __name__ == '__main__':

    N_VALUES = [500]    
    MASK_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    NUM_TRIALS = 20
    
    OUTPUT_DIRECTORY = Path("..") / "generated_data_clean"

    generate_and_save_clean_data(
        n_list=N_VALUES,
        mask_ratio_list=MASK_RATIOS,
        n_trials=NUM_TRIALS,
        n_dimensions=3,
        seed_base=2025,
        base_dir=OUTPUT_DIRECTORY
    )
    
    N_VALUES = [50, 100, 250, 500]    
    MASK_RATIOS = [0.05, 0.1]
    NUM_TRIALS = 20
    

    OUTPUT_DIRECTORY = Path("..") / "generated_data_clean"

    # 3. Call the function to generate and save the data
    generate_and_save_clean_data(
        n_list=N_VALUES,
        mask_ratio_list=MASK_RATIOS,
        n_trials=NUM_TRIALS,
        n_dimensions=3,
        seed_base=2025,
        base_dir=OUTPUT_DIRECTORY
    )
    
    N_VALUES = [500]
    MASK_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    SNR_VALUES = [20,30]
    NUM_TRIALS = 20
    

    OUTPUT_DIRECTORY = Path("..") / "generated_data"

    generate_and_save_data(
        n_list=N_VALUES,
        mask_ratio_list=MASK_RATIOS,
        snr_list = SNR_VALUES,
        n_trials=NUM_TRIALS,
        n_dimensions=3,
        seed_base=2025,
        base_dir=OUTPUT_DIRECTORY
    )