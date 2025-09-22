# BMC-GC: Bayesian Matrix Completion under Geometrix Constraints

A Python implementation of Bayesian methods for completing noisy and partial Euclidean Distance Matrices (EDMs) modeled by a Normal-wishart hyperprior and sampled using Metropolis-Hastings within Gibbs sampler

## Overview

This project implements:
1. **Bayesian Completion**: Probabilistic approach using Metropolis-Hastings within Gibbs sampler with Normal-Wishart hyperpriors

The method is designed to reconstruct missing distances in EDMs while preserving the geometric properties of the underlying point configurations.

## Features

- **Bayesian EDM Completion**: Gibbs sampler with Metropolis-Hastings updates for point positions
- **Alternating Optimization**: Barzilai-Borwein gradient descent with Augmented Lagrangian formulation
- **Comprehensive Evaluation**: Multiple error metrics including Frobenius norm and reconstruction quality
- **Parallel Simulations**: Multi-core support for Monte Carlo experiments
- **Noise Handling**: Built-in support for noisy distance observations
- **Visualization**: Plotting utilities for performance analysis

## Core Components

### Main Modules

- `bpmc.py` - Bayesian EDM completion implementation
- `optimization.py` - Augmented Lagrangian optimization methods
- `evaluation.py` - Error metrics and performance evaluation
- `math_utils.py` - Mathematical utilities for EDM operations
- `generate_data.py` - Synthetic data generation with noise models
- `run_bpmc_simulations.py` - Simulation pipeline and experiment orchestration

### Key Algorithms

**Bayesian Completion**
- Normal-Wishart priors for point positions and precision parameters
- Gibbs sampling with per-point Metropolis-Hastings updates
- Posterior predictive distance matrix estimation

**Alternating Optimization**
- Low-rank factorization of EDMs
- Augmented Lagrangian constraint handling
- Barzilai-Borwein step size selection

## Usage

### Basic Example

```python
import numpy as np
from bpmc import BayesianEDMCompletion
from optimization import alternating_completion

# Generate synthetic EDM with missing entries
n_points, n_dimensions = 50, 3
D_true = generate_ground_truth_edm(n_points, n_dimensions)
mask = create_mask(n_points, rate=0.7)  # 70% observed
D_obs = D_true * mask

# Bayesian completion
D_bayes = BayesianEDMCompletion(
    D_obs=D_obs, 
    Mask=mask, 
    T=1000,           # MCMC iterations
    rank=n_dimensions,
    burn_in=500
)

# Alternating optimization
P_alt = alternating_completion(D_obs, mask, rank=n_dimensions)
```

### Running Experiments

```python
from run_bpmc_simulations import run_experiments_on_mask_ratios

# Compare methods across different observation ratios
mask_ratios = [0.3, 0.5, 0.7, 0.9]
results = run_experiments_on_mask_ratios(
    mask_ratios=mask_ratios,
    n_points=30,
    n_dimensions=3,
    n_monte_carlo=50,
    results_dir="results/mask_experiment"
)
```

## Dependencies

- `numpy` - Numerical computations
- `scipy` - Statistical distributions and optimization
- `pandas` - Data manipulation and analysis
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars for long-running computations
- `torch` - GPU-accelerated tensor operations (optional)

## Installation

```bash
git clone https://github.com/RohitVarmaSixtyFive/BMC-GC.git
cd BMC-GC
pip install numpy scipy pandas matplotlib tqdm
```

## Experimental Framework

The simulation framework supports:

- **Mask Ratio Experiments**: Performance vs. observation density
- **Scale Experiments**: Behavior with varying problem sizes
- **Noise Experiments**: Robustness to measurement errors
- **Monte Carlo Analysis**: Statistical significance testing

Results are automatically saved with timestamps and summary statistics.

## Mathematical Background

The methods are based on the theoretical framework from:
- "Bayesian Matrix Completion under Geometric Constraints" by Rohit Varma, Santosh Nannuru et al.
- "Exact Reconstruction of Euclidean Distance Geometry Problem Using Low-rank Matrix Completion" by Abiy Tasissa, Rongjie Lai et al.

## File Structure

```
BMC-GC/
├── bpmc.py                     # Bayesian completion core
├── bpmc_noise.py              # Noise-aware variants
├── optimization.py            # Alternating optimization
├── evaluation.py              # Performance metrics
├── math_utils.py              # Mathematical utilities
├── generate_data.py           # Data generation
└── run_bpmc_simulations.py    # Experiment pipeline
```

## Output

The simulation pipeline generates:
- Per-iteration CSV files with detailed results
- Summary statistics with confidence intervals
- Comparative performance plots
- Timestamped experiment logs