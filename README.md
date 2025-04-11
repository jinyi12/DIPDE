# Denoisers as a Prior for Nonlinear PDE Inverse Problems

This repository implements a framework for solving nonlinear PDE inverse problems using pre-trained denoiser networks as a prior. The approach combines traditional optimization techniques with deep learning-based regularization to effectively recover coefficient fields in partial differential equations from noisy measurements.

## Overview

Solving inverse problems in PDEs is challenging due to ill-posedness and sensitivity to noise. This project demonstrates how trained denoiser networks can act as effective priors for coefficient field recovery, outperforming traditional regularization methods like Total Variation (TV) and L2 regularization.

### Key Features

- Implementation of a Finite Element Method (FEM) solver
- Denoiser-based regularization using convolutional neural networks
- Multiple regularization options (Denoiser, Total Variation, L2 norm, etc.)
- Flexible optimization approach with adaptive step sizing
- Scripts for dataset generation and model training
- Visualization tools for result analysis

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Denoisers-as-a-prior-for-Nonlinear-PDE-Inverse-Problems.git
cd Denoisers-as-a-prior-for-Nonlinear-PDE-Inverse-Problems
```

Required dependencies include:
- PyTorch
- NumPy
- Matplotlib
- SciPy
- tqdm
- Weights & Biases (wandb)

## Project Structure

- `src/` - Main source code
  - `ukappa_inverse_problem.py` - Main script for solving inverse problems
  - `train.py` - Training script for denoiser models
  - `kappa_denoiser.ipynb` - Notebook for interactive denoiser training
- `models/` - Neural network models
  - `denoiser.py` - Implementation of the CNN denoiser
- `fem/` - Finite Element Method implementation
  - `inverse_problem.py` - Classes for inverse problem setup
  - `regularizers.py` - Various regularization methods
  - `mesh.py`, `element.py`, etc. - FEM infrastructure
- `dataprep/` - Data preparation utilities
- `figures/`, `results/` - Output directories

## Usage

### Generate Dataset

#### Generate a dataset of non-Gaussian random coefficient fields:

```bash
python -m dataprep.genrf_vertices_coarse.py --n_samples 10000 --lx 1.0 --ly 1.0 --lc 0.2 --nx 32 --ny 32 --mu 2.7 --sigma 0.3 --error_threshold 1e-3 --seed 0
```

- `--n_samples`: Number of samples to generate
- `--lx`, `--ly`: Length of the domain
- `--lc`: Correlation length
- `--nx`, `--ny`: Number of grid points in the x and y directions
- `--mu`: Mean of the Gaussian distribution
- `--sigma`: Standard deviation of the Gaussian distribution
- `--error_threshold`: Error threshold for KL decomposition
- `--seed`: Random seed for reproducibility

#### Interpolate the non-Gaussian random coefficient fields:

```bash
python -m dataprep.genrf_vertices_interpolated.py --input randomfield --output randomfield_fine --nx_coarse 32 --ny_coarse 32 --nx_fine 256 --ny_fine 256 --method cubic --dataset_type train
```

- `--input`: Input folder path
- `--output`: Output folder path
- `--nx_coarse`, `--ny_coarse`: Number of grid points in the x and y directions of the coarse grid
- `--nx_fine`, `--ny_fine`: Number of grid points in the x and y directions of the fine grid
- `--method`: Interpolation method
- `--dataset_type`: Dataset type

#### Generate the solution field for corresponding coefficient fields:

```bash
python -m dataprep.generate_ukappa_dataset.py --data_path randomfield_fine --output_path randomfield_fine_solution --resolution 256 --num_samples (leave blank for all) --plot_examples 5 --use_wandb False --dataset_type train --seed 0
```

- `--data_path`: Path to the coefficient fields
- `--output_path`: Path to save the solution fields
- `--resolution`: Resolution of the coefficient fields
- `--num_samples`: Number of samples to process
- `--plot_examples`: Number of examples to plot
- `--use_wandb`: Whether to use wandb to download data
- `--dataset_type`: Dataset type
- `--seed`: Random seed for reproducibility

### Train a Denoiser

Train a denoiser model on the generated dataset:

```bash
./train.sh
```

### Solve Inverse Problem

Solve an inverse problem using different regularization techniques:

1. Using a trained denoiser as a prior:
```bash
./ukappa_inverse_problem_denoiser.sh
```

2. Using Total Variation regularization:
```bash
./ukappa_inverse_problem_TV.sh
```

3. Using L2 regularization:
```bash
./ukappa_inverse_problem_L2.sh
```

### Inverse Problem Solving

Run the inverse problem solver with custom parameters:

```bash
python -m src.ukappa_inverse_problem \
    --noise 0.01 \                   # Noise level for measurement data
    --num_samples 5 \                # Number of samples to process
    --max_iterations 500 \           # Maximum optimization iterations
    --initial_step_size 1.0 \        # Initial optimization step size
    --initial_lambda_reg 0.1 \       # Initial regularization strength
    --lambda_min_factor 0.1 \        # Min regularization as fraction of initial
    --lambda_schedule_iterations 50 \ # Iterations for lambda schedule
    --regularizer denoiser \         # Regularizer type (denoiser, tv, value, gradient)
    --p_norm 2                       # p-norm value (for value/gradient regularizers)
```

## Key Parameters

The main script `src/ukappa_inverse_problem.py` accepts the following important parameters:

- `--noise`: Noise level to add to the measurements
- `--num_samples`: Number of samples to process
- `--max_iterations`: Maximum number of optimization iterations
- `--initial_step_size`: Initial step size for optimization
- `--initial_lambda_reg`: Initial regularization strength
- `--regularizer`: Type of regularizer (denoiser, tv, value, gradient)
- `--p_norm`: p-norm value for value or gradient regularizers
- `--lambda_min_factor`: Minimum lambda as fraction of initial value
- `--lambda_schedule_iterations`: Iterations over which lambda decreases

## Example Results

The solver generates visualizations of:
- True vs. recovered coefficient fields
- Convergence metrics (objective function, error)
- Regularization strength evolution

Results are saved in the `figures/` and `results/` directories.


## License

This project is licensed under the ???..