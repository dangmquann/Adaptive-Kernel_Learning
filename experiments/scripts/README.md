# ATKQK Parameter Tuning Guide

## Overview

This script provides **parameter tuning** for the ATKQK model with two strategies:
- **Sequential tuning**: Optimizes one parameter at a time in a predefined order.
- **Grid search**: Tests all combinations of selected parameters in a grid.

Results are saved as `.csv` files and performance plots.

---

## Tuning Strategies

### 1. Sequential Tuning (Default)

- Optimizes parameters one by one (learning rate, number of layers, number of heads, etc.).
- At each step, only one parameter is changed; others are fixed at their current best values.
- Faster than grid search and suitable for large parameter spaces.

### 2. Grid Search

- Tests all possible combinations of selected parameters (usually a small subset).
- More time-consuming but can find the best combination within the grid.

---

## How to Run

### 1. Tune all datasets and modalities (sequential, default)

```bash
python tuning_ATKQK.py --strategy sequential
```

### 2. Tune a specific dataset and modality

```bash
python tuning_ATKQK.py --strategy sequential --dataset AD_CN --modality PET
```

### 3. Run grid search for all datasets and modalities

```bash
python tuning_ATKQK.py --strategy grid
```

## Results

- Tuning results are saved in the `experiments/results/tuning_parameter/` directory.
- **Performance plots**:  
  For each modality, the effect of each parameter is visualized as a plot.  
  - For the PET (imaging) modality, example plots showing the impact of each parameter are saved in:  
    ![PET modality parameter k impact example](results/tuning_parameter/AD_CN_PET/sequential/top_k_performance.png)
    ![PET modality parameter reg_coef impact example](results/tuning_parameter/AD_CN_PET/sequential/reg_coef_performance.png)
  - For other modalities, the plots are saved in their respective folders under `results/tuning_parameter/`.
- **Best parameter summary**:  
  - The file `all_best_parameters_sequential.csv` summarizes the best parameters found for each modality and dataset combination.
  - For grid search, see `all_best_parameters_grid.csv`.
