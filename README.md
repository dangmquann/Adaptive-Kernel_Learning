# ATtention-based Kernel (ATK)

## Introduction
Many studies have shown that Alzheimer's disease can be diagnosed and detected early by applying machine learning models on biological data of subjects. 
While recent deep learning approaches offer strong performance, they often overlook structured prior knowledge and the relational interactions between subject samples. In this work, we propose Attention-based Kernel (ATK), a novel model that combines prior kernels with an adaptive learning mechanism to better exploit the structural information in unimodal data and guide the learning process. We validate ATK on Alzheimer’s disease diagnosis tasks on ADNI and ROSMAP datasets, demonstrating that it improves predictive performance compared to traditional kernel methods and standard deep learning baselines.

## Table of contents
- [Repository's structure](#repositorys-structure)
- [High-level model architecture](#high-level-model-architecture)
- [Guide to install and run code](#guide-to-install-and-run-code)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Repository's structure
```
.
├── dataloader/         # Data loading and preprocessing utilities
├── models/             # Model implementations (ATK, DKL, Linformer, Performer, SVM Kernel, MOGONET, MHA, MKL, GP)
├── experiments/        # Experiment scripts and result analysis
├── data/               # Datasets (AD_CN, AD_MCI, CN_MCI, ROSMAP)
├── figures/            # Figures and visualizations
├── requirements.txt    # Python dependencies
├── enviroments.yml     # Conda environment (optional)
└── README.md           # This file
```

## High-level model architecture

![Model Architecture](figures/atk.png)


- **ATK**: ATtention-based Kernel.
- **DKL**: Deep Kernel Learning with Gaussian Process.
- **MOGONET**: Multi-omics Graph Neural Network.
- **Other baselines**: GAT, Performer, Linformer, EasyMKL, etc.

## Guide to install and run code

### 1. Installation

**With Conda:**
```bash
conda env create -f enviroments.yml
conda activate atk_env
```

**Or with pip:**
```bash
pip install -r requirements.txt
```

### 2. Prepare data
- Example data is available in the `data/` folder.
- For your own data, place CSV files in the correct structure and update paths in the scripts.

### 3. Run experiments

**Step 1: Change to the experiment scripts directory**
```bash
cd experiments/scripts
```

**Step 2: Set the PYTHONPATH environment variable (to ensure correct module imports)**
```bash
export PYTHONPATH=$PWD/../..:$PYTHONPATH
```

**ATK:**

***Parameter descripttions***
- `--reg_coef`:A coefficient to decide the KL regularization balance when training ATK (e.g., `0.1`).
- `--num_layers`: Number of attention/kernel layers in the ATK model (e.g., `4`).
- `--num_head`: Number of attention heads in each multi-head attention layer (e.g., `3`).
- `--top_k`: Number of top neighbors (samples) to consider in the attention/kernel computation (e.g., `2`).

```bash
cd experiments/scripts
python run_ATKQK.py --dataset AD_CN --modality PET \
--lr 0.0004 --patience 30 --epochs 5000 --reg_coef 0.1 \
--num_layers 4 --num_head 3 --top_k 2 \
```

**ATK without QK:**

***Incorporating a prior kernel***
- `--use_prior`:  
  Set this flag to `True` to incorporate a prior kernel into the model. If set to `False`, the model will use a random kernel instead of any prior knowledge.

- `--prior_method`:  
  Specifies the method for incorporating the prior kernel when `--use_prior` is `True`.  
  - `1`: Use the sum of the learned kernel and the prior kernel.  
  - `2`: Use only the prior kernel (ignore the learned kernel).

- Random Kernel (no prior knowledge):
```bash
python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
--use_prior False --reg_coef 0.1 \
--lr 0.001 --patience 20 --epochs 2000 \
--num_layers 5 --num_head 4 --top_k 2 \
```
- Prior Kernel (use only the prior kernel)
```bash
python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
--use_prior True --prior_method 2 --reg_coef 0.1 \
--lr 0.001 --patience 20 --epochs 2000 \
--num_layers 5 --num_head 4 --top_k 2 \
```
- Sum Kernel
```bash
python run_ATK_noQK.py --dataset AD_CN --modality PET GM CSF --selected_modality PET \
--use_prior True --prior_method 1 --reg_coef 0.1 \
--lr 0.001 --patience 20 --epochs 2000 \
--num_layers 5 --num_head 4 --top_k 2 \
```

- See more in `experiments/train.sh` or scripts in `experiments/scripts/`.


### 4. Analyze results
- Results are saved in `results/` and visualizations in `figures/`.
- Use `experiments/extract_results.py` to summarize results (table) in `result_analysis/`.
```bash
cd experiments/extract_results.py
python extract_results.py
```

### 5. Reproduce all results from the paper

To reproduce all experiments and results as reported in the paper, follow these steps:

**Step 1: Install dependencies and prepare the environment**
- See [Installation](#1-installation) and [Prepare data](#2-prepare-data) above.

**Step 2: Run all experiment scripts**
- All scripts for reproducing the main results are provided in `experiments/train.sh` and the scripts in `experiments/scripts/`.

```bash
cd experiments
bash table_2.sh
bash table_3.sh
```
- This will automatically run all main experiments for the datasets and methods described in the paper.

**Step 3: Analyze and summarize results**
- After running the experiments, results will be saved in the `results/` directory.
- To extract and summarize the results into tables and figures as in the paper, run:

```bash
cd experiments
python extract_results.py
```
- Visualizations and summary tables will be available in the `figures/` and `result_analysis/` folders.

> **Note:**  
> For custom or additional experiments, you can modify or add scripts in `experiments/scripts/` and update `train.sh` accordingly.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation
```bibtex
@misc{atk2025,
  title={ATtention-based Kernel Learning},
  author={Duy Thanh Vu, Quan Dang Minh et al.},
  year={2025},
  howpublished={\url{https://github.com/your-repo-link}}
}
```

