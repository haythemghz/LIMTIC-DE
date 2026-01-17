<<<<<<< HEAD
# LIMTIC-DE Experiments

This repository contains the official implementation of the experiments for the paper: **"LIMTIC-DE: A Comprehensive Dataset and Benchmark for Deglet Nour Date Fruit Classification"**.

[GitHub Repository](https://github.com/haythemghz/LIMTIC-DE)

It includes code for training baseline models (ViT, ResNet50, EfficientNet-B0), conducting ablation studies (viewpoint analysis), and generating statistical visualizations.

## 1. Setup

### Requirements
Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Structure
The code expects the LIMTIC-DE dataset to be located at `../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset` relative to this folder, or you can modify `DATA_DIR` in the scripts. The expected structure is:

```
LIMTIC-DE_Dataset/
├── Train/
├── Validation/
└── Test/
```

## 2. Reproducing Benchmarks

### Unified Runner
To execute the full **5-independent run over 5-seeds** protocol ($n=5$) for all baseline models:

```powershell
./run_all_experiments.ps1
```

### Manual Execution
To train individual models or run custom seeds:

```bash
# Run ViT with specific seed
python train_v2.py --model vit --epochs 10 --seed 42 --save_dir results/vit_seed42
```

### Ablation Studies
To reproduce the viewpoint contribution analysis (Full vs. Side views):

```bash
# Full View Only
python train_v2.py --model vit --view_mode full --save_dir results_ablation/vit_full

# Side View Only
python train_v2.py --model vit --view_mode side --save_dir results_ablation/vit_side
```

## 3. Analysis & Statistical Validation

To calculate p-values (T-test & Wilcoxon) using the real outputs from the `results_robustness/` directory:
```bash
python calculate_significance.py
```

### Advanced Visualization (t-SNE & Per-Class Analysis)
To generate the t-SNE plot (Figure 10 in manuscript) and per-class accuracy table:

```bash
# Generate t-SNE Plot
python generate_tsne.py

# Generate Per-Class View Contribution Table
python analyze_view_contributions.py
```

## 4. Hardware
Experiments were conducted on an **NVIDIA Quadro P1000 (4GB)**.
- **Inference Latency**: ~14.8ms (ViT), ~4.9ms (ResNet50).

## 5. Directory Layout
- `train_v2.py`: Main training script.
- `experiment_utils.py`: Datasets and transforms.
- `generate_tsne.py`: t-SNE visualization script.
- `analyze_view_contributions.py`: Per-class accuracy analysis.
- `calculate_significance.py`: Statistical testing script.
- `results/`: Directory for experiment outputs.
=======
# LIMTIC-DE Experiments

This repository contains the official implementation of the experiments for the paper: **"LIMTIC-DE: A Comprehensive Dataset and Benchmark for Deglet Nour Date Fruit Classification"**.

[GitHub Repository](https://github.com/haythemghz/LIMTIC-DE)

It includes code for training baseline models (ViT, ResNet50, EfficientNet-B0), conducting ablation studies (viewpoint analysis), and generating statistical visualizations.

## 1. Setup

### Requirements
Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Structure
The code expects the LIMTIC-DE dataset to be located at `../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset` relative to this folder, or you can modify `DATA_DIR` in the scripts. The expected structure is:

```
LIMTIC-DE_Dataset/
├── Train/
├── Validation/
└── Test/
```

## 2. Reproducing Benchmarks

### Unified Runner
To execute the full **5-independent run over 5-seeds** protocol ($n=5$) for all baseline models:

```powershell
./run_all_experiments.ps1
```

### Manual Execution
To train individual models or run custom seeds:

```bash
# Run ViT with specific seed
python train_v2.py --model vit --epochs 10 --seed 42 --save_dir results/vit_seed42
```

### Ablation Studies
To reproduce the viewpoint contribution analysis (Full vs. Side views):

```bash
# Full View Only
python train_v2.py --model vit --view_mode full --save_dir results_ablation/vit_full

# Side View Only
python train_v2.py --model vit --view_mode side --save_dir results_ablation/vit_side
```

## 3. Analysis & Statistical Validation

To calculate p-values (T-test & Wilcoxon) using the real outputs from the `results_robustness/` directory:
```bash
python calculate_significance.py
```

### Advanced Visualization (t-SNE & Per-Class Analysis)
To generate the t-SNE plot (Figure 10 in manuscript) and per-class accuracy table:

```bash
# Generate t-SNE Plot
python generate_tsne.py

# Generate Per-Class View Contribution Table
python analyze_view_contributions.py
```

## 4. Hardware
Experiments were conducted on an **NVIDIA Quadro P1000 (4GB)**.
- **Inference Latency**: ~14.8ms (ViT), ~4.9ms (ResNet50).

## 5. Directory Layout
- `train_v2.py`: Main training script.
- `experiment_utils.py`: Datasets and transforms.
- `generate_tsne.py`: t-SNE visualization script.
- `analyze_view_contributions.py`: Per-class accuracy analysis.
- `calculate_significance.py`: Statistical testing script.
- `results/`: Directory for experiment outputs.
>>>>>>> origin/main
