# AI_Mg_Alloy_Design

**A Physics-Informed Machine Learning Framework for Sustainable Magnesium Alloy Design**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Overview

This repository provides the implementation of a **physics-informed ensemble (PIE) learning framework** for predicting the yield strength (YS) and tensile strength (TS) of magnesium (Mg) alloys. The framework integrates multi-scale material descriptors, classical strengthening theories (Hall-Petch, solute strengthening, precipitate strengthening), and ensemble tree-based machine learning models to establish quantitative composition-microstructure-property relationships.

### Key Features

- **Physics-Informed Descriptors**: Incorporates elemental properties, compositional statistics, microstructural features (grain size, precipitate habit plane, phase fraction), and empirical strength estimates as model inputs.
- **Ensemble Learning**: Combines Random Forest and XGBoost via a two-step voting strategy for robust and accurate strength prediction.
- **Two-Tier Model Architecture**:
  - **PIE Model G (General)**: ~280 features, suitable for broad Mg alloy systems with basic microstructural information.
  - **PIE Model S (Expert)**: Top-10 features + quantitative precipitate descriptors, for high-precision prediction with detailed precipitate characterization.
- **SHAP Interpretability**: Global and local feature importance analysis via Shapley additive explanations.
- **High-Throughput Screening**: Tools for generating and screening millions of candidate Mg alloy compositions.

## Repository Structure

```
AI_Alloy/
├── Yield_Strength/          # YS model training pipeline (10 algorithms → top-5 → ensemble)
├── Tensile_Strength/        # TS model training pipeline (same workflow as YS)
├── Small_Data_new/          # Expert model (PIE Model S) training with precipitate features
├── Final_Model/             # Final ensemble sub-models and prediction scripts
├── shap_new/                # SHAP analysis for the general model (PIE Model G)
├── model_analysis/          # Single-variable analysis, decision path analysis, convergence plots
├── operations/              # High-throughput virtual dataset generation and screening
├── feature_compare/         # Comparison of different descriptor sets (9 types)
├── design/                  # Alloy composition grid generation and Hall-Petch filtering
├── Original_Data/           # Raw experimental dataset and validation data
├── Dataset_process.ipynb    # Full data processing pipeline (normalization, encoding, feature expansion)
├── strength_calculation.py  # Empirical strength calculations (solid solution, grain boundary, theoretical YS)
├── FULL_general.xlsx        # Processed dataset with all features and labels
└── Final_train_data.xlsx    # Final training dataset
```

## Installation

```bash
# Clone the repository
git clone https://github.com/nwpusl/AI_Alloy.git
cd AI_Alloy

# Install dependencies
pip install -r requirements.txt
```

**Dependencies**: numpy, pandas, scikit-learn, xgboost, shap, matminer, pymatgen, matplotlib, seaborn, openpyxl

## Usage

### 1. Data Processing
Run `Dataset_process.ipynb` to process raw experimental data through normalization, categorical encoding, and matminer-based feature expansion.

### 2. Model Training
Navigate to `Yield_Strength/` or `Tensile_Strength/` and execute the notebooks in order:
1. `data_set_process.ipynb` — Outlier removal and train/test split
2. `10_algorithms_compare.ipynb` — Grid search across 10 regression algorithms
3. `top_5_algorithms_optimization.ipynb` — Hyperparameter optimization for top-5 models
4. `weights_selection.ipynb` — Ensemble voting weight optimization
5. `get_final_ensemble_sub_models.ipynb` — 5-fold cross-validation training of final sub-models

### 3. Prediction
Use the trained ensemble models in `Final_Model/` for strength prediction on new alloy compositions.

### 4. High-Throughput Screening
In `operations/`, run sequentially:
```bash
python Dataset_generate.py    # Generate virtual alloy compositions
python Dataset_predict.py     # Predict strengths using trained models
python Dataset_select.py      # Filter alloys meeting strength thresholds
```

## Model Performance

| Model | Target | R² | MAE (MPa) |
|-------|--------|-----|-----------|
| PIE Model G (Model 6) | Yield Strength | ~0.80 | ~25.8 |
| PIE Model G (Model 6) | Tensile Strength | ~0.80 | ~25.8 |
| PIE Model S (Model 7) | Yield Strength | ~0.88 | ~19.0 |
| PIE Model S (Model 7) | Tensile Strength | ~0.88 | ~22.0 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2025physics,
  title={A physics-informed machine learning framework for sustainable alloy design},
  author={Li, Quan and Wang, Zhilong and Shi, Lei and Feng, Baojing and Liu, Changhui 
          and Wang, Jie and Wang, Jingya and Zheng, Weisen 
          and Jovičević-Klug, Matic and Rao, Ziyuan and Li, Jinjin 
          and Raabe, Dierk and Zeng, Xiaoqin},
  journal={Under review},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions, please open an issue on GitHub or contact the corresponding authors.
