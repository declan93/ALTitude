# ALTtitude

**A Machine Learning Pipeline for Predicting Alternative Lengthening of Telomeres (ALT) in Cancer**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ALTtitude is a robust machine learning pipeline for predicting **Alternative Lengthening of Telomeres (ALT)** activity in cancer samples. ALT is a telomere maintenance mechanism used by approximately 10-15% of human cancers to achieve cellular immortality, making its detection clinically significant for prognosis and treatment planning.

### Key Features

- **Ensemble Stacking Architecture**: Combines 9+ base models with meta-learners for robust predictions
- **Adaptive Cross-Validation**: Automatic selection between k-fold and leave-one-out CV based on dataset size
- **Missing Data Handling**: Native support for NaN values in tree-based models
- **Threshold Optimization**: Domain-specific decision boundary tuning for clinical applications
- **Comprehensive Logging**: Full prediction tracking, feature importance analysis, and SHAP values
- **Publication-Ready Visualizations**: ROC curves, precision-recall curves, and model comparisons

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm](#algorithm)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Input/Output Formats](#inputoutput-formats)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation](#citation)

---

## Installation

### Requirements

- Python 3.8 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/ALTtitude.git
cd ALTtitude

# Install required packages
pip install numpy pandas scikit-learn xgboost lightgbm catboost torch matplotlib seaborn

# Optional: Install SHAP for feature interpretation
pip install shap
```

### Dependencies List

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.19.0 | Numerical operations |
| pandas | ≥1.1.0 | Data manipulation |
| scikit-learn | ≥0.23.0 | ML algorithms & utilities |
| xgboost | ≥1.3.0 | Gradient boosting |
| lightgbm | ≥3.0.0 | Light gradient boosting |
| catboost | ≥0.24.0 | Categorical boosting |
| torch | ≥1.7.0 | TabPFN meta-learner |
| matplotlib | ≥3.2.0 | Visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| shap | ≥0.39.0 | Feature interpretation (optional) |

---

## Quick Start

```bash
# Train a model
python ALTitude_cli.py train your_data.tsv

# Make predictions on new samples
python ALTitude_cli.py predict new_samples.tsv --model-dir ./results/

# View model performance
python ALTitude_cli.py evaluate --model-dir ./results/
```

---

## Algorithm

ALTtitude uses a sophisticated **ensemble stacking** approach that combines multiple machine learning models for robust ALT prediction.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                │
│         (Telomere features, expression data, genomic markers)    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                 │
│  • Train/Test Split (70/30, stratified)                         │
│  • Missing Value Imputation (mean, for linear models)           │
│  • Feature Validation                                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BASE MODELS (Level 0)                         │
├─────────────────────────────────────────────────────────────────┤
│  Tree-Based (NaN-native)     │  Ensemble Trees                  │
│  ├── XGBoost                 │  ├── Random Forest               │
│  ├── CatBoost                │  ├── ExtraTrees                  │
│  ├── LightGBM                │  └── AdaBoost                    │
│  └── HistGradientBoosting    │                                  │
├─────────────────────────────────────────────────────────────────┤
│  Linear/Other                │  Meta-Learning                   │
│  ├── Logistic Regression     │  ├── TabPFN (optional)           │
│  └── Linear SVC              │  └── Naive Bayes                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                    (Out-of-Fold Predictions)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL SELECTION                                  │
│  • Rank by CV AUC performance                                   │
│  • Select top N models (default: 5)                             │
│  • Stack predictions as meta-features                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 META-LEARNERS (Level 1)                          │
│  ├── Meta Logistic Regression                                   │
│  ├── Meta Random Forest                                         │
│  └── Meta XGBoost                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 THRESHOLD OPTIMIZATION                           │
│  • Youden's J (balanced)                                        │
│  • Sensitivity-focused                                          │
│  • Specificity-focused                                          │
│  • ALT-focused (≥80% sensitivity priority)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
│  • Probability scores [0.0 - 1.0]                               │
│  • Binary predictions at optimized threshold                    │
│  • Confidence metrics                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

1. **Data Loading & Validation**
   - Load tab-separated input data
   - Validate required columns (ALT target, features)
   - Drop metadata columns (ID, annotations)

2. **Data Splitting**
   - Stratified train/test split (default 70/30)
   - Preserves class balance in both sets

3. **Cross-Validation**
   - 5-fold stratified CV (default)
   - Leave-one-out CV available for small datasets (≤500 samples)
   - Generates out-of-fold predictions for stacking

4. **Base Model Training**
   - Train 9+ diverse algorithms in parallel
   - Each model evaluated on CV and test set
   - Metrics: AUC, Sensitivity, Specificity, Precision

5. **Model Selection & Stacking**
   - Rank models by CV performance
   - Select top performers (default: 5)
   - Create stacked features from predictions

6. **Meta-Learner Training**
   - Train meta-learners on stacked predictions
   - Learn optimal combination of base models

7. **Threshold Optimization**
   - Find optimal decision boundary
   - Multiple optimization strategies available

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve - primary performance metric |
| **Sensitivity** | True positive rate (TP / (TP + FN)) |
| **Specificity** | True negative rate (TN / (TN + FP)) |
| **Precision** | Positive predictive value (TP / (TP + FP)) |
| **F1 Score** | Harmonic mean of precision and recall |
| **Youden's J** | Sensitivity + Specificity - 1 |

---

## Usage

### Training

```bash
# Basic training
python ALTitude_cli.py train data/training_data.tsv

# With threshold optimization
python ALTitude_cli.py train data/training_data.tsv \
    --optimize-threshold \
    --threshold-metric alt_focused

# With leave-one-out cross-validation
python ALTitude_cli.py train data/training_data.tsv --loo-cv

# Full options
python ALTitude_cli.py train data/training_data.tsv \
    --output-dir ./my_models \
    --seed 42 \
    --test-size 0.3 \
    --cv-folds 5 \
    --optimize-threshold \
    --threshold-metric balanced \
    --min-sensitivity 0.8 \
    --analyze-thresholds
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./results` | Output directory for models and results |
| `--seed` | `42` | Random seed for reproducibility |
| `--test-size` | `0.3` | Proportion of data for test set |
| `--cv-folds` | `5` | Number of cross-validation folds |
| `--loo-cv` | `False` | Use leave-one-out cross-validation |
| `--loo-max-samples` | `500` | Maximum samples for LOO CV |
| `--optimize-threshold` | `False` | Find optimal classification threshold |
| `--threshold-metric` | `balanced` | Optimization strategy (see below) |
| `--min-sensitivity` | `None` | Minimum required sensitivity |
| `--min-specificity` | `None` | Minimum required specificity |
| `--analyze-thresholds` | `False` | Generate threshold analysis plots |
| `--no-plots` | `False` | Skip visualization generation |

#### Threshold Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `balanced` | Youden's J statistic - maximizes TPR + TNR |
| `sensitivity` | Maximizes detection rate (minimize false negatives) |
| `specificity` | Minimizes false positives |
| `f1` | Maximizes F1 score |
| `alt_focused` | Prioritizes ≥80% sensitivity, then maximizes specificity |

### Prediction

```bash
# Basic prediction
python ALTitude_cli.py predict data/new_samples.tsv --model-dir ./results/

# With custom threshold
python ALTitude_cli.py predict data/new_samples.tsv \
    --model-dir ./results/ \
    --threshold 0.4

# Test multiple thresholds
python ALTitude_cli.py predict data/new_samples.tsv \
    --model-dir ./results/ \
    --multi-threshold "0.3,0.4,0.5,0.6,0.7"

# Save to specific file
python ALTitude_cli.py predict data/new_samples.tsv \
    --model-dir ./results/ \
    --output-file predictions.csv
```

#### Prediction Options

| Option | Description |
|--------|-------------|
| `--model-dir` | Directory containing trained models |
| `--threshold` | Classification threshold (default: 0.5) |
| `--multi-threshold` | Comma-separated list of thresholds to test |
| `--output-file` | Output file path for predictions |

### Evaluation

```bash
# View model performance summary
python ALTitude_cli.py evaluate --model-dir ./results/
```

---

## Configuration

### Config File (JSON)

Create a `config.json` file for reproducible experiments:

```json
{
    "random_seed": 42,
    "test_size": 0.3,
    "cv_folds": 5,
    "n_top_models": 5,
    "use_loo_cv": false,
    "loo_max_samples": 500,
    "use_feature_selection": false,
    "n_features_to_select": 20,
    "output_dir": "./results"
}
```

Use with:
```bash
python ALTitude_cli.py train data.tsv --config config.json
```

---

## Input/Output Formats

### Input Data Format

Tab-separated values (TSV) file with:

| Column | Type | Description |
|--------|------|-------------|
| `ALT` | Binary (0/1) | Target variable - ALT status |
| Features | Numeric | Telomere metrics, expression values, etc. |

**Example structure:**
```
ALT     TelSeq    TelFusDetector_rate    AAAGGG    AACGGG    ...
1       0.453     0.0012                 0.234     0.156     ...
0       0.122     0.0003                 0.089     0.045     ...
1       0.567     0.0018                 0.312     0.201     ...
```

**Auto-dropped columns:**
- ID, ID_BUCKET
- Achilles, order, StripCI
- ATRX_DAXX, Age, DG, SMARCAL1
- Other metadata columns

### Output Directory Structure

```
results/
├── config.json                    # Training configuration
├── model_summary.csv              # Performance metrics for all models
├── model_summary_report.txt       # Human-readable summary
├── optimal_thresholds.json        # Optimized thresholds per model
├── class_distribution.json        # Class balance statistics
│
├── training_results/
│   ├── training_report_*.txt      # Detailed training report
│   ├── training_results_*.json    # Comprehensive results
│   ├── cv_results_*.csv           # Cross-validation metrics
│   └── test_results_*.csv         # Test set metrics
│
├── predictions/
│   ├── oof_predictions_*.csv      # Out-of-fold predictions
│   ├── test_predictions_*.csv     # Test set predictions
│   └── predictions_summary_*.json
│
├── feature_importances/
│   ├── feature_importances.csv    # Ranked feature importance
│   ├── feature_importances.json   # Detailed statistics
│   ├── shap_summary.csv           # SHAP values (if available)
│   └── importance_report.txt      # Text report
│
└── images/
    ├── model_comparison.png       # Model performance comparison
    ├── roc_curves.png             # ROC curves
    ├── pr_curves.png              # Precision-recall curves
    ├── confusion_matrix_*.png     # Confusion matrices
    └── threshold_analysis.png     # Threshold effects
```

### Prediction Output Format

CSV file with columns:
- Original input columns (ID, metadata)
- `{ModelName}_prob`: Probability score from each model
- `{ModelName}_pred`: Binary prediction at threshold
- `SE_{MetaLearner}_prob`: Stacked ensemble probability
- `SE_{MetaLearner}_pred`: Stacked ensemble prediction
- `mean_prob`, `median_prob`, `max_prob`, `min_prob`: Summary statistics

---

## API Reference

### Programmatic Usage

```python
from altitude_core import MLPipeline, Config
from altitude_model_manager import ModelManager

# Configure pipeline
config = Config(
    data_path='training_data.tsv',
    output_dir='./results',
    random_seed=42,
    test_size=0.3,
    cv_folds=5
)

# Train models
pipeline = MLPipeline(config)
results = pipeline.train()

# Save trained models
model_manager = ModelManager('./results')
model_manager.save_training_state(
    fitted_models=results['fitted_models'],
    meta_learners=results['fitted_meta_learners'],
    data_imputer=results.get('data_imputer'),
    data_scaler=results.get('data_scaler'),
    meta_scaler=results.get('meta_scaler'),
    selected_models=results['selected_models'],
    selected_indices=results['selected_indices'],
    config=config,
    model_summary=results['summary']
)

# Load and predict on new data
state = model_manager.load_training_state()
pipeline.load_state(state)
predictions = pipeline.predict('new_data.tsv', threshold=0.5)
predictions.to_csv('predictions.csv', index=False)
```

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Config` | altitude_core | Configuration dataclass |
| `MLPipeline` | altitude_core | Main training/prediction pipeline |
| `DataLoader` | altitude_core | Data loading and preprocessing |
| `ModelFactory` | altitude_core | Model instantiation |
| `ModelTrainer` | altitude_core | Training orchestration |
| `MetricsCalculator` | altitude_core | Performance metrics |
| `ModelManager` | altitude_model_manager | Model persistence |
| `ThresholdOptimizer` | altitude_threshold | Threshold optimization |
| `FeatureImportanceExtractor` | altitude_feature_importance | Feature analysis |
| `ALTitudeVisualizer` | simple_visualizer | Visualization generation |

---

## Examples

### Example 1: Basic Training and Prediction

```bash
# Train on your dataset
python ALTitude_cli.py train my_cancer_samples.tsv --output-dir ./alt_model

# Predict on new samples
python ALTitude_cli.py predict new_samples.tsv \
    --model-dir ./alt_model \
    --output-file predictions.csv
```

### Example 2: Optimized for Increased Sensitivity

```bash
# Train with high sensitivity (minimize missed ALT cases)
python ALTitude_cli.py train training_data.tsv \
    --optimize-threshold \
    --threshold-metric sensitivity \
    --min-sensitivity 0.90 \
    --analyze-thresholds \
    --output-dir ./clinical_model
```

### Example 3: Small Dataset with LOO-CV

```bash
# Use leave-one-out for datasets with <100 samples
python ALTitude_cli.py train small_dataset.tsv \
    --loo-cv \
    --loo-max-samples 100 \
    --output-dir ./small_model
```

### Example 4: Complete Research Pipeline

```python
from altitude_core import MLPipeline, Config
from altitude_threshold import ThresholdOptimizer
from simple_visualizer import ALTitudeVisualizer

# Configuration
config = Config(
    data_path='data/training_cohort.tsv',
    output_dir='./research_results',
    random_seed=42,
    test_size=0.3,
    cv_folds=5,
    n_top_models=5
)

# Train
pipeline = MLPipeline(config)
results = pipeline.train()

# Optimize threshold for ALT detection
optimizer = ThresholdOptimizer()
optimal = optimizer.optimize(
    y_true=results['test_labels'],
    y_prob=results['test_predictions'],
    metric='alt_focused',
    min_sensitivity=0.80
)
print(f"Optimal threshold: {optimal['threshold']:.3f}")
print(f"Sensitivity: {optimal['sensitivity']:.3f}")
print(f"Specificity: {optimal['specificity']:.3f}")

# Generate visualizations
visualizer = ALTitudeVisualizer(config.output_dir)
visualizer.plot_roc_curves(results)
visualizer.plot_model_comparison(results)
visualizer.plot_confusion_matrices(results, threshold=optimal['threshold'])
```

---

## File Structure

```
ALTtitude/
├── ALTitude_cli.py                # Command-line interface
├── altitude_core.py               # Main ML pipeline
├── altitude_model_manager.py      # Model persistence
├── altitude_threshold.py          # Threshold optimization
├── altitude_feature_importance.py # Feature analysis
├── altitude_utils.py              # Utilities
├── altitude_results_logger.py     # Results logging
├── altitude_prediction_logger.py  # Prediction logging
├── simple_visualizer.py           # Visualization
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

---

## Citation

If you use ALTtitude in your research, please cite:

```bibtex
@software{altitude2024,
  title={Title},
  author={[Author Names]},
  year={},
  url={https://github.com/yourusername/ALTtitude}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Support

For questions and support:
- Open an issue on GitHub
- Contact: [your-email@example.com]
