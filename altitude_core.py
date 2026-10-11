"""
ALTitude Core - Machine Learning Pipeline for ALT (Alternative Lengthening of Telomeres) Prediction

This module implements a comprehensive machine learning pipeline for binary classification
of ALT status in cancer samples. It supports multiple ML algorithms, ensemble methods,
and flexible cross-validation strategies.

Architecture Overview
---------------------
The pipeline follows a standard ML workflow:

    1. Data Loading & Preprocessing (DataLoader)
       - Load tab-separated data files
       - Stratified train/test split (default 70/30)
       - Imputation for models that don't handle NaN
       - Class distribution verification

    2. Model Training (ModelTrainer)
       - Train multiple base models (XGBoost, CatBoost, LightGBM, etc.)
       - Generate Out-of-Fold (OOF) predictions via cross-validation
       - Support for k-fold CV or Leave-One-Out CV

    3. Ensemble Creation (MLPipeline)
       - Select top N models based on CV performance
       - Train meta-learners (stacking ensemble) on OOF predictions
       - Evaluate on held-out test set

    4. Deployment Model Refitting (Optional)
       - Refit all models on 100% of data (train + test) for deployment
       - Separate from evaluation models to prevent data leakage in metrics

Cross-Validation Strategy
-------------------------
Two CV strategies are supported:

    - **k-fold CV** (default): Faster, suitable for larger datasets
      - Configurable number of folds (default: 5)
      - Generates OOF predictions for all training samples

    - **Leave-One-Out CV**: More thorough, suitable for small datasets
      - Each sample is held out once for validation
      - Maximum sample limit to prevent excessive computation

Model Types
-----------
Base models are categorized by their ability to handle missing values:

    **NaN-native models** (use raw data):
    - XGBoost
    - CatBoost
    - LightGBM
    - HistGradientBoosting

    **Imputation-required models** (use imputed data):
    - Logistic Regression
    - Random Forest
    - ExtraTrees
    - AdaBoost
    - TabPFN (optional)

Ensemble (Stacking)
-------------------
Meta-learners are trained on stacked OOF predictions from top base models:

    - Logistic Regression
    - Random Forest
    - XGBoost

The ensemble predictions are prefixed with "SE_" (Stacking Ensemble).

Deployment vs Evaluation Models
-------------------------------
When `refit_final_on_all_data=True`:

    - **Evaluation models**: Trained on training set only (e.g., 90%)
      Used for test set evaluation and reported metrics

    - **Deployment models**: Trained on ALL data (100%)
      Used for predictions on new, unseen data

This separation ensures reported metrics are unbiased while maximizing
predictive performance for deployment.

Usage Example
-------------
    >>> from altitude_core import MLPipeline, Config
    >>>
    >>> # Configure the pipeline
    >>> config = Config(
    ...     data_path="training_data.tsv",
    ...     output_dir="./results",
    ...     cv_folds=5,
    ...     refit_final_on_all_data=True  # Enable deployment models
    ... )
    >>>
    >>> # Train models
    >>> pipeline = MLPipeline(config)
    >>> results = pipeline.train()
    >>>
    >>> # Make predictions on new data
    >>> pipeline.load_state(results, use_deployment_models=True)
    >>> predictions = pipeline.predict("new_samples.tsv")

CLI Usage
---------
    # Train with deployment models
    $ python ALTitude_cli.py train data.tsv --refit-all-data

    # Make predictions
    $ python ALTitude_cli.py predict new_data.tsv --model-dir ./results/

Output Files
------------
The pipeline generates several output files in the output directory:

    - model_summary.csv: Performance summary for all models
    - class_distribution.json: Train/test class proportions
    - predictions/: OOF and test set predictions
    - images/: Visualization plots (ROC, PR curves, etc.)
    - *.pkl: Serialized model files

Dependencies
------------
Required:
    - numpy, pandas, scikit-learn
    - xgboost, lightgbm, catboost
    - torch (for seed management)

Optional:
    - tabpfn_extensions (TabPFN classifier)

Author: ALTitude Development Team
Version: 2.0
"""

import gc
import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    confusion_matrix, recall_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut,
    cross_val_predict, cross_val_score, cross_validate
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# Try to import TabPFN - it's optional
try:
    from tabpfn_extensions import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.info("TabPFN not available - will skip")

from altitude_feature_importance import integrate_with_ml_pipeline_configurable


@dataclass
class Config:
    """Configuration dataclass for the ALTitude ML pipeline.

    This class holds all configuration parameters for training, evaluation,
    and prediction. It supports serialization to/from JSON files for
    reproducibility and experiment tracking.

    Attributes
    ----------

    random_seed : int, default=32
        Random seed for reproducibility across all libraries (numpy, torch, sklearn).

    test_size : float, default=0.3
        Proportion of data to hold out for testing (0.0 to 1.0).
        The remaining data is used for training and cross-validation.

    cv_folds : int, default=5
        Number of folds for k-fold cross-validation.
        Ignored if use_loo_cv is True.

    n_top_models : int, default=5
        Number of top-performing base models to select for the stacking ensemble.
        Models are ranked by CV ROC-AUC score.

    correlation_threshold : float, default=0.9
        Threshold for feature correlation filtering (currently unused).

    n_bootstraps : int, default=1000
        Number of bootstrap iterations for confidence intervals (currently unused).

    hyperparam_iterations : int, default=10
        Number of iterations for hyperparameter optimization (currently unused).

    use_loo_cv : bool, default=False
        If True, use Leave-One-Out cross-validation instead of k-fold CV.
        Recommended for small datasets (< 500 samples) for more robust estimates.

    loo_max_samples : int, default=500
        Maximum number of samples for LOO CV. If dataset exceeds this,
        falls back to k-fold CV to avoid excessive computation time.

    use_feature_selection : bool, default=False
        If True, perform feature selection before training (currently unused).

    n_features_to_select : int, default=20
        Number of features to select if feature selection is enabled.

    data_path : str, default=''
        Path to the training data file (tab-separated format).

    unseen_path : str, default=''
        Path to unseen/external validation data (optional).

    output_dir : Path, default=Path("./results")
        Directory for saving all output files (models, predictions, plots).

    output_dir_image : Path, default=Path("./results/images")
        Directory for saving visualization plots.

    tabpfn_max_time : int, default=40
        Maximum time (seconds) for TabPFN model inference.

    refit_final_on_all_data : bool, default=False
        If True, refit all models on 100% of data (train + test) after
        evaluation for deployment. This creates separate "deployment models"
        while preserving unbiased test metrics from evaluation models.

    Example
    -------
    >>> # Create config programmatically
    >>> config = Config(
    ...     data_path="data/training.tsv",
    ...     test_size=0.3,
    ...     cv_folds=5,
    ...     refit_final_on_all_data=True
    ... )
    >>>
    >>> # Save to file
    >>> config.to_file("config.json")
    >>>
    >>> # Load from file
    >>> config = Config.from_file("config.json")
    """

    random_seed: int = 32
    test_size: float = 0.3
    cv_folds: int = 5
    n_top_models: int = 5
    correlation_threshold: float = 0.9
    n_bootstraps: int = 1000
    hyperparam_iterations: int = 10

    # LOO CV parameters (replacing nested CV)
    use_loo_cv: bool = False
    loo_max_samples: int = 500

    # Feature selection (not used in simplified version but accepted for compatibility)
    use_feature_selection: bool = False
    n_features_to_select: int = 20

    # File paths
    data_path: str = ''
    unseen_path: str = ''
    output_dir: Path = Path("./results")
    output_dir_image: Path = Path("./results/images")

    # TabPFN settings
    tabpfn_max_time: int = 40

    # Final model refitting - refit on ALL data (train+test) for deployment
    refit_final_on_all_data: bool = False

    def __post_init__(self):
        """Initialize paths and create output directories."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_image = Path(self.output_dir_image)
        self.output_dir_image.mkdir(exist_ok=True, parents=True)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file.

        Parameters
        ----------
        config_path : str
            Path to the JSON configuration file.

        Returns
        -------
        Config
            A new Config instance with values loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the config file doesn't exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_file(self, config_path: str):
        """Save configuration to a JSON file.

        Parameters
        ----------
        config_path : str
            Path where the configuration file will be saved.

        Notes
        -----
        Path objects are automatically converted to strings for JSON serialization.
        """
        config_dict = asdict(self)
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class RandomSeedManager:
    """Manages random seed setting across all libraries for reproducibility.

    This class ensures deterministic behavior across numpy, PyTorch, and Python's
    random module. Critical for reproducible machine learning experiments.

    Example
    -------
    >>> RandomSeedManager.set_seeds(42)
    >>> # All subsequent random operations will be deterministic
    """

    @staticmethod
    def set_seeds(seed: int):
        """Set random seeds across all libraries.

        Parameters
        ----------
        seed : int
            The seed value to use for all random number generators.

        Notes
        -----
        Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and all CUDA devices)
        - PYTHONHASHSEED environment variable
        - cuDNN deterministic mode
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


class DataLoader:
    """Handles data loading and preprocessing with class balance verification"""

    def __init__(self, config: Config):
        self.config = config
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.feature_names = None
        self.train_indices = None
        self.test_indices = None
        self.imputer_fitted = False  # Track if imputer is fitted

    def load_and_split_data(self) -> Tuple:
        """Load data and perform train-test split with stratification and proportion reporting"""
        logging.info("Loading and splitting data...")

        # Load data
        df = pd.read_csv(self.config.data_path, sep="\t")
        logging.info(f"Loaded data: {df.shape}")

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
            logging.info(f"Data contains {missing_count} missing values ({missing_pct:.1f}%)")

        # Separate features and target
        y = df['ALT']
        X = df.drop(columns=['ALT', 'ID', 'ID_BUCKET', 'Achilles', 'order',
                             'StripCI', 'internalID', 'Source', 'SJUID', "PG_ATRX","TVR", "Lineage", "Age","DG","SMARCAL1","Adjusted Telomere Content","Human vs Mouse","TERT_exp2",
                             "C-circle Content", "CC Status", "Common Mutations", "Diagnosis", 'ATRX_DAXX',
                             "LoHFraction",   "WGD",   "CIN",   "Ploidy", "Aneuploidy","Sample_Name","LENGTH_ESTIMATE","Telomeric_Reads",
                             "TERRA_RPM" ,"Telomeric_Fraction", "Lineage", "Age","DG", "SMARCAL1_dep",  "MSI", "CIN2"], errors='ignore')

        self.feature_names = list(X.columns)

        # Print overall class distribution
        logging.info("\n" + "=" * 60)
        logging.info("OVERALL CLASS DISTRIBUTION")
        logging.info("=" * 60)
        total_samples = len(y)
        class_counts = y.value_counts().sort_index()

        for class_val, count in class_counts.items():
            proportion = count / total_samples
            logging.info(f"  Class {class_val}: {count:,} samples ({proportion:.2%})")

        # Calculate class ratio
        if len(class_counts) == 2:
            ratio = class_counts.iloc[1] / class_counts.iloc[0]
            logging.info(f"  Class ratio (1:0): {ratio:.3f}")

        # Split data with stratification to maintain class proportions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            stratify=y,  # This ensures equal proportions in train/test
            random_state=self.config.random_seed
        )

        self.train_indices = X_train.index
        self.test_indices = X_test.index

        # Print training set class distribution
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING SET CLASS DISTRIBUTION")
        logging.info("=" * 60)
        train_samples = len(y_train)
        train_counts = y_train.value_counts().sort_index()

        for class_val, count in train_counts.items():
            proportion = count / train_samples
            logging.info(f"  Class {class_val}: {count:,} samples ({proportion:.2%})")

        if len(train_counts) == 2:
            train_ratio = train_counts.iloc[1] / train_counts.iloc[0]
            logging.info(f"  Class ratio (1:0): {train_ratio:.3f}")

        # Print test set class distribution
        logging.info("\n" + "=" * 60)
        logging.info("TEST SET CLASS DISTRIBUTION")
        logging.info("=" * 60)
        test_samples = len(y_test)
        test_counts = y_test.value_counts().sort_index()

        for class_val, count in test_counts.items():
            proportion = count / test_samples
            logging.info(f"  Class {class_val}: {count:,} samples ({proportion:.2%})")

        if len(test_counts) == 2:
            test_ratio = test_counts.iloc[1] / test_counts.iloc[0]
            logging.info(f"  Class ratio (1:0): {test_ratio:.3f}")

        # Verify stratification worked correctly
        logging.info("\n" + "=" * 60)
        logging.info("STRATIFICATION VERIFICATION")
        logging.info("=" * 60)

        # Check if proportions are similar (within 1% tolerance)
        stratification_success = True
        for class_val in class_counts.index:
            overall_prop = class_counts[class_val] / total_samples
            train_prop = train_counts[class_val] / train_samples
            test_prop = test_counts[class_val] / test_samples

            train_diff = abs(train_prop - overall_prop) * 100
            test_diff = abs(test_prop - overall_prop) * 100

            logging.info(f"  Class {class_val}:")
            logging.info(f"    Overall: {overall_prop:.2%}")
            logging.info(f"    Train:   {train_prop:.2%} (diff: {train_diff:.2f}%)")
            logging.info(f"    Test:    {test_prop:.2%} (diff: {test_diff:.2f}%)")

            if train_diff > 1.0 or test_diff > 1.0:
                stratification_success = False
                logging.warning(f"    ⚠️ Proportion difference exceeds 1% threshold!")

        if stratification_success:
            logging.info("\n✅ Stratification successful - proportions maintained across splits")
        else:
            logging.warning("\n⚠️ Stratification may not be optimal - check class distributions")

        logging.info("=" * 60 + "\n")

        # Create both imputed and raw versions
        # Imputed version for models that can't handle NaN
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        self.imputer_fitted = True  # Mark as fitted

        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")
        logging.info(f"Features: {len(self.feature_names)}")

        # Save class distribution info to config output directory
        distribution_report = {
            'overall': {
                'total_samples': int(total_samples),
                'class_counts': {int(k): int(v) for k, v in class_counts.items()},
                'class_proportions': {int(k): float(v / total_samples) for k, v in class_counts.items()}
            },
            'training': {
                'total_samples': int(train_samples),
                'class_counts': {int(k): int(v) for k, v in train_counts.items()},
                'class_proportions': {int(k): float(v / train_samples) for k, v in train_counts.items()}
            },
            'test': {
                'total_samples': int(test_samples),
                'class_counts': {int(k): int(v) for k, v in test_counts.items()},
                'class_proportions': {int(k): float(v / test_samples) for k, v in test_counts.items()}
            }
        }

        # Save distribution report
        distribution_file = self.config.output_dir / 'class_distribution.json'
        with open(distribution_file, 'w') as f:
            json.dump(distribution_report, f, indent=2)
        logging.info(f"Class distribution report saved to: {distribution_file}")

        return (X_train, X_test, X_train_imputed, X_test_imputed,
                y_train, y_test)

class ModelFactory:
    """Factory for creating models"""

    @staticmethod
    def create_base_models(random_seed: int) -> List[Tuple[str, Any]]:
        """Create base models"""
        models = [
            # Models that handle NaN natively
            ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=random_seed, 
                                      use_label_encoder=False)),
            ('CatBoost', CatBoostClassifier(random_state=random_seed, verbose=False)),
            ('LightGBM', LGBMClassifier(random_state=random_seed, verbose=-1)),
            ('HistGB', HistGradientBoostingClassifier(random_state=random_seed)),
            
            # Models that need imputed data
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=random_seed)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=random_seed)),
            ('ExtraTrees', ExtraTreesClassifier(n_estimators=100, random_state=random_seed)),
            ('AdaBoost', AdaBoostClassifier(n_estimators=50, random_state=random_seed)),
        ]
        
        # Add TabPFN if available
        if TABPFN_AVAILABLE:
            models.append(('TabPFN', TabPFNClassifier(device='cpu', random_state=random_seed)))
        
        return models

    @staticmethod
    def create_meta_learners(random_seed: int) -> Dict[str, Any]:
        """Create meta-learners for stacking"""
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_seed),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_seed),
            'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', 
                                     random_state=random_seed, use_label_encoder=False),
        }


class MetricsCalculator:
    """Calculate metrics"""

    @staticmethod
    def calculate_metrics(y_true, y_probs, threshold=0.5):
        """Calculate classification metrics"""
        y_pred = (y_probs >= threshold).astype(int)
        
        try:
            auc_score = roc_auc_score(y_true, y_probs)
        except:
            auc_score = 0.5
            
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        except:
            sensitivity = 0
            specificity = 0
            
        try:
            avg_precision = average_precision_score(y_true, y_probs)
        except:
            avg_precision = y_true.mean()
            
        return {
            'auc': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'avg_precision': avg_precision
        }


class ModelTrainer:
    """Train models with LOO or k-fold cross-validation"""

    def __init__(self, config: Config):
        self.config = config
        # Choose cross-validation strategy
        if config.use_loo_cv:
            self.cv = None  # Will be set dynamically based on sample size
            self.cv_type = "LOO"
        else:
            self.cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, 
                                     random_state=config.random_seed)
            self.cv_type = f"{config.cv_folds}-fold"

    def get_cv_strategy(self, n_samples: int, y: np.ndarray):
        """Get appropriate CV strategy based on sample size and config"""
        if self.config.use_loo_cv:
            if n_samples > self.config.loo_max_samples:
                logging.warning(f"Dataset has {n_samples} samples, which exceeds LOO max of {self.config.loo_max_samples}")
                logging.warning(f"Using {self.config.cv_folds}-fold CV instead for computational efficiency")
                return StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                                      random_state=self.config.random_seed)
            else:
                logging.info(f"Using Leave-One-Out CV for {n_samples} samples")
                return LeaveOneOut()
        else:
            return self.cv

    def train_model_with_loo(self, model, model_name, X_use, y_train):
        """Train model using Leave-One-Out cross-validation"""
        n_samples = len(y_train)
        loo = LeaveOneOut()
        
        # Initialize arrays for predictions
        oof_preds = np.zeros(n_samples)
        
        logging.info(f"Running LOO CV for {model_name} ({n_samples} iterations)...")
        
        # Track progress
        for i, (train_idx, val_idx) in enumerate(loo.split(X_use)):
            if i % 50 == 0:
                logging.info(f"  LOO iteration {i}/{n_samples}")
            
            X_train_fold = X_use.iloc[train_idx]
            X_val_fold = X_use.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            
            # Clone and train model
            fold_model = deepcopy(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            if hasattr(fold_model, 'predict_proba'):
                pred = fold_model.predict_proba(X_val_fold)[:, 1]
            else:
                pred = fold_model.decision_function(X_val_fold)
                # Normalize to [0, 1]
                if len(np.unique(pred)) > 1:
                    pred = (pred - pred.min()) / (pred.max() - pred.min())
                else:
                    pred = np.array([0.5])
            
            oof_preds[val_idx[0]] = pred[0]
        
        # Calculate LOO AUC
        try:
            loo_auc = roc_auc_score(y_train, oof_preds)
        except:
            loo_auc = 0.5
        
        return oof_preds, loo_auc

    def train_model(self, model, model_name, X_train, X_train_imputed, y_train):
        """Train a single model with appropriate data and CV strategy"""
        
        # Determine which data to use
        nan_native_models = {'XGBoost', 'CatBoost', 'LightGBM', 'HistGB'}
        
        if model_name in nan_native_models:
            # Use raw data with NaN
            X_use = X_train
            logging.info(f"Training {model_name} with raw data (handles NaN)")
        else:
            # Use imputed data
            X_use = X_train_imputed
            logging.info(f"Training {model_name} with imputed data")
        
        try:
            # Get appropriate CV strategy
            cv_strategy = self.get_cv_strategy(len(y_train), y_train)
            
            if isinstance(cv_strategy, LeaveOneOut):
                # Use LOO cross-validation
                oof_preds, cv_score = self.train_model_with_loo(model, model_name, X_use, y_train)
                cv_scores = np.array([cv_score])  # Single score for LOO
                
                logging.info(f"  LOO CV complete. AUC: {cv_score:.4f}")
            else:
                # Use k-fold cross-validation
                cv_scores = cross_val_score(model, X_use, y_train, cv=cv_strategy, 
                                          scoring='roc_auc', n_jobs=-1)
                
                # Get OOF predictions
                if hasattr(model, 'predict_proba'):
                    oof_preds = cross_val_predict(model, X_use, y_train, cv=cv_strategy,
                                                 method='predict_proba', n_jobs=-1)[:, 1]
                else:
                    # For models without predict_proba
                    oof_preds = cross_val_predict(model, X_use, y_train, cv=cv_strategy,
                                                 method='decision_function', n_jobs=-1)
                    # Normalize to [0, 1]
                    oof_preds = (oof_preds - oof_preds.min()) / (oof_preds.max() - oof_preds.min())
                
                logging.info(f"  {self.config.cv_folds}-fold CV complete. Mean AUC: {cv_scores.mean():.4f}")
            
            # Train final model on all data
            final_model = deepcopy(model)
            final_model.fit(X_use, y_train)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(y_train, oof_preds)
            
            return {
                'model': final_model,
                'cv_scores': cv_scores,
                'oof_preds': oof_preds,
                'metrics': metrics,
                'success': True,
                'cv_type': 'LOO' if isinstance(cv_strategy, LeaveOneOut) else f'{self.config.cv_folds}-fold'
            }
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            # Return dummy model
            dummy = DummyClassifier(strategy='prior')
            dummy.fit(X_train_imputed, y_train)
            dummy_preds = np.full(len(y_train), y_train.mean())
            
            return {
                'model': dummy,
                'cv_scores': np.array([0.5]),
                'oof_preds': dummy_preds,
                'metrics': {'auc': 0.5, 'sensitivity': 0, 'specificity': 0, 'avg_precision': y_train.mean()},
                'success': False,
                'cv_type': 'failed'
            }


class MLPipeline:
    """Main pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.trainer = ModelTrainer(config)
        self.results = {}

    def train(self):
        """Train all models with prediction logging"""
        logging.info("Starting training pipeline...")
        logging.info(
            f"Cross-validation strategy: {'Leave-One-Out' if self.config.use_loo_cv else f'{self.config.cv_folds}-fold'}")
        RandomSeedManager.set_seeds(self.config.random_seed)

        # Initialize prediction logger
        from altitude_prediction_logger import PredictionLogger
        pred_logger = PredictionLogger(self.config.output_dir)

        # Load data
        (X_train, X_test, X_train_imputed, X_test_imputed,
         y_train, y_test) = self.data_loader.load_and_split_data()

        # Check if LOO is appropriate for this dataset size
        if self.config.use_loo_cv and len(y_train) > self.config.loo_max_samples:
            logging.warning(
                f"Training set has {len(y_train)} samples, exceeding LOO max of {self.config.loo_max_samples}")
            logging.warning("Consider using k-fold CV for better computational efficiency")

        # Create models
        base_models = ModelFactory.create_base_models(self.config.random_seed)

        # Train each model
        trained_models = []
        all_results = []

        for model_name, model in base_models:
            result = self.trainer.train_model(
                model, model_name, X_train, X_train_imputed, y_train
            )

            if result['success']:
                trained_models.append((model_name, result['model']))
                all_results.append({
                    'name': model_name,
                    'auc': result['metrics']['auc'],
                    'cv_scores': result['cv_scores'],
                    'oof_preds': result['oof_preds'],
                    'model': result['model'],
                    'cv_type': result['cv_type']
                })

                # Log OOF predictions
                pred_logger.log_oof_predictions(
                    model_name=model_name,
                    predictions=result['oof_preds'],
                    y_true=y_train,
                    indices=self.data_loader.train_indices,
                    cv_type=result['cv_type']
                )

                logging.info(f"✅ {model_name}: AUC = {result['metrics']['auc']:.4f} ({result['cv_type']} CV)")
            else:
                logging.warning(f"⚠️ {model_name} failed to train properly")

        # Select best models for ensemble
        all_results.sort(key=lambda x: x['auc'], reverse=True)
        n_select = min(self.config.n_top_models, len(all_results))
        selected = all_results[:n_select]

        logging.info(f"\nSelected {n_select} models for ensemble:")
        for item in selected:
            logging.info(f"  - {item['name']}: AUC = {item['auc']:.4f}")

        # Initialize meta_results and fitted_meta_learners
        meta_results = {}
        fitted_meta_learners = {}
        meta_cv_type = None  # Will be set if ensemble is created

        # Create ensemble if we have enough models
        if len(selected) >= 2:
            # Stack predictions
            stack_train = np.column_stack([item['oof_preds'] for item in selected])

            # Train meta-learners using same CV strategy as base models
            meta_learners = ModelFactory.create_meta_learners(self.config.random_seed)

            # Determine CV strategy for meta-learners (match base model CV)
            use_loo_for_meta = self.config.use_loo_cv and len(y_train) <= self.config.loo_max_samples
            meta_cv_type = 'LOO' if use_loo_for_meta else f'{min(5, len(y_train))}-fold'

            if use_loo_for_meta:
                logging.info("Using LOO CV for meta-learners (consistent with base models)")
            else:
                logging.info(f"Using {min(5, len(y_train))}-fold CV for meta-learners")

            for name, learner in meta_learners.items():
                try:
                    if use_loo_for_meta:
                        # LOO CV for meta-learners
                        n_samples = len(y_train)
                        meta_preds = np.zeros(n_samples)
                        loo = LeaveOneOut()

                        logging.info(f"Running LOO CV for meta-learner {name} ({n_samples} iterations)...")
                        for i, (train_idx, val_idx) in enumerate(loo.split(stack_train)):
                            if i % 50 == 0:
                                logging.info(f"  LOO iteration {i}/{n_samples}")

                            # Train on N-1, predict on 1
                            fold_learner = deepcopy(learner)
                            fold_learner.fit(stack_train[train_idx], y_train.iloc[train_idx])
                            meta_preds[val_idx[0]] = fold_learner.predict_proba(stack_train[val_idx])[:, 1][0]

                        # Fit final meta-learner on all data for test set evaluation
                        learner.fit(stack_train, y_train)
                    else:
                        # k-fold CV for meta-learners
                        meta_cv = StratifiedKFold(n_splits=min(5, len(y_train)), shuffle=True,
                                                  random_state=self.config.random_seed)

                        # Get OOF predictions via proper CV (train on fold, predict on held-out)
                        meta_preds = cross_val_predict(learner, stack_train, y_train,
                                                       cv=meta_cv, method='predict_proba')[:, 1]

                        # Fit final meta-learner on all data for test set evaluation
                        learner.fit(stack_train, y_train)

                    meta_metrics = MetricsCalculator.calculate_metrics(y_train, meta_preds)
                    meta_results[name] = {
                        'model': learner,
                        'metrics': meta_metrics,
                        'oof_preds': meta_preds
                    }
                    fitted_meta_learners[name] = learner

                    # Log meta-learner OOF predictions
                    pred_logger.log_oof_predictions(
                        model_name=f'SE_{name}',
                        predictions=meta_preds,
                        y_true=y_train,
                        indices=self.data_loader.train_indices,
                        cv_type=meta_cv_type
                    )

                    logging.info(f"Meta-learner {name}: AUC = {meta_metrics['auc']:.4f} ({meta_cv_type} CV)")
                except Exception as e:
                    logging.error(f"Meta-learner {name} failed: {e}")

        # TEST SET EVALUATION
        logging.info("\nEvaluating on test set...")
        test_results = {}
        roc_data = {}
        pr_data = {}
        test_predictions = {}

        nan_native_models = {'XGBoost', 'CatBoost', 'LightGBM', 'HistGB'}

        # Evaluate base models on test set
        for item in all_results:
            model_name = item['name']
            model = item['model']

            # Use appropriate test data
            if model_name in nan_native_models:
                X_test_use = X_test
            else:
                X_test_use = X_test_imputed

            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_probs = model.predict_proba(X_test_use)[:, 1]
                else:
                    y_scores = model.decision_function(X_test_use)
                    y_probs = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

                # Calculate metrics
                test_metrics = MetricsCalculator.calculate_metrics(y_test, y_probs)
                test_results[model_name] = test_metrics
                test_predictions[f'{model_name}_prob'] = y_probs

                # Log test predictions
                pred_logger.log_test_predictions(
                    model_name=model_name,
                    predictions=y_probs,
                    y_true=y_test,
                    indices=self.data_loader.test_indices,
                    threshold=0.5
                )

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_data[model_name] = (fpr, tpr)

                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_probs)
                pr_data[model_name] = (precision, recall, test_metrics['avg_precision'])

                logging.info(f"Test - {model_name}: AUC = {test_metrics['auc']:.4f}")

            except Exception as e:
                logging.error(f"Test evaluation failed for {model_name}: {e}")

        # Evaluate meta-learners on test set if they exist
        if fitted_meta_learners and len(selected) >= 2:
            # Create test stack
            stack_test = np.zeros((len(X_test), len(selected)))
            for i, item in enumerate(selected):
                model_name = item['name']
                model = item['model']

                if model_name in nan_native_models:
                    X_test_use = X_test
                else:
                    X_test_use = X_test_imputed

                if hasattr(model, 'predict_proba'):
                    stack_test[:, i] = model.predict_proba(X_test_use)[:, 1]
                else:
                    y_scores = model.decision_function(X_test_use)
                    stack_test[:, i] = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

            # Evaluate meta-learners
            for name, learner in fitted_meta_learners.items():
                try:
                    y_probs = learner.predict_proba(stack_test)[:, 1]
                    test_metrics = MetricsCalculator.calculate_metrics(y_test, y_probs)
                    test_results[f'SE_{name}'] = test_metrics
                    test_predictions[f'SE_{name}_prob'] = y_probs

                    # Log meta-learner test predictions
                    pred_logger.log_test_predictions(
                        model_name=f'SE_{name}',
                        predictions=y_probs,
                        y_true=y_test,
                        indices=self.data_loader.test_indices,
                        threshold=0.5
                    )

                    # ROC and PR curves
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_data[f'SE_{name}'] = (fpr, tpr)

                    precision, recall, _ = precision_recall_curve(y_test, y_probs)
                    pr_data[f'SE_{name}'] = (precision, recall, test_metrics['avg_precision'])

                    logging.info(f"Test - SE_{name}: AUC = {test_metrics['auc']:.4f}")

                except Exception as e:
                    logging.error(f"Test evaluation failed for SE_{name}: {e}")

        # REFIT ON ALL DATA FOR DEPLOYMENT (if enabled)
        deployment_models = []
        deployment_meta_learners = {}
        deployment_imputer = None

        if self.config.refit_final_on_all_data:
            logging.info("\n" + "=" * 60)
            logging.info("REFITTING MODELS ON ALL DATA FOR DEPLOYMENT")
            logging.info("=" * 60)
            logging.info("Note: Test metrics above are from models trained on training set only.")
            logging.info("Deployment models below are trained on 100% of data (train + test).")

            # Combine train and test data
            X_all = pd.concat([X_train, X_test], axis=0)
            X_all_imputed = pd.concat([X_train_imputed, X_test_imputed], axis=0)
            y_all = pd.concat([y_train, y_test], axis=0)

            logging.info(f"Combined dataset: {len(y_all)} samples (train: {len(y_train)} + test: {len(y_test)})")

            # Refit imputer on ALL data for deployment
            deployment_imputer = SimpleImputer(strategy='mean')
            X_all_imputed_new = pd.DataFrame(
                deployment_imputer.fit_transform(X_all),
                columns=X_all.columns,
                index=X_all.index
            )
            logging.info("Refitted imputer on all data")

            # Refit base models on all data. Ensemble members go through
            # ModelTrainer so they also return out-of-fold predictions on all
            # data, which the deployment meta-learners are trained on (as the
            # evaluated meta-learners were).
            trained_by_name = dict(trained_models)
            ensemble_members = [(item['name'], trained_by_name[item['name']])
                                for item in selected if item['name'] in trained_by_name]
            stack_all, refitted_members = self._deployment_oof_stack(
                ensemble_members, X_all, X_all_imputed_new, y_all)
            refitted_members = dict(refitted_members)

            for model_name, model in trained_models:
                if model_name in refitted_members:
                    deployment_models.append((model_name, refitted_members[model_name]))
                    continue
                try:
                    # Use appropriate data
                    if model_name in nan_native_models:
                        X_use = X_all
                    else:
                        X_use = X_all_imputed_new

                    # Create fresh model and fit on all data
                    deployment_model = deepcopy(model)
                    deployment_model.fit(X_use, y_all)
                    deployment_models.append((model_name, deployment_model))
                    logging.info(f"  ✅ {model_name}: refitted on all {len(y_all)} samples")

                except Exception as e:
                    logging.error(f"  ❌ Failed to refit {model_name}: {e}")

            # Refit meta-learners on all data if we have enough models
            if len(selected) >= 2 and fitted_meta_learners:
                logging.info("\nRefitting meta-learners on out-of-fold predictions from all data...")

                # Refit each meta-learner
                for name in fitted_meta_learners.keys():
                    try:
                        # Create fresh meta-learner
                        fresh_meta_learners = ModelFactory.create_meta_learners(self.config.random_seed)
                        deploy_meta = fresh_meta_learners[name]
                        deploy_meta.fit(stack_all, y_all)
                        deployment_meta_learners[name] = deploy_meta
                        logging.info(f"  ✅ SE_{name}: refitted on all {len(y_all)} samples")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to refit SE_{name}: {e}")

            logging.info("\n" + "=" * 60)
            logging.info("DEPLOYMENT MODEL REFITTING COMPLETE")
            logging.info(f"  Base models: {len(deployment_models)}")
            logging.info(f"  Meta-learners: {len(deployment_meta_learners)}")
            logging.info("=" * 60 + "\n")

        # SAVE ALL PREDICTIONS TO FILES
        logging.info("\n" + "=" * 60)
        logging.info("SAVING PREDICTIONS TO FILES")
        logging.info("=" * 60)

        saved_prediction_files = pred_logger.save_predictions()

        logging.info("\n📊 Prediction files saved:")
        for file_type, file_path in saved_prediction_files.items():
            if file_path:
                logging.info(f"  {file_type}: {file_path.name}")

        # Create summary DataFrame
        summary_df = pd.DataFrame([
            {
                'Model': item['name'],
                'Mean ROC AUC': item['auc'],
                'Std': np.std(item['cv_scores']) if len(item['cv_scores']) > 1 else 0.0,
                'CV Type': item['cv_type']
            }
            for item in all_results
        ])

        # Add meta-learner results to summary
        for name, result in meta_results.items():
            summary_df = pd.concat([summary_df, pd.DataFrame([{
                'Model': f'SE_{name}',
                'Mean ROC AUC': result['metrics']['auc'],
                'Std': 0.0,
                'CV Type': meta_cv_type if meta_cv_type else 'k-fold'
            }])], ignore_index=True)

        # Save summary
        summary_df.to_csv(self.config.output_dir / 'model_summary.csv', index=False)

        logging.info("Training complete!")

        # Prepare OOF predictions for visualization
        oof_predictions = {}
        for item in all_results:
            oof_predictions[f"{item['name']}_prob"] = item['oof_preds']
        for name, result in meta_results.items():
            oof_predictions[f"SE_{name}_prob"] = result['oof_preds']

        # COMPLETE RETURN WITH ALL VISUALIZATION DATA AND IMPUTER
        return {
            'fitted_models': trained_models,
            'fitted_meta_learners': fitted_meta_learners,
            'selected_models': [item['name'] for item in selected],
            'selected_indices': list(range(len(selected))),
            'data_imputer': self.data_loader.imputer,  # IMPORTANT: Save the fitted imputer
            'data_scaler': self.data_loader.scaler,  # Keep for compatibility
            'meta_scaler': StandardScaler(),  # Not used but kept
            'summary': summary_df,
            'best_model_name': summary_df.loc[
                summary_df['Mean ROC AUC'].idxmax(), 'Model'] if not summary_df.empty else 'None',
            'best_model_auc': summary_df['Mean ROC AUC'].max() if not summary_df.empty else 0.0,
            'all_results': all_results,
            'meta_results': meta_results,
            'X_test': X_test,
            'X_test_imputed': X_test_imputed,
            'y_test': y_test,
            'test_results': test_results,
            'base_results': {
                'roc_data': roc_data,
                'pr_data': pr_data,
                'oof_predictions': oof_predictions,
                'test_predictions': test_predictions
            },
            'saved_predictions': saved_prediction_files,
            # Deployment models (trained on ALL data) - only present if refit_final_on_all_data=True
            'deployment_models': deployment_models if deployment_models else None,
            'deployment_meta_learners': deployment_meta_learners if deployment_meta_learners else None,
            'deployment_imputer': deployment_imputer,
            'refit_on_all_data': self.config.refit_final_on_all_data
        }

    def _deployment_oof_stack(self, models: List[Tuple[str, Any]], X_all: pd.DataFrame,
                              X_all_imputed: pd.DataFrame, y_all: pd.Series):
        """Refit ensemble members on all data and return their out-of-fold predictions.

        Uses the same CV strategy as training (ModelTrainer), so the deployment
        meta-learners are trained on OOF predictions, like the evaluated
        meta-learners, rather than on predictions from models fitted on the same
        samples.

        Returns:
            stack: array (n_samples, n_models), columns in the order of `models`
                (0.5 for a model that failed to refit)
            refitted: list of (name, model fitted on all data)
        """
        stack = np.full((len(y_all), len(models)), 0.5)
        refitted = []

        for i, (model_name, model) in enumerate(models):
            result = self.trainer.train_model(model, model_name, X_all, X_all_imputed, y_all)
            if not result['success']:
                logging.error(f"  ❌ Failed to refit {model_name}; its stack column is set to 0.5")
                continue
            stack[:, i] = result['oof_preds']
            refitted.append((model_name, result['model']))
            logging.info(f"  ✅ {model_name}: refitted on all {len(y_all)} samples "
                         f"({result['cv_type']} OOF AUC = {result['metrics']['auc']:.4f})")

        return stack, refitted

    def load_state(self, state: Dict[str, Any], use_deployment_models: bool = True):
        """Load a saved training state into the pipeline

        Args:
            state: Dictionary containing the saved training state
            use_deployment_models: If True and deployment models exist (trained on 100% data),
                                  use them for predictions. If False, use the evaluation models
                                  (trained on 70% training data). Default: True
        """
        # Check if deployment models are available
        has_deployment = (state.get('deployment_models') is not None and
                         len(state.get('deployment_models', [])) > 0)

        if use_deployment_models and has_deployment:
            # Use deployment models (trained on ALL data)
            self.fitted_models = state.get('deployment_models', [])
            self.fitted_meta_learners = state.get('deployment_meta_learners', {})
            self.using_deployment_models = True
            logging.info("=" * 60)
            logging.info("LOADING DEPLOYMENT MODELS (trained on 100% of data)")
            logging.info("=" * 60)
        else:
            # Use evaluation models (trained on training set only)
            self.fitted_models = state.get('fitted_models', [])
            self.fitted_meta_learners = state.get('fitted_meta_learners', {})
            self.using_deployment_models = False
            if use_deployment_models and not has_deployment:
                logging.warning("Deployment models not available - using evaluation models")
                logging.warning("To get deployment models, retrain with refit_final_on_all_data=True")
            logging.info("Loading evaluation models (trained on 70% training data)")

        self.selected_models = state.get('selected_models', [])
        self.selected_indices = state.get('selected_indices', [])

        # Load the appropriate imputer
        if use_deployment_models and has_deployment and state.get('deployment_imputer') is not None:
            self.data_loader.imputer = state['deployment_imputer']
            self.data_loader.imputer_fitted = True
            logging.info("Loaded deployment imputer (fitted on all data)")
        elif 'data_imputer' in state and state['data_imputer'] is not None:
            self.data_loader.imputer = state['data_imputer']
            self.data_loader.imputer_fitted = True
            logging.info("Loaded fitted imputer")
        elif 'data_scaler' in state and isinstance(state['data_scaler'], SimpleImputer):
            # Old models might have saved imputer as data_scaler
            self.data_loader.imputer = state['data_scaler']
            self.data_loader.imputer_fitted = True
            logging.warning("Loaded imputer from data_scaler (old format)")
        else:
            # No fitted imputer available - create new but mark as not fitted
            logging.error("No fitted imputer found! Will need to fit on prediction data (NOT IDEAL)")
            logging.error("This may lead to different results than during training!")
            self.data_loader.imputer = SimpleImputer(strategy='mean')
            self.data_loader.imputer_fitted = False  # Mark as not fitted

        self.data_loader.scaler = state.get('meta_scaler', StandardScaler())
        self.model_summary = state.get('model_summary')

        # Store the full state for reference
        self.state = state

        logging.info(f"Loaded {len(self.fitted_models)} base models")
        logging.info(f"Loaded {len(self.fitted_meta_learners)} meta-learners")
        if self.using_deployment_models:
            logging.info("These models were trained on ALL available data (train + test)")
        else:
            logging.info("These models were trained on the training set only (for evaluation)")

    def predict(self, data_path: str, threshold: float = 0.5) -> pd.DataFrame:
        """Make predictions on new data using loaded models"""
        if not hasattr(self, 'fitted_models') or not self.fitted_models:
            raise RuntimeError("No models loaded. Use load_state() first or train models.")
        
        # Load new data
        df = pd.read_csv(data_path, sep="\t")
        
        # Keep original data for output
        original_df = df.copy()
        
        # Process features (same as training)
        X_new = df.drop(columns=['ALT', 'ID', 'ID_BUCKET', 'Achilles', 'order',
                             'StripCI', 'internalID', 'Source', 'SJUID', "PG_ATRX","TVR", "Lineage", "Age","DG","SMARCAL1","Adjusted Telomere Content","Human vs Mouse","TERT_exp2",
                             "C-circle Content", "CC Status", "Common Mutations", "Diagnosis", 'ATRX_DAXX',
                             "LoHFraction",   "WGD",   "CIN",   "Ploidy", "Aneuploidy","Sample_Name","LENGTH_ESTIMATE","Telomeric_Reads",
                             "TERRA_RPM" ,"Telomeric_Fraction", "Lineage", "Age","DG", "SMARCAL1_dep",  "MSI", "CIN2"], errors='ignore')
        
        # Handle imputation based on whether we have a fitted imputer
        if not self.data_loader.imputer_fitted:
            logging.warning("=" * 60)
            logging.warning("IMPUTER NOT FITTED - FITTING ON PREDICTION DATA")
            logging.warning("This is NOT recommended and may produce incorrect results!")
            logging.warning("Please retrain your models to save the fitted imputer.")
            logging.warning("=" * 60)
            
            # Fit imputer on the new data (not ideal!)
            X_new_imputed = pd.DataFrame(
                self.data_loader.imputer.fit_transform(X_new),
                columns=X_new.columns,
                index=X_new.index
            )
            self.data_loader.imputer_fitted = True
        else:
            # Use the properly fitted imputer from training
            X_new_imputed = pd.DataFrame(
                self.data_loader.imputer.transform(X_new),
                columns=X_new.columns,
                index=X_new.index
            )
        
        # Make predictions with base models
        predictions = {}
        nan_native_models = {'XGBoost', 'CatBoost', 'LightGBM', 'HistGB'}
        
        logging.info(f"Making predictions with {len(self.fitted_models)} base models...")
        
        for model_name, model in self.fitted_models:
            # Use appropriate data
            if model_name in nan_native_models:
                X_use = X_new
            else:
                X_use = X_new_imputed
            
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_use)[:, 1]
                else:
                    # For models without predict_proba
                    scores = model.decision_function(X_use)
                    preds = (scores - scores.min()) / (scores.max() - scores.min())
                
                predictions[model_name] = preds
                logging.info(f"  ✅ {model_name}: predictions completed")
                
            except Exception as e:
                logging.error(f"  ❌ Prediction failed for {model_name}: {e}")
                predictions[model_name] = np.full(len(X_new), 0.5)
        
        # Make ensemble predictions if we have meta-learners
        if self.fitted_meta_learners and self.selected_models:
            logging.info(f"Making ensemble predictions with {len(self.fitted_meta_learners)} meta-learners...")
            
            # Create stack of selected model predictions
            stack_new = np.zeros((len(X_new), len(self.selected_models)))
            
            for i, model_name in enumerate(self.selected_models):
                # Find the model
                model = None
                for name, m in self.fitted_models:
                    if name == model_name:
                        model = m
                        break
                
                if model is None:
                    logging.warning(f"Selected model {model_name} not found in fitted models")
                    stack_new[:, i] = 0.5
                    continue
                
                # Use appropriate data
                if model_name in nan_native_models:
                    X_use = X_new
                else:
                    X_use = X_new_imputed
                
                try:
                    if hasattr(model, 'predict_proba'):
                        stack_new[:, i] = model.predict_proba(X_use)[:, 1]
                    else:
                        scores = model.decision_function(X_use)
                        stack_new[:, i] = (scores - scores.min()) / (scores.max() - scores.min())
                except Exception as e:
                    logging.error(f"Failed to create stack for {model_name}: {e}")
                    stack_new[:, i] = 0.5
            
            # Get meta-learner predictions
            for name, learner in self.fitted_meta_learners.items():
                try:
                    meta_preds = learner.predict_proba(stack_new)[:, 1]
                    predictions[f'SE_{name}'] = meta_preds
                    logging.info(f"  ✅ SE_{name}: ensemble predictions completed")
                except Exception as e:
                    logging.error(f"  ❌ Ensemble prediction failed for SE_{name}: {e}")
                    predictions[f'SE_{name}'] = np.full(len(X_new), 0.5)
        
        # Create output dataframe
        output_df = original_df.copy()
        
        # Add predictions and binary classifications
        for name, preds in predictions.items():
            output_df[f'{name}_prob'] = preds
            output_df[f'{name}_pred'] = (preds >= threshold).astype(int)
        
        # Add summary statistics
        prob_cols = [col for col in output_df.columns if col.endswith('_prob')]
        if prob_cols:
            output_df['mean_prob'] = output_df[prob_cols].mean(axis=1)
            output_df['median_prob'] = output_df[prob_cols].median(axis=1)
            output_df['max_prob'] = output_df[prob_cols].max(axis=1)
            output_df['min_prob'] = output_df[prob_cols].min(axis=1)
        
        logging.info(f"Predictions complete for {len(output_df)} samples")
        
        return output_df
