"""
ALTitude Core - ML Pipeline with Leave-One-Out Cross-Validation
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
    """Configuration class for the ML pipeline"""
    random_seed: int = 32
    test_size: float = 0.3
    cv_folds: int = 5
    n_top_models: int = 5
    correlation_threshold: float = 0.9
    n_bootstraps: int = 1000
    hyperparam_iterations: int = 10
    
    # LOO CV parameters (replacing nested CV)
    use_loo_cv: bool = False  # Use Leave-One-Out CV instead of k-fold
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

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_image = Path(self.output_dir_image)
        self.output_dir_image.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class RandomSeedManager:
    """Manages random seed setting across all libraries"""

    @staticmethod
    def set_seeds(seed: int):
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

        # Create ensemble if we have enough models
        if len(selected) >= 2:
            # Stack predictions
            stack_train = np.column_stack([item['oof_preds'] for item in selected])

            # Train meta-learners (using regular k-fold for meta-learners to save computation)
            meta_learners = ModelFactory.create_meta_learners(self.config.random_seed)
            meta_cv = StratifiedKFold(n_splits=min(5, len(y_train)), shuffle=True,
                                      random_state=self.config.random_seed)

            for name, learner in meta_learners.items():
                try:
                    learner.fit(stack_train, y_train)
                    meta_preds = cross_val_predict(learner, stack_train, y_train,
                                                   cv=meta_cv, method='predict_proba')[:, 1]
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
                        cv_type='k-fold'
                    )

                    logging.info(f"Meta-learner {name}: AUC = {meta_metrics['auc']:.4f}")
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
                'CV Type': 'k-fold'
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
            'saved_predictions': saved_prediction_files  # Add reference to saved files
        }

    def load_state(self, state: Dict[str, Any]):
        """Load a saved training state into the pipeline"""
        self.fitted_models = state.get('fitted_models', [])
        self.fitted_meta_learners = state.get('fitted_meta_learners', {})
        self.selected_models = state.get('selected_models', [])
        self.selected_indices = state.get('selected_indices', [])
        
        # Load the imputer with proper fallback handling
        if 'data_imputer' in state and state['data_imputer'] is not None:
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
