"""
Prediction Logger for ALTitude Pipeline
Saves LOO/OOF predictions and test predictions to files
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union


class PredictionLogger:
    """Logs and saves all predictions from training and testing"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.predictions_dir = self.output_dir / 'predictions'
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

        # Storage for predictions
        self.oof_predictions = {}  # Out-of-fold (LOO/CV) predictions
        self.test_predictions = {}  # Test set predictions
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'cv_type': None,
            'n_train_samples': 0,
            'n_test_samples': 0,
            'models': []
        }

    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas objects to JSON-serializable formats"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def log_oof_predictions(self, model_name: str, predictions: Union[np.ndarray, pd.Series],
                           y_true: Union[np.ndarray, pd.Series] = None,
                           indices: Union[np.ndarray, pd.Series, pd.Index] = None,
                           cv_type: str = None):
        """
        Log out-of-fold predictions from cross-validation or LOO

        Args:
            model_name: Name of the model
            predictions: Predicted probabilities
            y_true: True labels (optional)
            indices: Original indices in dataset (optional)
            cv_type: Type of CV used (LOO, k-fold, etc.)
        """
        # Convert to numpy arrays for consistency
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(indices, (pd.Series, pd.Index)):
            indices = indices.values

        self.oof_predictions[model_name] = {
            'predictions': self._convert_to_serializable(predictions),
            'n_samples': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions))
        }

        if y_true is not None:
            self.oof_predictions[model_name]['y_true'] = self._convert_to_serializable(y_true)

            # Calculate metrics
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_true, predictions)
                self.oof_predictions[model_name]['auc'] = float(auc)
            except:
                pass

        if indices is not None:
            self.oof_predictions[model_name]['indices'] = self._convert_to_serializable(indices)

        if cv_type:
            self.metadata['cv_type'] = cv_type
            self.oof_predictions[model_name]['cv_type'] = cv_type

        if model_name not in self.metadata['models']:
            self.metadata['models'].append(model_name)

        self.metadata['n_train_samples'] = len(predictions)

        logging.info(f"Logged OOF predictions for {model_name}: {len(predictions)} samples")

    def log_test_predictions(self, model_name: str, predictions: Union[np.ndarray, pd.Series],
                           y_true: Union[np.ndarray, pd.Series] = None,
                           indices: Union[np.ndarray, pd.Series, pd.Index] = None,
                           threshold: float = 0.5):
        """
        Log test set predictions

        Args:
            model_name: Name of the model
            predictions: Predicted probabilities
            y_true: True labels (optional)
            indices: Original indices in dataset (optional)
            threshold: Classification threshold
        """
        # Convert to numpy arrays for consistency
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(indices, (pd.Series, pd.Index)):
            indices = indices.values

        binary_preds = (predictions >= threshold).astype(int)

        self.test_predictions[model_name] = {
            'predictions': self._convert_to_serializable(predictions),
            'threshold': float(threshold),
            'binary_predictions': self._convert_to_serializable(binary_preds),
            'n_samples': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'n_positive': int(binary_preds.sum()),
            'positive_rate': float(binary_preds.mean())
        }

        if y_true is not None:
            self.test_predictions[model_name]['y_true'] = self._convert_to_serializable(y_true)

            # Calculate metrics
            from sklearn.metrics import roc_auc_score, confusion_matrix
            try:
                auc = roc_auc_score(y_true, predictions)
                self.test_predictions[model_name]['auc'] = float(auc)

                tn, fp, fn, tp = confusion_matrix(y_true, binary_preds).ravel()

                self.test_predictions[model_name]['confusion_matrix'] = {
                    'tn': int(tn), 'fp': int(fp),
                    'fn': int(fn), 'tp': int(tp)
                }
                self.test_predictions[model_name]['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
                self.test_predictions[model_name]['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            except Exception as e:
                logging.debug(f"Could not calculate metrics for {model_name}: {e}")

        if indices is not None:
            self.test_predictions[model_name]['indices'] = self._convert_to_serializable(indices)

        self.metadata['n_test_samples'] = len(predictions)

        logging.info(f"Logged test predictions for {model_name}: {len(predictions)} samples")

    def save_predictions(self):
        """Save all predictions to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        saved_files = {}

        # Save OOF predictions
        if self.oof_predictions:
            # Save as JSON
            oof_json_path = self.predictions_dir / f'oof_predictions_{timestamp}.json'
            with open(oof_json_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'predictions': self.oof_predictions
                }, f, indent=2, default=str)  # default=str handles any remaining non-serializable objects
            saved_files['oof_json'] = oof_json_path
            logging.info(f"OOF predictions (JSON) saved to: {oof_json_path}")

            # Save as CSV
            oof_df = self._create_oof_dataframe()
            oof_csv_path = self.predictions_dir / f'oof_predictions_{timestamp}.csv'
            oof_df.to_csv(oof_csv_path, index=False)
            saved_files['oof_csv'] = oof_csv_path
            logging.info(f"OOF predictions (CSV) saved to: {oof_csv_path}")

            # Save detailed OOF predictions with true labels if available
            if any('y_true' in data for data in self.oof_predictions.values()):
                detailed_oof = self._create_detailed_oof_dataframe()
                if detailed_oof is not None:
                    detailed_path = self.predictions_dir / f'oof_predictions_detailed_{timestamp}.csv'
                    detailed_oof.to_csv(detailed_path, index=False)
                    saved_files['oof_detailed'] = detailed_path
                    logging.info(f"Detailed OOF predictions saved to: {detailed_path}")

        # Save test predictions
        if self.test_predictions:
            # Save as JSON
            test_json_path = self.predictions_dir / f'test_predictions_{timestamp}.json'
            with open(test_json_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'predictions': self.test_predictions
                }, f, indent=2, default=str)
            saved_files['test_json'] = test_json_path
            logging.info(f"Test predictions (JSON) saved to: {test_json_path}")

            # Save as CSV
            test_df = self._create_test_dataframe()
            test_csv_path = self.predictions_dir / f'test_predictions_{timestamp}.csv'
            test_df.to_csv(test_csv_path, index=False)
            saved_files['test_csv'] = test_csv_path
            logging.info(f"Test predictions (CSV) saved to: {test_csv_path}")

            # Save detailed test predictions with true labels if available
            if any('y_true' in data for data in self.test_predictions.values()):
                detailed_test = self._create_detailed_test_dataframe()
                if detailed_test is not None:
                    detailed_path = self.predictions_dir / f'test_predictions_detailed_{timestamp}.csv'
                    detailed_test.to_csv(detailed_path, index=False)
                    saved_files['test_detailed'] = detailed_path
                    logging.info(f"Detailed test predictions saved to: {detailed_path}")

        # Create symlinks to latest files
        self._create_latest_links(saved_files)

        return saved_files

    def _create_oof_dataframe(self) -> pd.DataFrame:
        """Create a summary dataframe of OOF predictions"""
        rows = []
        for model_name, data in self.oof_predictions.items():
            row = {
                'Model': model_name,
                'CV_Type': data.get('cv_type', self.metadata.get('cv_type', 'unknown')),
                'N_Samples': data['n_samples'],
                'Mean_Prediction': data['mean'],
                'Std_Prediction': data['std'],
                'Min_Prediction': data['min'],
                'Max_Prediction': data['max']
            }

            if 'auc' in data:
                row['AUC'] = data['auc']

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create a summary dataframe of test predictions"""
        rows = []
        for model_name, data in self.test_predictions.items():
            row = {
                'Model': model_name,
                'N_Samples': data['n_samples'],
                'Threshold': data['threshold'],
                'Mean_Prediction': data['mean'],
                'Std_Prediction': data['std'],
                'Min_Prediction': data['min'],
                'Max_Prediction': data['max'],
                'N_Positive': data['n_positive'],
                'Positive_Rate': data['positive_rate']
            }

            if 'auc' in data:
                row['AUC'] = data['auc']
            if 'sensitivity' in data:
                row['Sensitivity'] = data['sensitivity']
            if 'specificity' in data:
                row['Specificity'] = data['specificity']

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_detailed_oof_dataframe(self) -> Optional[pd.DataFrame]:
        """Create detailed OOF predictions dataframe with all samples"""
        # Get first model to determine structure
        first_model = list(self.oof_predictions.keys())[0]
        n_samples = self.oof_predictions[first_model]['n_samples']

        # Initialize dataframe
        df_data = {}

        # Add sample index
        df_data['sample_index'] = list(range(n_samples))

        # Add true labels if available
        if 'y_true' in self.oof_predictions[first_model]:
            df_data['y_true'] = self.oof_predictions[first_model]['y_true']

        # Add predictions for each model
        for model_name, data in self.oof_predictions.items():
            if 'predictions' in data:
                df_data[f'{model_name}_prob'] = data['predictions']
                # Add binary predictions at 0.5 threshold
                df_data[f'{model_name}_pred'] = [1 if p >= 0.5 else 0 for p in data['predictions']]

        # Add original indices if available
        if 'indices' in self.oof_predictions[first_model]:
            df_data['original_index'] = self.oof_predictions[first_model]['indices']

        return pd.DataFrame(df_data)

    def _create_detailed_test_dataframe(self) -> Optional[pd.DataFrame]:
        """Create detailed test predictions dataframe with all samples"""
        # Get first model to determine structure
        first_model = list(self.test_predictions.keys())[0]
        n_samples = self.test_predictions[first_model]['n_samples']

        # Initialize dataframe
        df_data = {}

        # Add sample index
        df_data['sample_index'] = list(range(n_samples))

        # Add true labels if available
        if 'y_true' in self.test_predictions[first_model]:
            df_data['y_true'] = self.test_predictions[first_model]['y_true']

        # Add predictions for each model
        for model_name, data in self.test_predictions.items():
            if 'predictions' in data:
                df_data[f'{model_name}_prob'] = data['predictions']
                df_data[f'{model_name}_pred'] = data['binary_predictions']

        # Add original indices if available
        if 'indices' in self.test_predictions[first_model]:
            df_data['original_index'] = self.test_predictions[first_model]['indices']

        # Calculate ensemble predictions (mean of all models)
        prob_cols = [col for col in df_data.keys() if col.endswith('_prob')]
        if prob_cols:
            df = pd.DataFrame(df_data)
            df_data['ensemble_mean_prob'] = df[prob_cols].mean(axis=1).tolist()
            df_data['ensemble_mean_pred'] = [1 if p >= 0.5 else 0 for p in df_data['ensemble_mean_prob']]

        return pd.DataFrame(df_data)

    def _create_latest_links(self, saved_files: Dict[str, Path]):
        """Create symlinks to latest prediction files"""
        try:
            for key, file_path in saved_files.items():
                if file_path and file_path.exists():
                    # Determine link name
                    if 'oof' in key:
                        if 'json' in key:
                            link_name = 'latest_oof_predictions.json'
                        elif 'detailed' in key:
                            link_name = 'latest_oof_detailed.csv'
                        else:
                            link_name = 'latest_oof_predictions.csv'
                    elif 'test' in key:
                        if 'json' in key:
                            link_name = 'latest_test_predictions.json'
                        elif 'detailed' in key:
                            link_name = 'latest_test_detailed.csv'
                        else:
                            link_name = 'latest_test_predictions.csv'
                    else:
                        continue

                    link_path = self.predictions_dir / link_name

                    # Remove old link if exists
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()

                    # Create new link
                    link_path.symlink_to(file_path.name)

            logging.info(f"Created latest links in {self.predictions_dir}")
        except Exception as e:
            logging.debug(f"Could not create symlinks: {e}")