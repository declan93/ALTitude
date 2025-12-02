"""
Results Logger for ALTitude Pipeline
Saves comprehensive training results including LOO CV details
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ResultsLogger:
    """Comprehensive results logging for training pipeline"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / 'training_results'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {},
            'data_info': {},
            'cv_results': {},
            'test_results': {},
            'model_performance': {},
            'feature_importance': {},
            'threshold_optimization': {},
            'ensemble_info': {}
        }
    
    def log_configuration(self, config: Any):
        """Log training configuration"""
        config_dict = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        self.results['configuration'] = config_dict
        
        # Determine CV type
        if config_dict.get('use_loo_cv', False):
            self.results['cv_type'] = 'Leave-One-Out'
            self.results['cv_details'] = {
                'max_samples': config_dict.get('loo_max_samples', 500)
            }
        else:
            self.results['cv_type'] = f"{config_dict.get('cv_folds', 5)}-fold"
            self.results['cv_details'] = {
                'n_folds': config_dict.get('cv_folds', 5)
            }
    
    def log_data_info(self, X_train_shape: tuple, X_test_shape: tuple, 
                     y_train_dist: Dict, y_test_dist: Dict,
                     feature_names: List[str] = None):
        """Log dataset information"""
        self.results['data_info'] = {
            'training_samples': X_train_shape[0],
            'test_samples': X_test_shape[0],
            'n_features': X_train_shape[1],
            'train_class_distribution': y_train_dist,
            'test_class_distribution': y_test_dist,
            'feature_count': len(feature_names) if feature_names else X_train_shape[1]
        }
        
        if feature_names and len(feature_names) <= 100:
            self.results['data_info']['feature_names'] = feature_names
    
    def log_cv_results(self, model_name: str, cv_scores: np.ndarray, 
                       oof_predictions: np.ndarray = None,
                       cv_type: str = None, fold_details: List[Dict] = None):
        """Log cross-validation results for a model"""
        if model_name not in self.results['cv_results']:
            self.results['cv_results'][model_name] = {}
        
        cv_result = {
            'cv_type': cv_type or self.results.get('cv_type', 'unknown'),
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'min_score': float(np.min(cv_scores)),
            'max_score': float(np.max(cv_scores)),
            'scores': cv_scores.tolist() if len(cv_scores) <= 10 else None
        }
        
        # For LOO CV, add detailed statistics
        if cv_type == 'LOO' and oof_predictions is not None:
            cv_result['loo_details'] = {
                'n_iterations': len(oof_predictions),
                'prediction_mean': float(np.mean(oof_predictions)),
                'prediction_std': float(np.std(oof_predictions)),
                'prediction_min': float(np.min(oof_predictions)),
                'prediction_max': float(np.max(oof_predictions))
            }
        
        # Add fold-level details if provided
        if fold_details and len(fold_details) <= 20:
            cv_result['fold_details'] = fold_details
        
        self.results['cv_results'][model_name] = cv_result
    
    def log_test_results(self, model_name: str, metrics: Dict):
        """Log test set performance metrics"""
        if model_name not in self.results['test_results']:
            self.results['test_results'][model_name] = {}
        
        # Ensure all values are JSON serializable
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        self.results['test_results'][model_name] = serializable_metrics
    
    def log_model_summary(self, summary_df: pd.DataFrame):
        """Log overall model performance summary"""
        self.results['model_performance']['summary'] = summary_df.to_dict('records')
        
        # Extract best model info
        if not summary_df.empty:
            best_idx = summary_df['Mean ROC AUC'].idxmax()
            best_row = summary_df.loc[best_idx]
            self.results['model_performance']['best_model'] = {
                'name': best_row['Model'],
                'auc': float(best_row['Mean ROC AUC']),
                'std': float(best_row.get('Std', 0)),
                'cv_type': best_row.get('CV Type', self.results.get('cv_type', 'unknown'))
            }
    
    def log_ensemble_info(self, selected_models: List[str], 
                         meta_learner_results: Dict = None):
        """Log ensemble/stacking information"""
        self.results['ensemble_info'] = {
            'n_selected_models': len(selected_models),
            'selected_models': selected_models
        }
        
        if meta_learner_results:
            meta_info = {}
            for name, result in meta_learner_results.items():
                if 'metrics' in result:
                    meta_info[name] = {
                        'auc': float(result['metrics'].get('auc', 0)),
                        'sensitivity': float(result['metrics'].get('sensitivity', 0)),
                        'specificity': float(result['metrics'].get('specificity', 0))
                    }
            self.results['ensemble_info']['meta_learners'] = meta_info
    
    def log_threshold_optimization(self, threshold_results: Dict):
        """Log threshold optimization results"""
        self.results['threshold_optimization'] = threshold_results
    
    def save_results(self):
        """Save all results to multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON (comprehensive)
        json_path = self.results_dir / f'training_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logging.info(f"Results saved to: {json_path}")
        
        # Save as formatted text report
        txt_path = self.results_dir / f'training_report_{timestamp}.txt'
        self._write_text_report(txt_path)
        logging.info(f"Report saved to: {txt_path}")
        
        # Save CV results as CSV
        if self.results['cv_results']:
            cv_df = self._create_cv_dataframe()
            cv_path = self.results_dir / f'cv_results_{timestamp}.csv'
            cv_df.to_csv(cv_path, index=False)
            logging.info(f"CV results saved to: {cv_path}")
        
        # Save test results as CSV
        if self.results['test_results']:
            test_df = self._create_test_dataframe()
            test_path = self.results_dir / f'test_results_{timestamp}.csv'
            test_df.to_csv(test_path, index=False)
            logging.info(f"Test results saved to: {test_path}")
        
        # Create and save latest symlinks for easy access
        self._create_latest_links(json_path, txt_path)
        
        return {
            'json': json_path,
            'text': txt_path,
            'cv_results': cv_path if self.results['cv_results'] else None,
            'test_results': test_path if self.results['test_results'] else None
        }
    
    def _write_text_report(self, path: Path):
        """Write formatted text report"""
        with open(path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ALTITUDE TRAINING RESULTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {self.results['timestamp']}\n")
            f.write(f"CV Type: {self.results.get('cv_type', 'Unknown')}\n\n")
            
            # Configuration section
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            config = self.results.get('configuration', {})
            f.write(f"  Random seed: {config.get('random_seed', 'N/A')}\n")
            f.write(f"  Test size: {config.get('test_size', 'N/A')}\n")
            if self.results.get('cv_type') == 'Leave-One-Out':
                f.write(f"  LOO max samples: {config.get('loo_max_samples', 'N/A')}\n")
            else:
                f.write(f"  CV folds: {config.get('cv_folds', 'N/A')}\n")
            f.write("\n")
            
            # Data information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            data_info = self.results.get('data_info', {})
            f.write(f"  Training samples: {data_info.get('training_samples', 'N/A')}\n")
            f.write(f"  Test samples: {data_info.get('test_samples', 'N/A')}\n")
            f.write(f"  Features: {data_info.get('n_features', 'N/A')}\n")
            
            if 'train_class_distribution' in data_info:
                f.write("  Training class distribution:\n")
                for cls, count in data_info['train_class_distribution'].items():
                    f.write(f"    Class {cls}: {count}\n")
            f.write("\n")
            
            # Best model
            if 'best_model' in self.results.get('model_performance', {}):
                best = self.results['model_performance']['best_model']
                f.write("BEST MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Model: {best['name']}\n")
                f.write(f"  AUC: {best['auc']:.4f} ± {best['std']:.4f}\n")
                f.write(f"  CV Type: {best['cv_type']}\n\n")
            
            # Cross-validation results
            if self.results.get('cv_results'):
                f.write("CROSS-VALIDATION RESULTS\n")
                f.write("-" * 40 + "\n")
                for model, cv_data in self.results['cv_results'].items():
                    f.write(f"\n{model}:\n")
                    f.write(f"  CV Type: {cv_data.get('cv_type', 'N/A')}\n")
                    f.write(f"  Mean Score: {cv_data.get('mean_score', 0):.4f}\n")
                    f.write(f"  Std Score: {cv_data.get('std_score', 0):.4f}\n")
                    f.write(f"  Min/Max: {cv_data.get('min_score', 0):.4f} / {cv_data.get('max_score', 0):.4f}\n")
                    
                    if 'loo_details' in cv_data:
                        loo = cv_data['loo_details']
                        f.write(f"  LOO Iterations: {loo.get('n_iterations', 0)}\n")
                        f.write(f"  Prediction Mean: {loo.get('prediction_mean', 0):.4f}\n")
                        f.write(f"  Prediction Std: {loo.get('prediction_std', 0):.4f}\n")
                f.write("\n")
            
            # Test results
            if self.results.get('test_results'):
                f.write("TEST SET PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                for model, metrics in self.results['test_results'].items():
                    f.write(f"\n{model}:\n")
                    f.write(f"  AUC: {metrics.get('auc', 0):.4f}\n")
                    f.write(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}\n")
                    f.write(f"  Specificity: {metrics.get('specificity', 0):.4f}\n")
                    if 'avg_precision' in metrics:
                        f.write(f"  Avg Precision: {metrics.get('avg_precision', 0):.4f}\n")
                f.write("\n")
            
            # Ensemble information
            if self.results.get('ensemble_info'):
                ensemble = self.results['ensemble_info']
                f.write("ENSEMBLE INFORMATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Selected models: {ensemble.get('n_selected_models', 0)}\n")
                if 'selected_models' in ensemble:
                    for model in ensemble['selected_models']:
                        f.write(f"    - {model}\n")
                
                if 'meta_learners' in ensemble:
                    f.write("\n  Meta-learners:\n")
                    for name, metrics in ensemble['meta_learners'].items():
                        f.write(f"    {name}: AUC={metrics['auc']:.4f}\n")
                f.write("\n")
            
            # Threshold optimization
            if self.results.get('threshold_optimization'):
                f.write("THRESHOLD OPTIMIZATION\n")
                f.write("-" * 40 + "\n")
                for model, opt_data in self.results['threshold_optimization'].items():
                    if isinstance(opt_data, dict):
                        f.write(f"\n{model}:\n")
                        f.write(f"  Optimal threshold: {opt_data.get('threshold', 0.5):.3f}\n")
                        f.write(f"  Sensitivity: {opt_data.get('sensitivity', 0):.3f}\n")
                        f.write(f"  Specificity: {opt_data.get('specificity', 0):.3f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
    
    def _create_cv_dataframe(self) -> pd.DataFrame:
        """Create dataframe from CV results"""
        rows = []
        for model, cv_data in self.results['cv_results'].items():
            row = {
                'Model': model,
                'CV_Type': cv_data.get('cv_type', ''),
                'Mean_Score': cv_data.get('mean_score', 0),
                'Std_Score': cv_data.get('std_score', 0),
                'Min_Score': cv_data.get('min_score', 0),
                'Max_Score': cv_data.get('max_score', 0)
            }
            
            if 'loo_details' in cv_data:
                row['LOO_Iterations'] = cv_data['loo_details'].get('n_iterations', 0)
                row['LOO_Pred_Mean'] = cv_data['loo_details'].get('prediction_mean', 0)
                row['LOO_Pred_Std'] = cv_data['loo_details'].get('prediction_std', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_test_dataframe(self) -> pd.DataFrame:
        """Create dataframe from test results"""
        rows = []
        for model, metrics in self.results['test_results'].items():
            row = {'Model': model}
            row.update(metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_latest_links(self, json_path: Path, txt_path: Path):
        """Create symlinks to latest results"""
        try:
            latest_json = self.results_dir / 'latest_results.json'
            latest_txt = self.results_dir / 'latest_report.txt'
            
            # Remove old symlinks if they exist
            if latest_json.exists() or latest_json.is_symlink():
                latest_json.unlink()
            if latest_txt.exists() or latest_txt.is_symlink():
                latest_txt.unlink()
            
            # Create new symlinks
            latest_json.symlink_to(json_path.name)
            latest_txt.symlink_to(txt_path.name)
            
            logging.info(f"Created latest links in {self.results_dir}")
        except Exception as e:
            logging.debug(f"Could not create symlinks: {e}")
