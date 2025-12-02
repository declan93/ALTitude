"""
ALTitude Utilities - Simplified helper functions
"""

import logging
import sys
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
import pandas as pd
import numpy as np


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    for lib in ['matplotlib', 'PIL', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """Validate that a file path exists and is readable"""
    path = Path(file_path)
    
    if not path.exists():
        logging.error(f"File does not exist: {path}")
        return False
    
    if not path.is_file():
        logging.error(f"Path is not a file: {path}")
        return False
    
    return True


def validate_data_file(file_path: Union[str, Path], 
                      required_columns: Optional[List[str]] = None) -> bool:
    """Validate that a data file can be loaded and has required columns"""
    if not validate_file_path(file_path):
        return False
    
    try:
        # Try to read the file
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, sep='\t')
        
        logging.info(f"Data file loaded: {df.shape}")
        
        # Check for required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                return False
        
        # Check for empty dataframe
        if df.empty:
            logging.error("Data file is empty")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        return False


def log_file_info(file_path: Union[str, Path], data_type: str = "data"):
    """Log detailed information about a loaded data file"""
    try:
        # Read file
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, sep='\t')
        
        logging.info(f"📊 {data_type.capitalize()} file details:")
        logging.info(f"   Path: {file_path}")
        logging.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logging.info(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show first few column names
        if len(df.columns) <= 10:
            logging.info(f"   Columns: {list(df.columns)}")
        else:
            logging.info(f"   Columns: {list(df.columns[:5])} ... (+{len(df.columns)-5} more)")
        
        # Show target distribution if ALT column exists
        if 'ALT' in df.columns:
            alt_counts = df['ALT'].value_counts().sort_index()
            total = len(df)
            logging.info(f"   Target distribution:")
            for value, count in alt_counts.items():
                pct = (count / total) * 100
                logging.info(f"     Class {value}: {count:,} ({pct:.1f}%)")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
            logging.warning(f"   Missing values: {missing:,} ({missing_pct:.1f}%)")
        else:
            logging.info(f"   Missing values: None ✅")
            
    except Exception as e:
        logging.error(f"Error analyzing {data_type} file: {e}")


def summarize_predictions(predictions_df: pd.DataFrame, 
                         threshold: float = 0.6) -> Dict:
    """Summarize prediction results"""
    prob_cols = [col for col in predictions_df.columns if col.endswith('_prob')]
    pred_cols = [col for col in predictions_df.columns if col.endswith('_pred')]
    
    summary = {
        'total_samples': len(predictions_df),
        'threshold': threshold,
        'n_probability_columns': len(prob_cols),
        'n_prediction_columns': len(pred_cols)
    }
    
    # Analyze ensemble predictions
    ensemble_cols = [col for col in pred_cols if col.startswith('SE_')]
    if ensemble_cols:
        summary['ensemble_predictions'] = {}
        for col in ensemble_cols:
            model_name = col.replace('SE_', '').replace('_pred', '')
            pos_count = predictions_df[col].sum()
            pos_pct = (pos_count / len(predictions_df)) * 100
            
            summary['ensemble_predictions'][model_name] = {
                'positive': int(pos_count),
                'negative': len(predictions_df) - int(pos_count),
                'positive_pct': round(pos_pct, 2)
            }
    
    # Get probability statistics
    if prob_cols:
        summary['probability_stats'] = {}
        for col in prob_cols[:5]:  # Limit to first 5 for brevity
            probs = predictions_df[col].dropna()
            if len(probs) > 0:
                model_name = col.replace('_prob', '').replace('SE_', '')
                summary['probability_stats'][model_name] = {
                    'mean': round(float(probs.mean()), 4),
                    'std': round(float(probs.std()), 4),
                    'min': round(float(probs.min()), 4),
                    'max': round(float(probs.max()), 4),
                    'median': round(float(probs.median()), 4)
                }
    
    return summary


def analyze_model_performance(summary_df: pd.DataFrame) -> Dict:
    """Analyze model performance from summary dataframe"""
    if summary_df.empty:
        return {}
    
    analysis = {
        'n_models': len(summary_df),
        'n_base_models': len(summary_df[~summary_df['Model'].str.startswith('SE')]),
        'n_meta_models': len(summary_df[summary_df['Model'].str.startswith('SE')]),
        'best_model': summary_df.loc[summary_df['Mean ROC AUC'].idxmax(), 'Model'],
        'best_auc': summary_df['Mean ROC AUC'].max(),
        'mean_auc': summary_df['Mean ROC AUC'].mean(),
        'std_auc': summary_df['Mean ROC AUC'].std()
    }
    
    # Compare base vs meta models
    base_aucs = summary_df[~summary_df['Model'].str.startswith('SE')]['Mean ROC AUC']
    meta_aucs = summary_df[summary_df['Model'].str.startswith('SE')]['Mean ROC AUC']
    
    if len(base_aucs) > 0:
        analysis['base_models_mean_auc'] = base_aucs.mean()
        analysis['base_models_best_auc'] = base_aucs.max()
    
    if len(meta_aucs) > 0:
        analysis['meta_models_mean_auc'] = meta_aucs.mean()
        analysis['meta_models_best_auc'] = meta_aucs.max()
    
    return analysis


def create_results_report(results: Dict, output_path: Path) -> Path:
    """Create a comprehensive results report"""
    report_path = output_path / "results_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALTITUDE PIPELINE RESULTS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Model performance
        if 'summary' in results:
            summary = results['summary']
            analysis = analyze_model_performance(summary)
            
            f.write("MODEL PERFORMANCE OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total models trained: {analysis.get('n_models', 0)}\n")
            f.write(f"  Base models: {analysis.get('n_base_models', 0)}\n")
            f.write(f"  Meta-learners: {analysis.get('n_meta_models', 0)}\n\n")
            
            f.write(f"Best model: {analysis.get('best_model', 'Unknown')}\n")
            f.write(f"Best AUC: {analysis.get('best_auc', 0):.4f}\n")
            f.write(f"Mean AUC (all models): {analysis.get('mean_auc', 0):.4f}\n\n")
            
            if 'base_models_mean_auc' in analysis:
                f.write(f"Base models - Mean AUC: {analysis['base_models_mean_auc']:.4f}\n")
                f.write(f"Base models - Best AUC: {analysis['base_models_best_auc']:.4f}\n")
            
            if 'meta_models_mean_auc' in analysis:
                f.write(f"Meta models - Mean AUC: {analysis['meta_models_mean_auc']:.4f}\n")
                f.write(f"Meta models - Best AUC: {analysis['meta_models_best_auc']:.4f}\n")
            
            f.write("\n")
        
        # Detailed model results
        if 'summary' in results:
            f.write("DETAILED MODEL RESULTS\n")
            f.write("-" * 40 + "\n")
            summary_sorted = results['summary'].sort_values('Mean ROC AUC', ascending=False)
            
            for _, row in summary_sorted.iterrows():
                f.write(f"\n{row['Model']}\n")
                f.write(f"  AUC: {row['Mean ROC AUC']:.4f}")
                if 'Std' in row and not pd.isna(row['Std']):
                    f.write(f" ± {row['Std']:.4f}")
                f.write("\n")
                
                if 'Sensitivity' in row and not pd.isna(row['Sensitivity']):
                    f.write(f"  Sensitivity: {row['Sensitivity']:.4f}\n")
                if 'Specificity' in row and not pd.isna(row['Specificity']):
                    f.write(f"  Specificity: {row['Specificity']:.4f}\n")
        
        # Selected models for ensemble
        if 'selected_models' in results:
            f.write("\n\nSELECTED MODELS FOR ENSEMBLE\n")
            f.write("-" * 40 + "\n")
            for model in results['selected_models']:
                f.write(f"  - {model}\n")
        
        # Feature importance summary
        if 'feature_importance' in results and results['feature_importance']:
            f.write("\n\nFEATURE IMPORTANCE\n")
            f.write("-" * 40 + "\n")
            f.write("Feature importance analysis completed\n")
            
            if 'csv_path' in results['feature_importance']:
                f.write(f"  CSV file: {results['feature_importance']['csv_path']}\n")
            if 'json_path' in results['feature_importance']:
                f.write(f"  JSON file: {results['feature_importance']['json_path']}\n")
            if 'report_path' in results['feature_importance']:
                f.write(f"  Report: {results['feature_importance']['report_path']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
    
    logging.info(f"Results report saved to: {report_path}")
    return report_path


def check_memory_usage(df: pd.DataFrame) -> Tuple[float, str]:
    """Check dataframe memory usage and provide recommendations"""
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_gb = memory_bytes / (1024**3)
    
    # Estimate model training memory (roughly 3-4x data size)
    estimated_gb = memory_gb * 4
    
    if estimated_gb < 4:
        recommendation = "Should run fine on most systems (8GB+ RAM)"
    elif estimated_gb < 8:
        recommendation = "Recommend at least 16GB RAM"
    elif estimated_gb < 16:
        recommendation = "Recommend at least 32GB RAM"
    else:
        recommendation = "Large dataset - consider 64GB+ RAM or data reduction"
    
    return memory_gb, recommendation


def validate_predictions(predictions_df: pd.DataFrame) -> List[str]:
    """Validate prediction dataframe for common issues"""
    issues = []
    
    # Check for prediction columns
    pred_cols = [col for col in predictions_df.columns if col.endswith('_pred')]
    prob_cols = [col for col in predictions_df.columns if col.endswith('_prob')]
    
    if not pred_cols and not prob_cols:
        issues.append("No prediction columns found")
    
    # Check probability ranges
    for col in prob_cols:
        probs = predictions_df[col].dropna()
        if len(probs) > 0:
            if probs.min() < 0 or probs.max() > 1:
                issues.append(f"{col}: probabilities outside [0,1] range")
    
    # Check for NaN values
    for col in pred_cols + prob_cols:
        nan_count = predictions_df[col].isna().sum()
        if nan_count > 0:
            issues.append(f"{col}: {nan_count} NaN values")
    
    # Check for constant predictions
    for col in pred_cols:
        unique_vals = predictions_df[col].dropna().nunique()
        if unique_vals == 1:
            issues.append(f"{col}: constant predictions (no variation)")
    
    return issues
