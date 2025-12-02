"""
Feature Importance Extractor for ALTitude Pipeline - Simplified version
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Optional SHAP support
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.info("SHAP not installed. Install with: pip install shap")


class FeatureImportanceExtractor:
    """Extract and save feature importances from trained models"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir / "feature_importances"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def extract_importances(self, models: List[tuple], feature_names: List[str]) -> Dict[str, Dict]:
        """
        Extract feature importances from all models
        
        Args:
            models: List of (name, model) tuples
            feature_names: List of feature names
            
        Returns:
            Dictionary with model names as keys and importance dicts as values
        """
        all_importances = {}
        
        for model_name, model in models:
            logging.info(f"Extracting importances for {model_name}...")
            
            try:
                importances = self._get_model_importance(model, model_name, feature_names)
                
                if importances is not None:
                    all_importances[model_name] = importances
                    logging.info(f"✅ Extracted {len(importances)} features for {model_name}")
                else:
                    logging.debug(f"No importances available for {model_name}")
                    
            except Exception as e:
                logging.warning(f"Could not extract importances for {model_name}: {e}")
                # Use uniform importance as fallback
                all_importances[model_name] = {feat: 1.0/len(feature_names) 
                                              for feat in feature_names}
        
        return all_importances
    
    def _get_model_importance(self, model: Any, model_name: str, 
                             feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract importances from a single model"""
        
        # Handle pipeline models
        if hasattr(model, 'steps'):
            # Get the last step from sklearn Pipeline
            model = model.steps[-1][1]
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        
        # Linear models
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-class: use mean absolute coefficient
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            return {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        
        # TabPFN or other models without native importance
        elif 'TabPFN' in model_name or 'Naive' in model_name:
            # Return None to indicate no native importance
            return None
        
        else:
            logging.debug(f"Model {model_name} has no known importance attribute")
            return None
    
    def calculate_shap_values(self, models: List[tuple], X_sample: np.ndarray,
                             feature_names: List[str], max_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate SHAP values for interpretable models only
        
        Args:
            models: List of (name, model) tuples
            X_sample: Sample of training data
            feature_names: List of feature names
            max_samples: Maximum samples for SHAP calculation
            
        Returns:
            Dictionary with model names as keys and SHAP arrays as values
        """
        if not SHAP_AVAILABLE:
            logging.info("SHAP not available")
            return {}
        
        shap_results = {}
        
        # Limit samples for performance
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]
        
        for model_name, model in models:
            # Only calculate SHAP for tree-based models (fast)
            if any(name in model_name for name in ['Forest', 'XGBoost', 'LightGBM', 'CatBoost']):
                logging.info(f"Calculating SHAP values for {model_name}...")
                
                try:
                    # Handle pipeline
                    if hasattr(model, 'steps'):
                        model = model.steps[-1][1]
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Handle binary classification output
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]  # Use positive class
                    
                    shap_results[model_name] = shap_values
                    logging.info(f"✅ SHAP values calculated for {model_name}")
                    
                except Exception as e:
                    logging.warning(f"SHAP calculation failed for {model_name}: {e}")
        
        return shap_results
    
    def save_results(self, importances: Dict[str, Dict], 
                    shap_values: Optional[Dict] = None) -> Dict[str, Path]:
        """Save importance results to files"""
        
        saved_files = {}
        
        # Save as CSV
        if importances:
            csv_path = self._save_csv(importances)
            saved_files['csv'] = csv_path
        
        # Save as JSON
        if importances:
            json_path = self._save_json(importances)
            saved_files['json'] = json_path
        
        # Save SHAP values if available
        if shap_values:
            shap_path = self._save_shap(shap_values, list(next(iter(importances.values())).keys()))
            saved_files['shap'] = shap_path
        
        # Create summary report
        report_path = self._create_report(importances, shap_values)
        saved_files['report'] = report_path
        
        return saved_files
    
    def _save_csv(self, importances: Dict[str, Dict]) -> Path:
        """Save importances as CSV"""
        
        # Convert to DataFrame
        df = pd.DataFrame(importances)
        
        # Add mean importance
        df['mean_importance'] = df.mean(axis=1)
        df = df.sort_values('mean_importance', ascending=False)
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['rank', 'mean_importance'] + [col for col in df.columns 
                                              if col not in ['rank', 'mean_importance']]
        df = df[cols]
        
        # Save
        csv_path = self.output_dir / "feature_importances.csv"
        df.to_csv(csv_path, index_label='feature')
        
        logging.info(f"✅ CSV saved to: {csv_path}")
        return csv_path
    
    def _save_json(self, importances: Dict[str, Dict]) -> Path:
        """Save importances as JSON with statistics"""
        
        # Calculate statistics
        features = list(next(iter(importances.values())).keys())
        stats = {}
        
        for feature in features:
            values = [imp.get(feature, 0) for imp in importances.values()]
            stats[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Create output structure
        output = {
            'model_importances': importances,
            'feature_statistics': stats,
            'metadata': {
                'n_models': len(importances),
                'n_features': len(features),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save
        json_path = self.output_dir / "feature_importances.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, default=float)
        
        logging.info(f"✅ JSON saved to: {json_path}")
        return json_path
    
    def _save_shap(self, shap_values: Dict[str, np.ndarray], 
                   feature_names: List[str]) -> Path:
        """Save SHAP values summary"""
        
        # Calculate mean absolute SHAP values
        shap_summary = pd.DataFrame()
        
        for model_name, values in shap_values.items():
            mean_abs_shap = np.mean(np.abs(values), axis=0)
            shap_summary[model_name] = mean_abs_shap
        
        shap_summary.index = feature_names[:len(shap_summary)]
        shap_summary['mean_shap'] = shap_summary.mean(axis=1)
        shap_summary = shap_summary.sort_values('mean_shap', ascending=False)
        
        # Save
        shap_path = self.output_dir / "shap_summary.csv"
        shap_summary.to_csv(shap_path, index_label='feature')
        
        logging.info(f"✅ SHAP summary saved to: {shap_path}")
        return shap_path
    
    def _create_report(self, importances: Dict[str, Dict], 
                      shap_values: Optional[Dict] = None) -> Path:
        """Create text report of feature importances"""
        
        report_path = self.output_dir / "importance_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FEATURE IMPORTANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Models analyzed: {len(importances)}\n")
            
            if importances:
                features = list(next(iter(importances.values())).keys())
                f.write(f"Features: {len(features)}\n\n")
                
                # Calculate mean importances
                mean_imp = {}
                for feature in features:
                    values = [imp.get(feature, 0) for imp in importances.values()]
                    mean_imp[feature] = np.mean(values)
                
                # Top features
                f.write("TOP 20 FEATURES BY MEAN IMPORTANCE\n")
                f.write("-" * 40 + "\n")
                
                sorted_features = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:20], 1):
                    f.write(f"{i:3d}. {feature[:30]:30s} {importance:.6f}\n")
                
                # Model-specific top features
                f.write("\n\nTOP 10 FEATURES BY MODEL\n")
                f.write("=" * 60 + "\n")
                
                for model_name, imp in importances.items():
                    f.write(f"\n{model_name}\n")
                    f.write("-" * len(model_name) + "\n")
                    
                    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                    for i, (feature, value) in enumerate(sorted_imp[:10], 1):
                        f.write(f"{i:2d}. {feature[:30]:30s} {value:.6f}\n")
            
            if shap_values:
                f.write("\n\nSHAP VALUE ANALYSIS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Models with SHAP values: {len(shap_values)}\n")
                
                for model_name in shap_values:
                    f.write(f"  - {model_name}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
        
        logging.info(f"✅ Report saved to: {report_path}")
        return report_path


def integrate_with_ml_pipeline_configurable(pipeline, models_final, X_train, 
                                           X_train_scaled, y_train,
                                           calculate_shap=False, 
                                           shap_max_samples=100):
    """
    Integrate feature importance extraction with ML pipeline
    
    Args:
        pipeline: MLPipeline instance
        models_final: List of trained models
        X_train: Training data (DataFrame)
        X_train_scaled: Scaled training data (array)
        y_train: Training labels
        calculate_shap: Whether to calculate SHAP values
        shap_max_samples: Max samples for SHAP calculation
        
    Returns:
        Dictionary with paths to saved files
    """
    
    extractor = FeatureImportanceExtractor(pipeline.config)
    
    # Extract feature importances
    logging.info("Extracting feature importances...")
    importances = extractor.extract_importances(models_final, list(X_train.columns))
    
    # Calculate SHAP values if requested
    shap_values = None
    if calculate_shap and SHAP_AVAILABLE:
        logging.info("Calculating SHAP values...")
        # Use appropriate data format for different models
        shap_values = extractor.calculate_shap_values(
            models_final, X_train_scaled, list(X_train.columns), shap_max_samples
        )
    
    # Save results
    saved_files = extractor.save_results(importances, shap_values)
    
    logging.info("✅ Feature importance extraction complete")
    
    return {
        'importances': importances,
        'shap_values': shap_values,
        **saved_files
    }
