"""
Feature Importance and SHAP Value Extractor for ALTitude Pipeline
Extracts and saves feature importances from all trained models
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Optional: SHAP support
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")


class FeatureImportanceExtractor:
    """Extract and save feature importances from all trained models"""
    
    def __init__(self, config):
        self.config = config
        self.importances_dir = config.output_dir / "feature_importances"
        self.importances_dir.mkdir(exist_ok=True, parents=True)
    
    def extract_all_importances(self, models_final: List[Tuple[str, Any]], 
                               X_train: pd.DataFrame,
                               X_train_scaled: np.ndarray) -> Dict[str, Dict]:
        """
        Extract feature importances from all models
        
        Returns:
            Dictionary with model names as keys and importance dictionaries as values
        """
        all_importances = {}
        feature_names = list(X_train.columns)
        
        for model_name, model in models_final:
            logging.info(f"Extracting importances for {model_name}...")
            
            try:
                importances = self._extract_model_importance(
                    model, model_name, feature_names, X_train, X_train_scaled
                )
                
                if importances is not None:
                    all_importances[model_name] = importances
                    logging.info(f"✅ Extracted {len(importances)} feature importances for {model_name}")
                else:
                    logging.warning(f"⚠️ No importances available for {model_name}")
                    
            except Exception as e:
                logging.error(f"❌ Failed to extract importances for {model_name}: {e}")
                all_importances[model_name] = {feat: 0.0 for feat in feature_names}
        
        return all_importances
    
    def _extract_model_importance(self, model: Any, model_name: str, 
                                 feature_names: List[str],
                                 X_train: pd.DataFrame,
                                 X_train_scaled: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract importances from a single model"""
        
        # Handle pipeline models
        if hasattr(model, 'steps'):
            # It's a sklearn Pipeline, get the last step
            model = model.steps[-1][1]
        
        # Tree-based models with feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Convert to native Python floats
            return {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        
        # Linear models with coef_
        elif hasattr(model, 'coef_'):
            # Use absolute values of coefficients as importance
            if model.coef_.ndim > 1:
                # Multi-class or multi-output
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
            # Convert to native Python floats
            return {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        
        # TabPFN models - special handling
        elif 'TabPFN' in model_name:
            # TabPFN doesn't have native feature importance
            # Could use permutation importance instead
            logging.info(f"TabPFN model detected - using uniform importance as placeholder")
            uniform_importance = 1.0 / len(feature_names)
            return {feat: float(uniform_importance) for feat in feature_names}
        
        else:
            logging.warning(f"Model {model_name} has no known importance attribute")
            return None
    
    def calculate_shap_values(self, models_final: List[Tuple[str, Any]], 
                             X_train: pd.DataFrame,
                             X_train_scaled: np.ndarray,
                             y_train: pd.Series,
                             max_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate SHAP values for all models
        
        Args:
            max_samples: Maximum number of samples to use for SHAP calculation
            
        Returns:
            Dictionary with model names as keys and SHAP value arrays as values
        """
        if not SHAP_AVAILABLE:
            logging.error("SHAP is not installed. Cannot calculate SHAP values.")
            return {}
        
        all_shap_values = {}
        
        # Limit samples for performance
        if len(X_train) > max_samples:
            sample_indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_sample = X_train.iloc[sample_indices]
            X_scaled_sample = X_train_scaled[sample_indices]
        else:
            X_sample = X_train
            X_scaled_sample = X_train_scaled
        
        for model_name, model in models_final:
            logging.info(f"Calculating SHAP values for {model_name}...")
            
            try:
                # Choose appropriate data based on model type
                if 'TabPFN' in model_name:
                    X_for_shap = X_sample.values
                else:
                    X_for_shap = X_scaled_sample
                
                # Create appropriate explainer
                if any(name in model_name for name in ['Forest', 'XGBoost', 'LightGBM', 'CatBoost']):
                    # Tree-based model
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_for_shap)
                    
                    # Handle binary classification output
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]  # Use positive class
                        
                else:
                    # Use KernelExplainer for other models
                    explainer = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        shap.sample(X_for_shap, min(100, len(X_for_shap)))
                    )
                    shap_values = explainer.shap_values(X_for_shap)
                    
                    # Handle binary classification output
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                
                all_shap_values[model_name] = shap_values
                logging.info(f"✅ Calculated SHAP values for {model_name}: shape {shap_values.shape}")
                
            except Exception as e:
                logging.error(f"❌ Failed to calculate SHAP values for {model_name}: {e}")
        
        return all_shap_values
    
    def save_importances_to_csv(self, all_importances: Dict[str, Dict], 
                               filename: str = "feature_importances.csv"):
        """Save feature importances to CSV file"""
        
        # Convert to DataFrame, ensuring all values are native Python types
        df_data = {}
        for model_name, importances in all_importances.items():
            df_data[model_name] = {k: float(v) for k, v in importances.items()}
        
        df = pd.DataFrame(df_data)
        
        # Sort by mean importance across all models
        df['mean_importance'] = df.mean(axis=1)
        df = df.sort_values('mean_importance', ascending=False)
        
        # Add rank column
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['rank', 'mean_importance'] + [col for col in df.columns if col not in ['rank', 'mean_importance']]
        df = df[cols]
        
        # Save to CSV
        output_path = self.importances_dir / filename
        df.to_csv(output_path, index_label='feature')
        
        logging.info(f"✅ Feature importances saved to: {output_path}")
        return output_path
    
    def save_importances_to_json(self, all_importances: Dict[str, Dict],
                                filename: str = "feature_importances.json"):
        """Save feature importances to JSON file"""
        
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        # Calculate summary statistics
        features = list(next(iter(all_importances.values())).keys())
        summary_stats = {}
        
        for feature in features:
            values = [importances.get(feature, 0) for importances in all_importances.values()]
            summary_stats[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Create output structure
        output_data = {
            'model_importances': all_importances,
            'summary_statistics': summary_stats,
            'metadata': {
                'n_models': len(all_importances),
                'n_features': len(features),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save to JSON using custom encoder
        output_path = self.importances_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"✅ Feature importances saved to: {output_path}")
        return output_path
    
    def save_shap_values(self, all_shap_values: Dict[str, np.ndarray],
                        feature_names: List[str],
                        filename_prefix: str = "shap_values"):
        """Save SHAP values to files"""
        
        if not all_shap_values:
            logging.warning("No SHAP values to save")
            return
        
        # Save mean absolute SHAP values as CSV
        mean_shap_df = pd.DataFrame()
        
        for model_name, shap_values in all_shap_values.items():
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            mean_shap_df[model_name] = mean_abs_shap
        
        mean_shap_df.index = feature_names
        mean_shap_df['mean_shap'] = mean_shap_df.mean(axis=1)
        mean_shap_df = mean_shap_df.sort_values('mean_shap', ascending=False)
        
        # Save to CSV
        csv_path = self.importances_dir / f"{filename_prefix}_summary.csv"
        mean_shap_df.to_csv(csv_path, index_label='feature')
        
        logging.info(f"✅ SHAP summary saved to: {csv_path}")
        
        # Save detailed SHAP values for each model as numpy arrays
        for model_name, shap_values in all_shap_values.items():
            npy_path = self.importances_dir / f"{filename_prefix}_{model_name.replace(' ', '_')}.npy"
            np.save(npy_path, shap_values)
            logging.info(f"✅ SHAP values for {model_name} saved to: {npy_path}")
        
        return csv_path
    
    def create_importance_report(self, all_importances: Dict[str, Dict],
                                all_shap_values: Optional[Dict[str, np.ndarray]] = None,
                                top_n: int = 20):
        """Create a comprehensive importance report"""
        
        report_path = self.importances_dir / "importance_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE IMPORTANCE REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Number of models: {len(all_importances)}\n")
            f.write(f"Number of features: {len(next(iter(all_importances.values())))}\n\n")
            
            # Top features by mean importance
            f.write("TOP FEATURES BY MEAN IMPORTANCE\n")
            f.write("-"*50 + "\n")
            
            # Calculate mean importances
            features = list(next(iter(all_importances.values())).keys())
            mean_importances = {}
            
            for feature in features:
                values = [imp.get(feature, 0) for imp in all_importances.values()]
                mean_importances[feature] = np.mean(values)
            
            # Sort and display top features
            sorted_features = sorted(mean_importances.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
                f.write(f"{i:3d}. {feature:30s} {importance:.6f}\n")
            
            # Model-specific top features
            f.write("\n\nTOP 10 FEATURES BY MODEL\n")
            f.write("="*70 + "\n")
            
            for model_name, importances in all_importances.items():
                f.write(f"\n{model_name}\n")
                f.write("-"*len(model_name) + "\n")
                
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, imp) in enumerate(sorted_imp[:10], 1):
                    f.write(f"{i:2d}. {feature:30s} {imp:.6f}\n")
            
            # SHAP values section if available
            if all_shap_values:
                f.write("\n\nSHAP VALUE ANALYSIS\n")
                f.write("="*70 + "\n")
                
                for model_name, shap_values in all_shap_values.items():
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    top_shap_indices = np.argsort(mean_abs_shap)[-10:][::-1]
                    
                    f.write(f"\n{model_name} - Top 10 by SHAP\n")
                    f.write("-"*len(model_name) + "\n")
                    
                    for i, idx in enumerate(top_shap_indices, 1):
                        f.write(f"{i:2d}. {features[idx]:30s} {mean_abs_shap[idx]:.6f}\n")
        
        logging.info(f"✅ Importance report saved to: {report_path}")
        return report_path


# METHOD 2: Add a parameter to control SHAP calculation
def integrate_with_ml_pipeline_configurable(pipeline, models_final, X_train, X_train_scaled, y_train,
                                            calculate_shap=False, shap_max_samples=500):
    """
    Integrate feature importance extraction with configurable SHAP

    Args:
        calculate_shap: Whether to calculate SHAP values (default: False)
        shap_max_samples: Maximum samples for SHAP if enabled
    """

    extractor = FeatureImportanceExtractor(pipeline.config)

    # Extract feature importances
    logging.info("🎯 Extracting feature importances from all models...")
    all_importances = extractor.extract_all_importances(models_final, X_train, X_train_scaled)

    # Save importances
    csv_path = extractor.save_importances_to_csv(all_importances)
    json_path = extractor.save_importances_to_json(all_importances)

    # Calculate SHAP values only if requested
    all_shap_values = None
    if calculate_shap and SHAP_AVAILABLE:
        logging.info("🎯 Calculating SHAP values (this may take a while)...")
        all_shap_values = extractor.calculate_shap_values(
            models_final, X_train, X_train_scaled, y_train, max_samples=shap_max_samples
        )

        if all_shap_values:
            extractor.save_shap_values(all_shap_values, list(X_train.columns))
    else:
        logging.info("ℹ️ SHAP calculations disabled")

    # Create comprehensive report
    report_path = extractor.create_importance_report(all_importances, all_shap_values)

    logging.info("✅ Feature importance extraction complete!")
    logging.info(f"📁 Results saved to: {extractor.importances_dir}")

    return {
        'importances': all_importances,
        'shap_values': all_shap_values,
        'csv_path': csv_path,
        'json_path': json_path,
        'report_path': report_path
    }


# Example usage function
def extract_importances_from_saved_models(model_dir: Path, data_path: str):
    """
    Extract importances from already saved models
    
    Args:
        model_dir: Directory containing saved models
        data_path: Path to training data (needed for SHAP)
    """
    from altitude_model_manager import ModelManager
    from altitude_core import DataLoader, Config
    
    # Load saved models
    model_manager = ModelManager(model_dir)
    training_state = model_manager.load_training_state()
    
    # Load training data
    config = training_state['config']
    data_loader = DataLoader(config)
    X_train_scaled, _, X_train, _, y_train, _ = data_loader.load_and_split_data()
    
    # Extract importances
    extractor = FeatureImportanceExtractor(config)
    all_importances = extractor.extract_all_importances(
        training_state['fitted_models'], X_train, X_train_scaled
    )
    
    # Save results
    extractor.save_importances_to_csv(all_importances)
    extractor.save_importances_to_json(all_importances)
    extractor.create_importance_report(all_importances)
    
    return all_importances


if __name__ == "__main__":
    print("Feature Importance Extractor for ALTitude Pipeline")
    print("="*50)
    print("This module extracts and saves feature importances from all trained models")
    print("\nFeatures:")
    print("- Extract native feature importances from tree-based models")
    print("- Extract coefficient-based importances from linear models")
    print("- Calculate SHAP values for model-agnostic importance")
    print("- Save results in multiple formats (CSV, JSON, NPY)")
    print("- Generate comprehensive importance reports")
    print("\nIntegrate with your pipeline by calling integrate_with_ml_pipeline()")
