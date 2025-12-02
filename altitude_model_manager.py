"""
ALTitude Model Manager - Simplified model persistence and loading
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler

from altitude_core import Config


class ModelManager:
    """Manages saving and loading of trained model states"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Define file paths - INCLUDING data_imputer
        self.paths = {
            'base_models': self.model_dir / "base_models.pkl",
            'meta_learners': self.model_dir / "meta_learners.pkl",
            'data_imputer': self.model_dir / "data_imputer.pkl",  # Added
            'data_scaler': self.model_dir / "data_scaler.pkl",
            'meta_scaler': self.model_dir / "meta_scaler.pkl",
            'config': self.model_dir / "config.json",
            'model_info': self.model_dir / "model_info.json",
            'summary': self.model_dir / "model_summary.csv",
            'completion_marker': self.model_dir / "training_complete.txt"
        }

    def save_training_state(self, fitted_models: List, meta_learners: Dict,
                           data_imputer: Any, data_scaler: Any, meta_scaler: Any,
                           selected_models: List[str], selected_indices: List[int],
                           config: Config, model_summary: pd.DataFrame) -> bool:
        """Save complete training state to disk"""
        
        logging.info("Saving training state...")
        
        try:
            # Save models and preprocessing objects
            components = {
                'base_models': fitted_models,
                'meta_learners': meta_learners,
                'data_imputer': data_imputer,   # Save the imputer
                'data_scaler': data_scaler,     # Keep for compatibility
                'meta_scaler': meta_scaler
            }
            
            for name, component in components.items():
                if component is not None:
                    path = self.paths.get(name, self.model_dir / f"{name}.pkl")
                    with open(path, 'wb') as f:
                        pickle.dump(component, f)
                    logging.info(f"✅ {name} saved")
                else:
                    logging.warning(f"⚠️ {name} is None, skipping")
            
            # Save configuration
            config.to_file(str(self.paths['config']))
            logging.info("✅ Configuration saved")
            
            # Save model selection info
            model_info = {
                'selected_models': selected_models,
                'selected_indices': selected_indices,
                'n_base_models': len(fitted_models),
                'n_meta_learners': len(meta_learners),
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_model': model_summary.loc[model_summary['Mean ROC AUC'].idxmax(), 'Model'],
                'best_auc': float(model_summary['Mean ROC AUC'].max())
            }
            
            with open(self.paths['model_info'], 'w') as f:
                json.dump(model_info, f, indent=2)
            logging.info("✅ Model info saved")
            
            # Save model summary
            model_summary.to_csv(self.paths['summary'], index=False)
            logging.info("✅ Model summary saved")
            
            # Create completion marker
            with open(self.paths['completion_marker'], 'w') as f:
                f.write(f"Training completed: {pd.Timestamp.now()}\n")
                f.write(f"Best model: {model_info['best_model']}\n")
                f.write(f"Best AUC: {model_info['best_auc']:.4f}\n")
            
            logging.info(f"✅ Training state saved to: {self.model_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving training state: {e}")
            return False
    
    def load_training_state(self) -> Dict[str, Any]:
        """Load complete training state from disk"""
        
        logging.info("Loading training state...")
        
        if not self.check_saved_state():
            raise FileNotFoundError("No complete saved training state found")
        
        try:
            state = {}
            
            # Load models and scalers - INCLUDING data_imputer
            components = ['base_models', 'meta_learners', 'data_imputer', 'data_scaler', 'meta_scaler']
            
            for name in components:
                path = self.paths.get(name, self.model_dir / f"{name}.pkl")
                if path.exists():
                    with open(path, 'rb') as f:
                        state[name] = pickle.load(f)
                    logging.info(f"✅ {name} loaded")
                elif name == 'data_imputer':
                    # Try fallback for old models that don't have data_imputer.pkl
                    logging.warning(f"data_imputer.pkl not found, checking data_scaler.pkl as fallback")
                    if self.paths['data_scaler'].exists():
                        with open(self.paths['data_scaler'], 'rb') as f:
                            loaded = pickle.load(f)
                            # Check if it's actually an imputer (SimpleImputer has fit_transform method)
                            if hasattr(loaded, 'fit_transform') and hasattr(loaded, 'statistics_'):
                                state['data_imputer'] = loaded
                                logging.info("✅ Loaded imputer from data_scaler.pkl (old format)")
                            else:
                                logging.error("data_scaler.pkl does not contain a fitted imputer")
                                state['data_imputer'] = None
                    else:
                        logging.error("No imputer found in saved state")
                        state['data_imputer'] = None
                else:
                    logging.warning(f"{name} not found")
            
            # Rename for consistency
            state['fitted_models'] = state.pop('base_models', [])
            
            # Load configuration
            state['config'] = Config.from_file(str(self.paths['config']))
            logging.info("✅ Configuration loaded")
            
            # Load model info
            with open(self.paths['model_info'], 'r') as f:
                model_info = json.load(f)
            state.update({
                'selected_models': model_info['selected_models'],
                'selected_indices': model_info['selected_indices'],
                'model_info': model_info
            })
            logging.info("✅ Model info loaded")
            
            # Load summary
            state['model_summary'] = pd.read_csv(self.paths['summary'])
            logging.info("✅ Model summary loaded")
            
            logging.info(f"✅ Training state loaded from: {self.model_dir}")
            return state
            
        except Exception as e:
            logging.error(f"Error loading training state: {e}")
            raise
    
    def check_saved_state(self) -> bool:
        """Check if a complete saved training state exists"""
        
        # For backward compatibility, data_imputer is optional
        required_files = ['base_models', 'meta_learners', 'data_scaler', 
                         'meta_scaler', 'config', 'model_info']
        
        missing = []
        for name in required_files:
            if not self.paths[name].exists():
                missing.append(name)
        
        if missing:
            logging.warning(f"Missing required files: {missing}")
            return False
        
        # Check for data_imputer but don't require it (for backward compatibility)
        if not self.paths['data_imputer'].exists():
            logging.warning("data_imputer.pkl not found (will try fallback)")
        
        if not self.paths['completion_marker'].exists():
            logging.warning("Training completion marker not found")
            return False
        
        return True
    
    def get_model_info(self) -> Optional[Dict]:
        """Get basic model information without loading full state"""
        
        if not self.paths['model_info'].exists():
            return None
        
        try:
            with open(self.paths['model_info'], 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading model info: {e}")
            return None
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate that all saved models can be loaded properly"""
        
        validation_results = {}
        
        # Check each component
        checks = [
            ('base_models', lambda x: len(x) > 0),
            ('meta_learners', lambda x: len(x) >= 0),  # Can be empty
            ('data_imputer', lambda x: x is None or hasattr(x, 'transform')),
            ('data_scaler', lambda x: hasattr(x, 'transform') or hasattr(x, 'fit_transform')),
            ('meta_scaler', lambda x: hasattr(x, 'transform'))
        ]
        
        for name, validator in checks:
            try:
                path = self.paths.get(name)
                if path and path.exists():
                    with open(path, 'rb') as f:
                        component = pickle.load(f)
                    validation_results[name] = validator(component)
                elif name == 'data_imputer':
                    # Optional for backward compatibility
                    validation_results[name] = True
                    logging.warning(f"{name} not found (optional)")
                else:
                    validation_results[name] = False
            except Exception as e:
                logging.error(f"{name} validation failed: {e}")
                validation_results[name] = False
        
        # Check config
        try:
            Config.from_file(str(self.paths['config']))
            validation_results['config'] = True
        except Exception:
            validation_results['config'] = False
        
        all_valid = all(validation_results.values())
        logging.info(f"Model validation: {'✅ All valid' if all_valid else '⚠️ Some invalid'}")
        
        return validation_results
    
    def export_model_summary(self, output_path: Optional[str] = None) -> Path:
        """Export a human-readable model summary"""
        
        if output_path is None:
            output_path = self.model_dir / "model_summary_report.txt"
        else:
            output_path = Path(output_path)
        
        try:
            model_info = self.get_model_info()
            if not model_info:
                raise FileNotFoundError("No model info available")
            
            summary = pd.read_csv(self.paths['summary']) if self.paths['summary'].exists() else None
            
            with open(output_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("ALTITUDE MODEL SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Training completed: {model_info.get('timestamp', 'Unknown')}\n")
                f.write(f"Model directory: {self.model_dir}\n\n")
                
                f.write("MODEL COUNTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Base models: {model_info.get('n_base_models', 0)}\n")
                f.write(f"Meta-learners: {model_info.get('n_meta_learners', 0)}\n")
                f.write(f"Selected for ensemble: {len(model_info.get('selected_models', []))}\n\n")
                
                f.write("BEST MODEL\n")
                f.write("-" * 30 + "\n")
                f.write(f"Model: {model_info.get('best_model', 'Unknown')}\n")
                f.write(f"AUC: {model_info.get('best_auc', 0):.4f}\n\n")
                
                if model_info.get('selected_models'):
                    f.write("SELECTED MODELS\n")
                    f.write("-" * 30 + "\n")
                    for model in model_info['selected_models']:
                        f.write(f"  - {model}\n")
                    f.write("\n")
                
                if summary is not None:
                    f.write("PERFORMANCE SUMMARY\n")
                    f.write("-" * 30 + "\n")
                    for _, row in summary.iterrows():
                        f.write(f"{row['Model']}: {row['Mean ROC AUC']:.4f}")
                        if 'Std' in row:
                            f.write(f" ± {row['Std']:.4f}")
                        f.write("\n")
                
                # Check for imputer
                if self.paths['data_imputer'].exists():
                    f.write("\n✅ Data imputer: Saved\n")
                else:
                    f.write("\n⚠️ Data imputer: Not found (predictions may fail)\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            logging.info(f"Model summary exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error exporting model summary: {e}")
            raise
    
    def cleanup_old_models(self, keep_latest: int = 3):
        """Clean up old model directories, keeping only the most recent ones"""
        
        parent_dir = self.model_dir.parent
        model_dirs = sorted(
            [d for d in parent_dir.glob("*") if d.is_dir() and (d / "training_complete.txt").exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(model_dirs) <= keep_latest:
            logging.info(f"Found {len(model_dirs)} model directories, no cleanup needed")
            return
        
        # Remove older directories
        for old_dir in model_dirs[keep_latest:]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                logging.info(f"Removed old model directory: {old_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove {old_dir}: {e}")
