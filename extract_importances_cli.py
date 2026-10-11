#!/usr/bin/env python3
"""
Extract Feature Importances from Saved ALTitude Models
Usage:
    python extract_importances.py --model-dir ./results
    python extract_importances.py --model-dir ./results --calculate-shap --max-samples 1000
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np


from altitude_model_manager import ModelManager
from altitude_core import DataLoader, Config
from altitude_feature_importance import FeatureImportanceExtractor, SHAP_AVAILABLE


def main():
    parser = argparse.ArgumentParser(
        description="Extract feature importances from saved ALTitude models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-dir', 
        type=str, 
        default='./results',
        help='Directory containing saved models (default: ./results)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data (optional, uses config if not provided)'
    )
    
    parser.add_argument(
        '--calculate-shap',
        action='store_true',
        help='Calculate SHAP values (requires shap package)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=500,
        help='Maximum samples for SHAP calculation (default: 500)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top features to show in report (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load saved models
        logging.info(f"Loading models from: {args.model_dir}")
        model_manager = ModelManager(Path(args.model_dir))
        
        if not model_manager.check_saved_state():
            logging.error(f"No saved models found in: {args.model_dir}")
            sys.exit(1)
        
        training_state = model_manager.load_training_state()
        config = training_state['config']
        
        # Override data path if provided
        if args.data_path:
            config.data_path = args.data_path
        
        # Load training data
        logging.info(f"Loading training data from: {config.data_path}")
        data_loader = DataLoader(config)
        data_loader.scaler = training_state['data_scaler']  # Use saved scaler
        
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = data_loader.load_and_split_data()
        
        # Extract feature importances
        extractor = FeatureImportanceExtractor(config)
        
        logging.info("Extracting feature importances...")
        all_importances = extractor.extract_all_importances(
            training_state['fitted_models'], 
            X_train, 
            X_train_scaled
        )
        
        # Save importances
        csv_path = extractor.save_importances_to_csv(all_importances)
        json_path = extractor.save_importances_to_json(all_importances)
        
        print(f"\n✅ Feature importances saved:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        # Calculate SHAP values if requested
        all_shap_values = None
        if args.calculate_shap:
            if not SHAP_AVAILABLE:
                logging.error("SHAP is not installed. Install with: pip install shap")
            else:
                logging.info(f"Calculating SHAP values (max {args.max_samples} samples)...")
                all_shap_values = extractor.calculate_shap_values(
                    training_state['fitted_models'],
                    X_train,
                    X_train_scaled,
                    y_train,
                    max_samples=args.max_samples
                )
                
                if all_shap_values:
                    shap_path = extractor.save_shap_values(
                        all_shap_values, 
                        list(X_train.columns)
                    )
                    print(f"  SHAP: {shap_path}")
        
        # Create comprehensive report
        report_path = extractor.create_importance_report(
            all_importances, 
            all_shap_values,
            top_n=args.top_n
        )
        print(f"  Report: {report_path}")
        
        # Display top features
        print(f"\n📊 Top {args.top_n} Features by Mean Importance:")
        print("-" * 60)
        
        features = list(next(iter(all_importances.values())).keys())
        mean_importances = {}
        
        for feature in features:
            values = [imp.get(feature, 0) for imp in all_importances.values()]
            mean_importances[feature] = np.mean(values)
        
        sorted_features = sorted(mean_importances.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:args.top_n], 1):
            print(f"{i:3d}. {feature:40s} {importance:.6f}")
        
        print(f"\n✅ All results saved to: {extractor.importances_dir}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
