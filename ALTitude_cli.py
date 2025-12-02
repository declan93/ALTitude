#!/usr/bin/env python3
"""
ALTitude CLI - Machine Learning Pipeline for ALT Prediction with Threshold Optimization
Usage:
    python ALTitude-cli.py train <data_path> [options]
    python ALTitude-cli.py predict <data_path> --model-dir <path>
    python ALTitude-cli.py evaluate --model-dir <path>
"""

import argparse
import sys
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import pipeline components
from altitude_core import MLPipeline, Config
from altitude_model_manager import ModelManager
from altitude_utils import setup_logging, validate_file_path
from simple_visualizer import ALTitudeVisualizer
from altitude_threshold import ThresholdOptimizer


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ALTitude - Machine Learning Pipeline for ALT Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python ALTitude-cli.py train data/training_data.txt
  
  # Train with Leave-One-Out CV
  python ALTitude-cli.py train data/training_data.txt --loo-cv
  
  # Train with threshold optimization for ALT detection
  python ALTitude-cli.py train data/training_data.txt --optimize-threshold --threshold-metric alt_focused
  
  # Train with custom configuration
  python ALTitude-cli.py train data/training_data.txt --config config.json
  
  # Make predictions with custom threshold
  python ALTitude-cli.py predict data/new_data.txt --model-dir ./results/ --threshold 0.4
  
  # Test multiple thresholds
  python ALTitude-cli.py predict data/new_data.txt --model-dir ./results/ --multi-threshold "0.3,0.4,0.5,0.6,0.7"
  
  # Evaluate saved models
  python ALTitude-cli.py evaluate --model-dir ./results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the machine learning models')
    train_parser.add_argument('data_path', type=str, help='Path to training data file')
    train_parser.add_argument('--output-dir', type=str, default='./results', 
                             help='Output directory (default: ./results)')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    train_parser.add_argument('--test-size', type=float, default=0.3, 
                             help='Test set proportion (default: 0.3)')
    train_parser.add_argument('--cv-folds', type=int, default=5, 
                             help='Number of CV folds (default: 5)')
    
    # LOO CV arguments
    train_parser.add_argument('--loo-cv', action='store_true',
                             help='Use Leave-One-Out cross-validation instead of k-fold')
    train_parser.add_argument('--loo-max-samples', type=int, default=500,
                             help='Maximum samples for LOO CV before falling back to k-fold (default: 500)')
    
    # Feature selection arguments
    train_parser.add_argument('--feature-selection', action='store_true',
                             help='Enable feature selection')
    train_parser.add_argument('--n-features', type=int, default=20,
                             help='Number of features to select (default: 20)')
    
    # Threshold optimization arguments
    train_parser.add_argument('--optimize-threshold', action='store_true',
                             help='Find optimal classification threshold')
    train_parser.add_argument('--threshold-metric', type=str, default='balanced',
                             choices=['balanced', 'sensitivity', 'specificity', 'f1', 'alt_focused'],
                             help='Metric to optimize threshold (default: balanced)')
    train_parser.add_argument('--min-sensitivity', type=float, default=None,
                             help='Minimum required sensitivity when optimizing threshold')
    train_parser.add_argument('--min-specificity', type=float, default=None,
                             help='Minimum required specificity when optimizing threshold')
    train_parser.add_argument('--analyze-thresholds', action='store_true',
                             help='Generate detailed threshold analysis plots and reports')
    
    train_parser.add_argument('--no-plots', action='store_true',
                             help='Skip visualization generation')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('data_path', type=str, help='Path to data file')
    predict_parser.add_argument('--model-dir', type=str, required=True,
                               help='Directory containing saved models')
    predict_parser.add_argument('--output-file', type=str, 
                               help='Output file for predictions')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                               help='Decision threshold for classification (default: 0.5)')
    predict_parser.add_argument('--multi-threshold', type=str, default=None,
                               help='Comma-separated list of thresholds to test (e.g., "0.3,0.5,0.7")')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate saved models')
    eval_parser.add_argument('--model-dir', type=str, default='./results',
                            help='Directory containing saved models (default: ./results)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    try:
        if args.command == 'train':
            run_training(args)
        elif args.command == 'predict':
            run_prediction(args)
        elif args.command == 'evaluate':
            run_evaluation(args)
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


def run_training(args):
    """Run the training pipeline with comprehensive results logging"""
    logging.info("=" * 60)
    logging.info("STARTING TRAINING PIPELINE")
    logging.info("=" * 60)

    # Validate input
    if not validate_file_path(args.data_path):
        raise FileNotFoundError(f"Training data not found: {args.data_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize results logger
    from altitude_results_logger import ResultsLogger
    results_logger = ResultsLogger(output_dir)

    # Create configuration
    if args.config and Path(args.config).exists():
        logging.info(f"Loading configuration from: {args.config}")
        config = Config.from_file(args.config)
        # Override with command line arguments
        config.data_path = args.data_path
        config.output_dir = output_dir
    else:
        # Build config dict with only valid parameters
        config_params = {
            'data_path': args.data_path,
            'output_dir': output_dir,
            'random_seed': args.seed,
            'test_size': args.test_size,
            'cv_folds': args.cv_folds,
            'use_feature_selection': args.feature_selection,
            'n_features_to_select': args.n_features
        }

        # Add LOO parameters if they exist
        if hasattr(args, 'loo_cv'):
            config_params['use_loo_cv'] = args.loo_cv
        if hasattr(args, 'loo_max_samples'):
            config_params['loo_max_samples'] = args.loo_max_samples

        config = Config(**config_params)

    # Log configuration
    results_logger.log_configuration(config)

    # Save configuration
    config_path = output_dir / "config.json"
    config.to_file(str(config_path))
    logging.info(f"Configuration saved to: {config_path}")

    # Log settings
    logging.info("Training Configuration:")
    logging.info(f"  Data: {args.data_path}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Test size: {config.test_size}")
    logging.info(f"  Random seed: {config.random_seed}")

    if hasattr(config, 'use_loo_cv') and config.use_loo_cv:
        logging.info(f"  Leave-One-Out CV: Enabled (max {config.loo_max_samples} samples)")
    else:
        logging.info(f"  Regular CV: {config.cv_folds} folds")

    if config.use_feature_selection:
        logging.info(f"  Feature selection: Top {config.n_features_to_select} features")

    # Train models
    pipeline = MLPipeline(config)
    results = pipeline.train()

    if results is None:
        raise RuntimeError("Training failed")

    # Log data information
    if 'X_test' in results and 'y_test' in results:
        # Get class distributions
        y_train_dist = {}
        y_test_dist = {}

        # For training set (need to reconstruct from test indices)
        if 'all_results' in results and results['all_results']:
            # Get training size from OOF predictions
            train_size = len(results['all_results'][0]['oof_preds'])
            y_test_size = len(results['y_test'])

            # Get test distribution
            for val in results['y_test'].unique():
                y_test_dist[int(val)] = int((results['y_test'] == val).sum())

        # Get feature names if available
        feature_names = None
        if hasattr(pipeline.data_loader, 'feature_names'):
            feature_names = pipeline.data_loader.feature_names

        results_logger.log_data_info(
            X_train_shape=(train_size if 'train_size' in locals() else 0,
                           len(feature_names) if feature_names else results['X_test'].shape[1]),
            X_test_shape=results['X_test'].shape,
            y_train_dist=y_train_dist,
            y_test_dist=y_test_dist,
            feature_names=feature_names
        )

    # Log CV results for each model
    if 'all_results' in results:
        for model_result in results['all_results']:
            cv_type = model_result.get('cv_type', 'k-fold')
            results_logger.log_cv_results(
                model_name=model_result['name'],
                cv_scores=np.array(model_result['cv_scores']),
                oof_predictions=model_result.get('oof_preds'),
                cv_type=cv_type
            )

    # Log test results
    if 'test_results' in results:
        for model_name, metrics in results['test_results'].items():
            results_logger.log_test_results(model_name, metrics)

    # Log model summary
    if 'summary' in results:
        results_logger.log_model_summary(results['summary'])

    # Log ensemble information
    if 'selected_models' in results:
        results_logger.log_ensemble_info(
            selected_models=results['selected_models'],
            meta_learner_results=results.get('meta_results', {})
        )

    # THRESHOLD OPTIMIZATION
    optimal_thresholds = {}
    if args.optimize_threshold and 'y_test' in results and 'test_predictions' in results['base_results']:
        logging.info("\n" + "=" * 60)
        logging.info("OPTIMIZING CLASSIFICATION THRESHOLDS")
        logging.info("=" * 60)

        from altitude_threshold import ThresholdOptimizer
        optimizer = ThresholdOptimizer()

        y_test = results['y_test']
        test_preds = results['base_results']['test_predictions']

        # Find optimal threshold for each model
        for y_probs_key in test_preds.keys():
            if y_probs_key.endswith('_prob'):
                model_display_name = y_probs_key.replace('_prob', '')
                y_probs = test_preds[y_probs_key]

                optimal = optimizer.find_optimal_threshold(
                    y_test, y_probs,
                    optimization_metric=args.threshold_metric,
                    min_sensitivity=args.min_sensitivity,
                    min_specificity=args.min_specificity
                )

                optimal_thresholds[model_display_name] = optimal

                logging.info(f"\n{model_display_name}:")
                logging.info(f"  Optimal threshold: {optimal['threshold']:.3f}")
                logging.info(f"  Sensitivity: {optimal['sensitivity']:.3f}")
                logging.info(f"  Specificity: {optimal['specificity']:.3f}")
                logging.info(f"  F1 Score: {optimal['f1']:.3f}")

        # Log threshold optimization results
        results_logger.log_threshold_optimization(optimal_thresholds)

        # Save optimal thresholds
        threshold_file = output_dir / 'optimal_thresholds.json'
        with open(threshold_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_thresholds = {}
            for model, data in optimal_thresholds.items():
                serializable_thresholds[model] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in data.items()
                }
            json.dump(serializable_thresholds, f, indent=2)
        logging.info(f"\n✅ Optimal thresholds saved to: {threshold_file}")

        # Generate threshold analysis if requested
        if args.analyze_thresholds:
            logging.info("\nGenerating threshold analysis...")

            # Create images directory if it doesn't exist
            images_dir = output_dir / 'images'
            images_dir.mkdir(exist_ok=True, parents=True)

            # Analyze best model
            best_model_name = results['best_model_name']
            if f'{best_model_name}_prob' in test_preds:
                y_probs = test_preds[f'{best_model_name}_prob']

                # Print comparison table
                optimizer.print_threshold_comparison(
                    y_test, y_probs,
                    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                )

                # Generate plot
                best_threshold = optimal_thresholds.get(best_model_name, {}).get('threshold', 0.5)
                fig = optimizer.plot_threshold_analysis(
                    y_test, y_probs,
                    optimal_threshold=best_threshold,
                    save_path=images_dir / f'{best_model_name}_threshold_analysis.png'
                )
                plt.close(fig)

                # Generate detailed report
                analysis_df = optimizer.analyze_threshold_range(y_test, y_probs, n_points=30)
                analysis_df.to_csv(output_dir / f'{best_model_name}_threshold_analysis.csv', index=False)

                logging.info(f"✅ Threshold analysis saved for {best_model_name}")

    # SAVE ALL RESULTS TO FILES
    logging.info("\n" + "=" * 60)
    logging.info("SAVING COMPREHENSIVE RESULTS")
    logging.info("=" * 60)

    saved_files = results_logger.save_results()

    logging.info("\n📁 Results files saved:")
    for file_type, file_path in saved_files.items():
        if file_path:
            logging.info(f"  {file_type}: {file_path}")

    # Save models
    logging.info("\nSaving trained models...")
    model_manager = ModelManager(output_dir)
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

    # Generate visualizations
    if not args.no_plots:
        logging.info("Generating visualizations...")
        visualizer = ALTitudeVisualizer(config)
        successful, failed = visualizer.generate_all_plots(results)
        logging.info(f"Visualizations: {successful} successful, {len(failed)} failed")

    # Print summary
    print_training_summary(results, output_dir, optimal_thresholds if args.optimize_threshold else None)

    # Print final message with results location
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey files:")
    print(f"  📄 Latest report: {output_dir}/training_results/latest_report.txt")
    print(f"  📊 Latest results: {output_dir}/training_results/latest_results.json")
    print(f"  📈 Model summary: {output_dir}/model_summary.csv")
    if args.optimize_threshold:
        print(f"  🎯 Optimal thresholds: {output_dir}/optimal_thresholds.json")
    print("\n✅ Training pipeline completed successfully!")
    print("=" * 60)

def run_prediction(args):
    """Run prediction using saved models with custom threshold(s)"""
    logging.info("=" * 60)
    logging.info("STARTING PREDICTION")
    logging.info("=" * 60)
    
    # Validate inputs
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not validate_file_path(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Load models
    logging.info(f"Loading models from: {model_dir}")
    model_manager = ModelManager(model_dir)
    
    if not model_manager.check_saved_state():
        raise RuntimeError(f"No saved models found in: {model_dir}")
    
    training_state = model_manager.load_training_state()
    
    # Create pipeline and load state
    pipeline = MLPipeline(training_state['config'])
    pipeline.load_state(training_state)
    
    # Check for optimal thresholds file
    optimal_threshold_file = model_dir / 'optimal_thresholds.json'
    optimal_thresholds = {}
    if optimal_threshold_file.exists() and not args.multi_threshold:
        with open(optimal_threshold_file, 'r') as f:
            optimal_thresholds = json.load(f)
        logging.info(f"Loaded optimal thresholds from: {optimal_threshold_file}")
    
    # Determine thresholds to use
    if args.multi_threshold:
        thresholds = [float(t.strip()) for t in args.multi_threshold.split(',')]
        logging.info(f"Using multiple thresholds: {thresholds}")
    else:
        thresholds = [args.threshold]
        if args.threshold == 0.5 and optimal_thresholds:
            logging.info("Threshold 0.5 (default) specified, but optimal thresholds available.")
            logging.info("Consider using optimal thresholds for better ALT detection performance.")
    
    # Make predictions for each threshold
    all_predictions = {}
    for threshold in thresholds:
        logging.info(f"\nMaking predictions with threshold = {threshold:.3f}")
        predictions_df = pipeline.predict(args.data_path, threshold=threshold)
        
        # Apply model-specific optimal thresholds if available and using default threshold
        if optimal_thresholds and threshold == args.threshold and len(thresholds) == 1:
            for model_name, opt_data in optimal_thresholds.items():
                if isinstance(opt_data, dict) and 'threshold' in opt_data:
                    opt_thresh = opt_data['threshold']
                    prob_col = f'{model_name}_prob'
                    pred_col = f'{model_name}_pred'
                    
                    if prob_col in predictions_df.columns:
                        # Add optimal threshold predictions
                        predictions_df[f'{pred_col}_opt'] = (
                            predictions_df[prob_col] >= opt_thresh
                        ).astype(int)
                        logging.info(f"  Applied optimal threshold {opt_thresh:.3f} for {model_name}")
        
        all_predictions[threshold] = predictions_df
        
        # Save predictions
        if len(thresholds) > 1:
            output_file = model_dir / f"{Path(args.data_path).stem}_predictions_t{int(threshold*100)}.csv"
        else:
            if args.output_file:
                output_file = Path(args.output_file)
            else:
                input_name = Path(args.data_path).stem
                output_file = model_dir / f"{input_name}_predictions.csv"
        
        predictions_df.to_csv(output_file, index=False)
        logging.info(f"Predictions saved to: {output_file}")
    
    # Print comparison if multiple thresholds
    if len(thresholds) > 1:
        print_threshold_comparison_summary(all_predictions)
    
    # Print summary
    print_prediction_summary(predictions_df, threshold if len(thresholds) == 1 else thresholds)


def run_evaluation(args):
    """Evaluate saved models"""
    logging.info("=" * 60)
    logging.info("MODEL EVALUATION")
    logging.info("=" * 60)
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_manager = ModelManager(model_dir)
    
    # Check model validity
    logging.info("Validating saved models...")
    validation_results = model_manager.validate_models()
    
    all_valid = all(validation_results.values())
    
    print("\nModel Validation Results:")
    print("-" * 40)
    for component, is_valid in validation_results.items():
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {component}: {status}")
    
    if not all_valid:
        print("\n⚠️ Some model components are invalid!")
        return
    
    # Get model info
    model_info = model_manager.get_model_info()
    
    if model_info:
        print("\nModel Information:")
        print("-" * 40)
        print(f"  Training date: {model_info.get('timestamp', 'Unknown')}")
        print(f"  Base models: {model_info.get('n_base_models', 'Unknown')}")
        print(f"  Meta-learners: {model_info.get('n_meta_learners', 'Unknown')}")
        
        if 'best_model' in model_info:
            print(f"  Best model: {model_info['best_model']}")
            print(f"  Best AUC: {model_info['best_auc']:.4f}")
        
        if 'selected_models' in model_info:
            print(f"\n  Selected models for ensemble:")
            for model in model_info['selected_models']:
                print(f"    - {model}")
    
    # Check for optimal thresholds
    optimal_threshold_file = model_dir / 'optimal_thresholds.json'
    if optimal_threshold_file.exists():
        with open(optimal_threshold_file, 'r') as f:
            optimal_thresholds = json.load(f)
        
        print("\nOptimal Thresholds:")
        print("-" * 40)
        for model_name, opt_data in optimal_thresholds.items():
            if isinstance(opt_data, dict) and 'threshold' in opt_data:
                print(f"  {model_name}:")
                print(f"    Threshold: {opt_data['threshold']:.3f}")
                if 'sensitivity' in opt_data:
                    print(f"    Sensitivity: {opt_data['sensitivity']:.3f}")
                    print(f"    Specificity: {opt_data['specificity']:.3f}")
    
    # Load and display summary
    summary_path = model_dir / "model_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print("\nModel Performance Summary:")
        print("-" * 40)
        print(summary.to_string(index=False))
    
    # Export detailed report
    report_path = model_manager.export_model_summary()
    print(f"\n📄 Detailed report saved to: {report_path}")


def print_training_summary(results: dict, output_dir: Path, optimal_thresholds: dict = None):
    """Print training summary to console"""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    print(f"\nBest Model: {results['best_model_name']}")
    print(f"Best AUC: {results['best_model_auc']:.4f}")
    
    print(f"\nModels saved to: {output_dir}")
    
    # Top 5 models
    print("\nTop 5 Models:")
    print("-" * 40)
    summary = results['summary']
    top_5 = summary.nlargest(5, 'Mean ROC AUC')
    for _, row in top_5.iterrows():
        cv_type = row.get('CV Type', 'k-fold')
        print(f"  {row['Model']}: {row['Mean ROC AUC']:.4f} ± {row.get('Std', 0):.4f} ({cv_type})")
    
    # Test set performance
    if 'test_results' in results and results['test_results']:
        print("\nTest Set Performance:")
        print("-" * 40)
        for model, metrics in list(results['test_results'].items())[:5]:
            print(f"  {model}:")
            print(f"    AUC: {metrics['auc']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
            
            # Add optimal threshold info if available
            if optimal_thresholds and model in optimal_thresholds:
                opt = optimal_thresholds[model]
                if 'threshold' in opt:
                    print(f"    Optimal threshold: {opt['threshold']:.3f}")
    
    print("\n✅ Training pipeline completed successfully!")
    print("=" * 60)


def print_prediction_summary(predictions_df: pd.DataFrame, threshold):
    """Print prediction summary to console"""
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(predictions_df)}")
    
    if isinstance(threshold, list):
        print(f"Thresholds tested: {threshold}")
    else:
        print(f"Threshold: {threshold}")
    
    # Count predictions
    pred_cols = [col for col in predictions_df.columns if col.endswith('_pred')]
    prob_cols = [col for col in predictions_df.columns if col.endswith('_prob')]
    opt_cols = [col for col in predictions_df.columns if col.endswith('_pred_opt')]
    
    print(f"Prediction columns: {len(pred_cols)}")
    print(f"Probability columns: {len(prob_cols)}")
    if opt_cols:
        print(f"Optimal threshold columns: {len(opt_cols)}")
    
    # Show meta-learner predictions
    meta_pred_cols = [col for col in pred_cols if col.startswith('SE_')]
    if meta_pred_cols:
        print("\nMeta-learner Predictions:")
        print("-" * 40)
        for col in meta_pred_cols[:3]:  # Show first 3
            model_name = col.replace('SE_', '').replace('_pred', '')
            pos_count = predictions_df[col].sum()
            pos_pct = (pos_count / len(predictions_df)) * 100
            print(f"  {model_name}: {pos_count}/{len(predictions_df)} ({pos_pct:.1f}%) ALT+")
    
    print("\n✅ Predictions completed successfully!")
    print("=" * 60)


def print_threshold_comparison_summary(all_predictions: dict):
    """Print comparison of predictions at different thresholds"""
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON SUMMARY")
    print("=" * 60)
    
    for threshold, preds_df in all_predictions.items():
        print(f"\nThreshold = {threshold:.3f}:")
        pred_cols = [col for col in preds_df.columns if col.endswith('_pred') and not col.endswith('_pred_opt')]
        
        # Show first 3 models and meta-learners
        shown_models = 0
        for col in pred_cols:
            if shown_models >= 3:
                break
            model_name = col.replace('_pred', '')
            positive_count = preds_df[col].sum()
            positive_pct = (positive_count / len(preds_df)) * 100
            print(f"  {model_name}: {positive_count}/{len(preds_df)} ({positive_pct:.1f}%) predicted ALT+")
            shown_models += 1
        
        # Always show meta-learners if present
        for col in pred_cols:
            if col.startswith('SE_'):
                model_name = col.replace('_pred', '')
                positive_count = preds_df[col].sum()
                positive_pct = (positive_count / len(preds_df)) * 100
                print(f"  {model_name}: {positive_count}/{len(preds_df)} ({positive_pct:.1f}%) predicted ALT+")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
