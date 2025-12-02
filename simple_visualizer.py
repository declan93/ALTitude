"""
ALTitude Visualizer - Core visualization functions only (no dashboard)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve, precision_recall_curve

warnings.filterwarnings('ignore')


class ALTitudeVisualizer:
    """Core visualization system for ALTitude pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.setup_publication_style()
    
    def setup_publication_style(self):
        """Setup matplotlib for publication-quality figures"""
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.alpha'] = 0.3
    
    def save_figure(self, filename: str):
        """Save figure in multiple formats"""
        output_dir = self.config.output_dir / 'images'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, summary: pd.DataFrame):
        """Plot model performance comparison"""
        if summary is None or summary.empty:
            logging.warning("No summary data for model comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by performance
        summary_sorted = summary.sort_values('Mean ROC AUC', ascending=True)
        
        # Create horizontal bar plot
        colors = ['#2E86AB' if 'SE' not in m else '#A23B72' for m in summary_sorted['Model']]
        bars = ax.barh(range(len(summary_sorted)), summary_sorted['Mean ROC AUC'], color=colors)
        
        # Add error bars if available
        if 'Std' in summary_sorted.columns:
            ax.errorbar(summary_sorted['Mean ROC AUC'], range(len(summary_sorted)),
                       xerr=summary_sorted['Std'], fmt='none', color='black', 
                       capsize=3, linewidth=1)
        
        # Customize plot
        ax.set_yticks(range(len(summary_sorted)))
        ax.set_yticklabels(summary_sorted['Model'])
        ax.set_xlabel('Mean ROC AUC')
        ax.set_title('Model Performance Comparison')
        ax.set_xlim([0, 1])
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Base Models'),
            Patch(facecolor='#A23B72', label='Meta-Learners')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        self.save_figure('model_comparison')
    
    def plot_roc_curves(self, roc_data_dict: Dict, title: str = 'ROC Curves'):
        """Plot ROC curves for multiple models"""
        if not roc_data_dict:
            logging.warning("No ROC data available")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data_dict)))
        
        for i, (name, roc_data) in enumerate(roc_data_dict.items()):
            if len(roc_data) >= 2:
                fpr, tpr = roc_data[:2]
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                       label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        self.save_figure(f'roc_curves_{title.lower().replace(" ", "_")}')
    
    def plot_pr_curves(self, pr_data_dict: Dict, title: str = 'Precision-Recall Curves'):
        """Plot Precision-Recall curves for multiple models"""
        if not pr_data_dict:
            logging.warning("No PR data available")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data_dict)))
        
        for i, (name, pr_data) in enumerate(pr_data_dict.items()):
            if len(pr_data) >= 3:
                precision, recall, avg_precision = pr_data[:3]
                ax.plot(recall, precision, color=colors[i], linewidth=2,
                       label=f'{name} (AP = {avg_precision:.3f})')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        self.save_figure(f'pr_curves_{title.lower().replace(" ", "_")}')
    
    def plot_feature_importance(self, importance_dict: Dict[str, Dict], top_n: int = 20):
        """Plot feature importance comparison"""
        if not importance_dict:
            logging.warning("No feature importance data available")
            return
        
        # Get all features and calculate mean importance
        all_features = set()
        for model_imp in importance_dict.values():
            all_features.update(model_imp.keys())
        
        mean_importance = {}
        for feature in all_features:
            importances = [imp.get(feature, 0) for imp in importance_dict.values()]
            mean_importance[feature] = np.mean(importances)
        
        # Get top features
        top_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        feature_names = [f[0] for f in top_features]
        
        # Create comparison matrix
        n_models = len(importance_dict)
        importance_matrix = np.zeros((top_n, n_models))
        
        model_names = list(importance_dict.keys())
        for j, model in enumerate(model_names):
            for i, feature in enumerate(feature_names):
                importance_matrix[i, j] = importance_dict[model].get(feature, 0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(importance_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=model_names, yticklabels=feature_names,
                   cbar_kws={'label': 'Importance'}, ax=ax)
        
        ax.set_title(f'Top {top_n} Feature Importance Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Feature')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        self.save_figure('feature_importance_comparison')
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = 'Confusion Matrix'):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'],
                   ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        self.save_figure(f'confusion_matrix_{title.lower().replace(" ", "_")}')
    
    def plot_prediction_distribution(self, predictions_dict: Dict[str, np.ndarray]):
        """Plot distribution of predictions"""
        if not predictions_dict:
            logging.warning("No predictions available")
            return
        
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(10, 3*n_models), sharex=True)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, preds) in zip(axes, predictions_dict.items()):
            ax.hist(preds, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} - Prediction Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Prediction Probability')
        
        self.save_figure('prediction_distributions')
    
    def plot_model_correlation(self, predictions_dict: Dict[str, np.ndarray]):
        """Plot correlation heatmap between model predictions"""
        if not predictions_dict or len(predictions_dict) < 2:
            logging.warning("Insufficient models for correlation plot")
            return
        
        # Create correlation matrix
        model_names = list(predictions_dict.keys())
        pred_array = np.column_stack([predictions_dict[name] for name in model_names])
        corr_matrix = np.corrcoef(pred_array.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=model_names, yticklabels=model_names,
                   vmin=-1, vmax=1, center=0, square=True,
                   cbar_kws={'label': 'Correlation'}, ax=ax)
        
        ax.set_title('Model Prediction Correlations')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        self.save_figure('model_correlation_heatmap')
    
    def plot_performance_by_threshold(self, y_true: np.ndarray, y_probs: np.ndarray,
                                     model_name: str = 'Model'):
        """Plot performance metrics across different thresholds"""
        thresholds = np.linspace(0.1, 0.9, 20)
        metrics = {'sensitivity': [], 'specificity': [], 'precision': [], 'f1': []}
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            from sklearn.metrics import confusion_matrix, precision_score, f1_score
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in metrics.items():
            ax.plot(thresholds, values, marker='o', label=metric_name.capitalize())
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Performance by Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add vertical line at default threshold
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        
        self.save_figure(f'threshold_analysis_{model_name.lower().replace(" ", "_")}')
    
    def create_summary_report(self, results: Dict):
        """Create a text summary report of results"""
        report_path = self.config.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ALTITUDE PIPELINE RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Configuration: {self.config.data_path}\n\n")
            
            if 'summary' in results:
                f.write("MODEL PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                summary = results['summary']
                for _, row in summary.iterrows():
                    f.write(f"{row['Model']}: AUC = {row['Mean ROC AUC']:.4f} ± {row.get('Std', 0):.4f}\n")
                f.write("\n")
            
            if 'best_model_name' in results:
                f.write("BEST MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model: {results['best_model_name']}\n")
                f.write(f"AUC: {results['best_model_auc']:.4f}\n\n")
            
            if 'selected_models' in results:
                f.write("SELECTED MODELS FOR ENSEMBLE\n")
                f.write("-" * 40 + "\n")
                for model in results['selected_models']:
                    f.write(f"- {model}\n")
                f.write("\n")
            
            if 'test_results' in results:
                f.write("TEST SET PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                for model, metrics in results['test_results'].items():
                    f.write(f"{model}:\n")
                    f.write(f"  AUC: {metrics['auc']:.4f}\n")
                    f.write(f"  Sensitivity: {metrics['sensitivity']:.4f}\n")
                    f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
        
        logging.info(f"Summary report saved to: {report_path}")
    
    def generate_all_plots(self, results: Dict):
        """Generate all standard plots from results"""
        successful = 0
        failed = []
        
        plot_functions = [
            ('Model Comparison', lambda: self.plot_model_comparison(results.get('summary'))),
            ('ROC Curves - Base', lambda: self.plot_roc_curves(
                results.get('base_results', {}).get('roc_data', {}), 'Base Models ROC')),
            ('PR Curves - Base', lambda: self.plot_pr_curves(
                results.get('base_results', {}).get('pr_data', {}), 'Base Models PR')),
            ('Prediction Distribution', lambda: self.plot_prediction_distribution(
                {k: v for k, v in results.get('base_results', {}).get('oof_predictions', {}).items() 
                 if k.endswith('_prob')})),
            ('Model Correlation', lambda: self.plot_model_correlation(
                {k: v for k, v in results.get('base_results', {}).get('oof_predictions', {}).items() 
                 if k.endswith('_prob')})),
        ]
        
        # Add feature importance if available
        if results.get('feature_importance') and results['feature_importance'].get('importances'):
            plot_functions.append(
                ('Feature Importance', lambda: self.plot_feature_importance(
                    results['feature_importance']['importances']))
            )
        
        # Generate plots
        for name, func in plot_functions:
            try:
                func()
                successful += 1
                logging.info(f"✅ {name} completed")
            except Exception as e:
                failed.append(name)
                logging.warning(f"⚠️ {name} failed: {e}")
        
        # Create summary report
        try:
            self.create_summary_report(results)
            successful += 1
        except Exception as e:
            logging.warning(f"Summary report failed: {e}")
        
        logging.info(f"Visualization complete: {successful} successful, {len(failed)} failed")
        
        return successful, failed
