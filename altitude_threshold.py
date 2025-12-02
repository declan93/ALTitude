"""
Threshold Optimization Utilities for ALT Detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import logging


class ThresholdOptimizer:
    """Optimize classification threshold for ALT detection"""
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, 
                              optimization_metric: str = 'balanced',
                              min_sensitivity: Optional[float] = None,
                              min_specificity: Optional[float] = None) -> Dict:
        """
        Find optimal threshold based on different criteria
        
        Args:
            y_true: True labels (0/1)
            y_probs: Predicted probabilities
            optimization_metric: One of:
                - 'balanced': Balance between sensitivity and specificity (Youden's J)
                - 'sensitivity': Maximize sensitivity (minimize false negatives)
                - 'specificity': Maximize specificity (minimize false positives)
                - 'f1': Maximize F1 score
                - 'alt_focused': Custom for ALT - prioritize sensitivity with min specificity
            min_sensitivity: Minimum required sensitivity (optional)
            min_specificity: Minimum required specificity (optional)
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        
        # Get ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        
        # Calculate metrics for each threshold
        results = []
        for i, thresh in enumerate(thresholds):
            if thresh == np.inf or thresh == -np.inf:
                continue
                
            y_pred = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = f1_score(y_true, y_pred)
            
            # Skip if doesn't meet minimum requirements
            if min_sensitivity and sensitivity < min_sensitivity:
                continue
            if min_specificity and specificity < min_specificity:
                continue
            
            results.append({
                'threshold': thresh,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1': f1,
                'youden_j': sensitivity + specificity - 1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        if not results:
            logging.warning("No thresholds meet the minimum requirements")
            # Return default threshold
            return {'threshold': 0.5, 'message': 'Using default - no valid thresholds found'}
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(results)
        
        # Find optimal based on metric
        if optimization_metric == 'balanced':
            # Youden's J statistic - balances sensitivity and specificity
            optimal_idx = df['youden_j'].idxmax()
            metric_used = "Youden's J (balanced)"
            
        elif optimization_metric == 'sensitivity':
            # Maximize sensitivity (catch all ALT+ cases)
            optimal_idx = df['sensitivity'].idxmax()
            metric_used = "Maximum sensitivity"
            
        elif optimization_metric == 'specificity':
            # Maximize specificity (minimize false positives)
            optimal_idx = df['specificity'].idxmax()
            metric_used = "Maximum specificity"
            
        elif optimization_metric == 'f1':
            # Maximize F1 score
            optimal_idx = df['f1'].idxmax()
            metric_used = "Maximum F1 score"
            
        elif optimization_metric == 'alt_focused':
            # Custom for ALT: Prioritize sensitivity ≥ 0.80, then maximize specificity
            high_sens = df[df['sensitivity'] >= 0.80]
            if len(high_sens) > 0:
                optimal_idx = high_sens['specificity'].idxmax()
                metric_used = "ALT-focused (≥80% sensitivity)"
            else:
                # If can't achieve 80% sensitivity, get closest
                optimal_idx = df['sensitivity'].idxmax()
                metric_used = "ALT-focused (max sensitivity available)"
        else:
            # Default to balanced
            optimal_idx = df['youden_j'].idxmax()
            metric_used = "Youden's J (default)"
        
        optimal = df.iloc[optimal_idx].to_dict()
        optimal['optimization_metric'] = metric_used
        optimal['n_thresholds_tested'] = len(results)
        
        return optimal
    
    @staticmethod
    def analyze_threshold_range(y_true: np.ndarray, y_probs: np.ndarray,
                               threshold_range: Optional[Tuple[float, float]] = None,
                               n_points: int = 20) -> pd.DataFrame:
        """
        Analyze performance across a range of thresholds
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities
            threshold_range: (min, max) thresholds to test (default: 0.1 to 0.9)
            n_points: Number of thresholds to test
            
        Returns:
            DataFrame with metrics for each threshold
        """
        
        if threshold_range is None:
            threshold_range = (0.1, 0.9)
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        results = []
        
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except:
                continue
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Calculate positive and negative rates
            n_positive = (y_true == 1).sum()
            n_negative = (y_true == 0).sum()
            
            results.append({
                'threshold': thresh,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1': f1_score(y_true, y_pred),
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'positive_rate': (y_pred == 1).sum() / len(y_pred),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'alt_positive_found': f"{tp}/{n_positive}",
                'alt_negative_correct': f"{tn}/{n_negative}"
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_threshold_analysis(y_true: np.ndarray, y_probs: np.ndarray,
                               optimal_threshold: float = None,
                               save_path: Optional[str] = None):
        """
        Create visualization of threshold effects on metrics
        """
        # Get threshold analysis
        df = ThresholdOptimizer.analyze_threshold_range(y_true, y_probs)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Sensitivity and Specificity
        ax = axes[0, 0]
        ax.plot(df['threshold'], df['sensitivity'], 'b-', label='Sensitivity (TPR)', linewidth=2)
        ax.plot(df['threshold'], df['specificity'], 'r-', label='Specificity (TNR)', linewidth=2)
        if optimal_threshold:
            ax.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Sensitivity vs Specificity by Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Plot 2: F1 Score and Accuracy
        ax = axes[0, 1]
        ax.plot(df['threshold'], df['f1'], 'g-', label='F1 Score', linewidth=2)
        ax.plot(df['threshold'], df['accuracy'], 'm-', label='Accuracy', linewidth=2)
        if optimal_threshold:
            ax.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('F1 Score and Accuracy by Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Plot 3: Precision and Positive Rate
        ax = axes[1, 0]
        ax.plot(df['threshold'], df['precision'], 'c-', label='Precision (PPV)', linewidth=2)
        ax.plot(df['threshold'], df['positive_rate'], 'orange', label='Positive Rate', linewidth=2)
        if optimal_threshold:
            ax.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Precision and Positive Prediction Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Plot 4: Confusion Matrix Components
        ax = axes[1, 1]
        ax.plot(df['threshold'], df['true_positives'], 'g-', label='True Positives', linewidth=2)
        ax.plot(df['threshold'], df['false_positives'], 'r-', label='False Positives', linewidth=2)
        ax.plot(df['threshold'], df['false_negatives'], 'orange', label='False Negatives', linewidth=2)
        if optimal_threshold:
            ax.axvline(optimal_threshold, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Count')
        ax.set_title('Confusion Matrix Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        
        plt.suptitle('Threshold Analysis for ALT Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Threshold analysis plot saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def print_threshold_comparison(y_true: np.ndarray, y_probs: np.ndarray,
                                  thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7]):
        """
        Print detailed comparison of different thresholds for ALT detection
        """
        print("\n" + "=" * 80)
        print("THRESHOLD COMPARISON FOR ALT DETECTION")
        print("=" * 80)
        
        # Get total counts
        n_alt_positive = (y_true == 1).sum()
        n_alt_negative = (y_true == 0).sum()
        n_total = len(y_true)
        
        print(f"\nDataset: {n_alt_positive} ALT+ cases, {n_alt_negative} ALT- cases (Total: {n_total})")
        print(f"ALT+ prevalence: {n_alt_positive/n_total:.1%}\n")
        
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = f1_score(y_true, y_pred)
            
            print(f"Threshold = {thresh:.2f}:")
            print(f"  Sensitivity: {sensitivity:.3f} ({tp}/{n_alt_positive} ALT+ detected)")
            print(f"  Specificity: {specificity:.3f} ({tn}/{n_alt_negative} ALT- correct)")
            print(f"  Precision:   {precision:.3f} ({tp}/{tp+fp} positive predictions correct)")
            print(f"  F1 Score:    {f1:.3f}")
            print(f"  False Negatives: {fn} ALT+ cases missed")
            print(f"  False Positives: {fp} ALT- cases misclassified")
            print("-" * 40)
        
        # Find and report optimal thresholds for different scenarios
        print("\nOPTIMAL THRESHOLDS FOR DIFFERENT PRIORITIES:")
        print("=" * 60)
        
        scenarios = [
            ('balanced', None, None, "Balanced sensitivity/specificity"),
            ('sensitivity', None, None, "Maximum sensitivity (catch all ALT+)"),
            ('alt_focused', None, None, "ALT-focused (≥80% sensitivity)"),
            ('f1', None, None, "Maximum F1 score"),
            ('balanced', 0.7, None, "Balanced with min 70% sensitivity"),
            ('balanced', None, 0.8, "Balanced with min 80% specificity")
        ]
        
        for opt_metric, min_sens, min_spec, description in scenarios:
            optimal = ThresholdOptimizer.find_optimal_threshold(
                y_true, y_probs, opt_metric, min_sens, min_spec
            )
            
            if 'threshold' in optimal:
                print(f"\n{description}:")
                print(f"  Optimal threshold: {optimal['threshold']:.3f}")
                if 'sensitivity' in optimal:
                    print(f"  Sensitivity: {optimal['sensitivity']:.3f}")
                    print(f"  Specificity: {optimal['specificity']:.3f}")
                    print(f"  F1 Score: {optimal['f1']:.3f}")
        
        print("\n" + "=" * 80)
