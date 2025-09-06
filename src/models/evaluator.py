"""
Model evaluation utilities for cow disease detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import Model
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime


class ModelEvaluator:
    """Class for comprehensive model evaluation."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize ModelEvaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ['Healthy Cow', 'Lumpy Cow']
        self.results = {}
    
    def evaluate_model(self, model: Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained Keras model
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (for binary classification)
        if len(self.class_names) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        # Average Precision
        avg_precision = average_precision_score(y_test, y_pred_proba[:, 1] if len(self.class_names) == 2 else y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist(),
            'true_labels': y_test.tolist()
        }
        
        return self.results
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        if len(self.class_names) != 2:
            print("ROC curve is only available for binary classification.")
            return
        
        y_true = np.array(self.results['true_labels'])
        y_proba = np.array(self.results['probabilities'])[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = self.results['roc_auc']
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        if len(self.class_names) != 2:
            print("Precision-Recall curve is only available for binary classification.")
            return
        
        y_true = np.array(self.results['true_labels'])
        y_proba = np.array(self.results['probabilities'])[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = self.results['average_precision']
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot class distribution comparison.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # True labels distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar([self.class_names[i] for i in unique_true], counts_true, color='skyblue')
        ax1.set_title('True Labels Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar([self.class_names[i] for i in unique_pred], counts_pred, color='lightcoral')
        ax2.set_title('Predicted Labels Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, model_name: str, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        report = f"""
# Model Evaluation Report: {model_name}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance Metrics
- **Accuracy**: {self.results['accuracy']:.4f}
- **Precision**: {self.results['precision']:.4f}
- **Recall**: {self.results['recall']:.4f}
- **F1-Score**: {self.results['f1_score']:.4f}
- **ROC AUC**: {self.results['roc_auc']:.4f}
- **Average Precision**: {self.results['average_precision']:.4f}

## Classification Report
"""
        
        # Add detailed classification report
        class_report = self.results['classification_report']
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                report += f"""
### {class_name}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1-score']:.4f}
- Support: {metrics['support']}
"""
        
        # Add macro and weighted averages
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            report += f"""
## Macro Averages
- Precision: {macro_avg['precision']:.4f}
- Recall: {macro_avg['recall']:.4f}
- F1-Score: {macro_avg['f1-score']:.4f}
"""
        
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            report += f"""
## Weighted Averages
- Precision: {weighted_avg['precision']:.4f}
- Recall: {weighted_avg['recall']:.4f}
- F1-Score: {weighted_avg['f1-score']:.4f}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def save_results(self, model_name: str, results_dir: str = "results"):
        """
        Save all evaluation results to files.
        
        Args:
            model_name: Name of the model
            results_dir: Directory to save results
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model first.")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(results_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save evaluation report
        report_path = os.path.join(results_dir, f"{model_name}_evaluation_report.md")
        self.generate_evaluation_report(model_name, report_path)
        
        # Save plots
        plot_dir = os.path.join(results_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        self.plot_confusion_matrix(os.path.join(plot_dir, f"{model_name}_confusion_matrix.png"))
        self.plot_roc_curve(os.path.join(plot_dir, f"{model_name}_roc_curve.png"))
        self.plot_precision_recall_curve(os.path.join(plot_dir, f"{model_name}_precision_recall_curve.png"))
        
        print(f"Evaluation results saved to {results_dir}")
    
    def compare_models(self, model_results: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Compare multiple models' performance.
        
        Args:
            model_results: Dictionary with model names as keys and results as values
            save_path: Path to save the comparison plot
        """
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            axes[i].bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
