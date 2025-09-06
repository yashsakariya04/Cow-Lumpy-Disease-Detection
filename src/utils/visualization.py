"""
Visualization utilities for cow disease detection project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


class VisualizationUtils:
    """Utility class for creating various visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualization utilities.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
    
    def plot_data_distribution(self, labels: np.ndarray, class_names: List[str], 
                             title: str = "Data Distribution", save_path: Optional[str] = None):
        """
        Plot the distribution of classes in the dataset.
        
        Args:
            labels: Array of labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar([class_names[i] for i in unique], counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sample_images(self, images: np.ndarray, labels: np.ndarray, 
                          class_names: List[str], num_samples: int = 8,
                          title: str = "Sample Images", save_path: Optional[str] = None):
        """
        Plot sample images from the dataset.
        
        Args:
            images: Array of images
            labels: Array of labels
            class_names: List of class names
            num_samples: Number of samples to display
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique classes
        unique_classes = np.unique(labels)
        samples_per_class = num_samples // len(unique_classes)
        
        fig, axes = plt.subplots(len(unique_classes), samples_per_class, 
                               figsize=(samples_per_class * 3, len(unique_classes) * 3))
        
        if len(unique_classes) == 1:
            axes = axes.reshape(1, -1)
        elif samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for i, class_idx in enumerate(unique_classes):
            class_images = images[labels == class_idx]
            class_name = class_names[class_idx]
            
            for j in range(samples_per_class):
                if j < len(class_images):
                    img = class_images[j]
                    if img.max() <= 1.0:  # Normalized images
                        img = (img * 255).astype(np.uint8)
                    
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{class_name}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_curves(self, history: Dict, metrics: List[str] = None, 
                           save_path: Optional[str] = None):
        """
        Plot training curves for multiple metrics.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in history:
                axes[i].plot(history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history:
                    axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
                
                axes[i].set_title(f'{metric.title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.title())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict], metric: str = 'accuracy',
                            title: str = "Model Comparison", save_path: Optional[str] = None):
        """
        Plot comparison of multiple models.
        
        Args:
            results: Dictionary with model names and their results
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel(metric.title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_maps(self, model, image: np.ndarray, layer_name: str,
                         num_filters: int = 16, save_path: Optional[str] = None):
        """
        Plot feature maps from a specific layer.
        
        Args:
            model: Trained model
            image: Input image
            layer_name: Name of the layer to visualize
            num_filters: Number of filters to display
            save_path: Path to save the plot
        """
        # Create a model that outputs the feature maps
        feature_extractor = plt.Model(inputs=model.input, 
                                    outputs=model.get_layer(layer_name).output)
        
        # Get feature maps
        features = feature_extractor.predict(np.expand_dims(image, axis=0))
        
        # Plot feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_filters, len(axes))):
            axes[i].imshow(features[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_rate_schedule(self, epochs: int, initial_lr: float = 0.001,
                                  decay_factor: float = 0.5, patience: int = 5,
                                  save_path: Optional[str] = None):
        """
        Plot learning rate schedule.
        
        Args:
            epochs: Total number of epochs
            initial_lr: Initial learning rate
            decay_factor: Factor by which to reduce learning rate
            patience: Number of epochs to wait before reducing LR
            save_path: Path to save the plot
        """
        lr_schedule = []
        current_lr = initial_lr
        
        for epoch in range(epochs):
            lr_schedule.append(current_lr)
            if (epoch + 1) % patience == 0:
                current_lr *= decay_factor
        
        plt.figure(figsize=self.figsize)
        plt.plot(range(epochs), lr_schedule)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_plot(self, data_info: Dict, model_info: Dict, 
                          training_history: Dict, save_path: Optional[str] = None):
        """
        Create a comprehensive summary plot.
        
        Args:
            data_info: Dataset information
            model_info: Model information
            training_history: Training history
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Data distribution
        ax1 = plt.subplot(2, 3, 1)
        classes = list(data_info['class_distribution'].keys())
        counts = list(data_info['class_distribution'].values())
        plt.bar(classes, counts)
        plt.title('Data Distribution')
        plt.xticks(rotation=45)
        
        # Training accuracy
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(training_history['accuracy'], label='Training')
        plt.plot(training_history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Training loss
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(training_history['loss'], label='Training')
        plt.plot(training_history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)
        
        # Model parameters
        ax4 = plt.subplot(2, 3, 4)
        param_info = f"Parameters: {model_info.get('number_of_parameters', 'N/A'):,}"
        plt.text(0.1, 0.5, param_info, fontsize=12, transform=ax4.transAxes)
        plt.title('Model Information')
        ax4.axis('off')
        
        # Dataset info
        ax5 = plt.subplot(2, 3, 5)
        dataset_info = f"Total Samples: {data_info['total_samples']}\n"
        dataset_info += f"Class Balance: {data_info['class_balance']:.2f}"
        plt.text(0.1, 0.5, dataset_info, fontsize=12, transform=ax5.transAxes)
        plt.title('Dataset Information')
        ax5.axis('off')
        
        # Best epoch
        ax6 = plt.subplot(2, 3, 6)
        best_epoch = np.argmax(training_history['val_accuracy']) + 1
        best_acc = np.max(training_history['val_accuracy'])
        best_info = f"Best Epoch: {best_epoch}\nBest Accuracy: {best_acc:.3f}"
        plt.text(0.1, 0.5, best_info, fontsize=12, transform=ax6.transAxes)
        plt.title('Training Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
