"""
Training utilities for cow disease detection models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    LearningRateScheduler, CSVLogger
)
from tensorflow.keras.models import Model
from typing import Dict, List, Tuple, Optional
import yaml
import json
from datetime import datetime


class ModelTrainer:
    """Class for training and managing cow disease detection models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ModelTrainer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.paths_config = self.config['paths']
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for training."""
        directories = [
            self.paths_config['models_dir'],
            self.paths_config['logs_dir'],
            self.paths_config['results_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_callbacks(self, model_name: str) -> List:
        """
        Create training callbacks.
        
        Args:
            model_name: Name of the model for saving
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        if self.training_config['early_stopping']:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_config['patience'],
                verbose=self.training_config['verbose'],
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        if self.training_config['model_checkpoint']:
            checkpoint_path = os.path.join(
                self.paths_config['models_dir'], 
                f"{model_name}_best.h5"
            )
            model_checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=self.training_config['verbose']
            )
            callbacks.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        if self.training_config['reduce_lr_on_plateau']:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=self.training_config['verbose']
            )
            callbacks.append(reduce_lr)
        
        # CSV logger
        log_path = os.path.join(
            self.paths_config['logs_dir'], 
            f"{model_name}_training_log.csv"
        )
        csv_logger = CSVLogger(log_path)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train_model(self, model: Model, train_generator, val_generator, 
                   model_name: str = "cow_disease_model") -> Dict:
        """
        Train the model with the given data generators.
        
        Args:
            model: Keras model to train
            train_generator: Training data generator
            val_generator: Validation data generator
            model_name: Name for saving the model
            
        Returns:
            Training history dictionary
        """
        callbacks = self.create_callbacks(model_name)
        
        # Train the model
        history = model.fit(
            train_generator,
            epochs=self.model_config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=self.training_config['verbose']
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.paths_config['models_dir'], 
            f"{model_name}_final.h5"
        )
        model.save(final_model_path)
        
        # Save training history
        self._save_training_history(history.history, model_name)
        
        return history.history
    
    def train_model_with_data(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             model_name: str = "cow_disease_model") -> Dict:
        """
        Train the model with numpy arrays.
        
        Args:
            model: Keras model to train
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            model_name: Name for saving the model
            
        Returns:
            Training history dictionary
        """
        callbacks = self.create_callbacks(model_name)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            batch_size=self.config['data']['batch_size'],
            epochs=self.model_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=self.training_config['verbose']
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.paths_config['models_dir'], 
            f"{model_name}_final.h5"
        )
        model.save(final_model_path)
        
        # Save training history
        self._save_training_history(history.history, model_name)
        
        return history.history
    
    def _save_training_history(self, history: Dict, model_name: str):
        """
        Save training history to JSON file.
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
        """
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            else:
                history_serializable[key] = value
        
        history_path = os.path.join(
            self.paths_config['results_dir'], 
            f"{model_name}_history.json"
        )
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def plot_training_history(self, history: Dict, model_name: str, save_plot: bool = True):
        """
        Plot and save training history.
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16)
        
        # Plot accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision')
            axes[1, 0].plot(history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall')
            axes[1, 1].plot(history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(
                self.paths_config['results_dir'], 
                f"{model_name}_training_plots.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def load_training_history(self, model_name: str) -> Dict:
        """
        Load training history from JSON file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Training history dictionary
        """
        history_path = os.path.join(
            self.paths_config['results_dir'], 
            f"{model_name}_history.json"
        )
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Training history not found for {model_name}")
    
    def get_best_epoch(self, history: Dict) -> int:
        """
        Get the epoch with the best validation accuracy.
        
        Args:
            history: Training history dictionary
            
        Returns:
            Best epoch number
        """
        val_acc = history['val_accuracy']
        best_epoch = np.argmax(val_acc) + 1
        return best_epoch
