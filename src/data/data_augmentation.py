"""
Data augmentation utilities for improving model generalization.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import yaml


class DataAugmentation:
    """Class for handling data augmentation operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataAugmentation with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.aug_config = self.config['augmentation']
    
    def create_train_generator(self, X_train: np.ndarray, y_train: np.ndarray, 
                             batch_size: int = 32) -> ImageDataGenerator:
        """
        Create training data generator with augmentation.
        
        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size for training
            
        Returns:
            ImageDataGenerator for training
        """
        train_datagen = ImageDataGenerator(
            rotation_range=self.aug_config['rotation_range'],
            width_shift_range=self.aug_config['width_shift_range'],
            height_shift_range=self.aug_config['height_shift_range'],
            horizontal_flip=self.aug_config['horizontal_flip'],
            zoom_range=self.aug_config['zoom_range'],
            brightness_range=self.aug_config['brightness_range'],
            fill_mode='nearest'
        )
        
        return train_datagen.flow(X_train, y_train, batch_size=batch_size)
    
    def create_val_generator(self, X_val: np.ndarray, y_val: np.ndarray, 
                           batch_size: int = 32) -> ImageDataGenerator:
        """
        Create validation data generator without augmentation.
        
        Args:
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size for validation
            
        Returns:
            ImageDataGenerator for validation
        """
        val_datagen = ImageDataGenerator()
        return val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the entire dataset by a specified factor.
        
        Args:
            X: Input images
            y: Input labels
            augmentation_factor: How many times to augment the dataset
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        train_datagen = ImageDataGenerator(
            rotation_range=self.aug_config['rotation_range'],
            width_shift_range=self.aug_config['width_shift_range'],
            height_shift_range=self.aug_config['height_shift_range'],
            horizontal_flip=self.aug_config['horizontal_flip'],
            zoom_range=self.aug_config['zoom_range'],
            brightness_range=self.aug_config['brightness_range'],
            fill_mode='nearest'
        )
        
        for _ in range(augmentation_factor):
            for i in range(len(X)):
                # Generate augmented image
                img = X[i].reshape(1, *X[i].shape)
                augmented = train_datagen.flow(img, batch_size=1)
                augmented_img = next(augmented)[0]
                
                augmented_images.append(augmented_img)
                augmented_labels.append(y[i])
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset by augmenting the minority class.
        
        Args:
            X: Input images
            y: Input labels
            
        Returns:
            Tuple of (balanced_images, balanced_labels)
        """
        unique, counts = np.unique(y, return_counts=True)
        
        if len(unique) != 2:
            return X, y
        
        # Find minority and majority classes
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_count = np.min(counts)
        majority_count = np.max(counts)
        
        # Calculate augmentation factor needed
        augmentation_factor = majority_count // minority_count
        
        # Separate minority and majority class data
        minority_mask = y == minority_class
        majority_mask = y == majority_class
        
        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]
        
        # Augment minority class
        if augmentation_factor > 1:
            X_minority_aug, y_minority_aug = self.augment_dataset(
                X_minority, y_minority, augmentation_factor - 1
            )
            
            # Combine original and augmented minority data
            X_minority_balanced = np.concatenate([X_minority, X_minority_aug])
            y_minority_balanced = np.concatenate([y_minority, y_minority_aug])
        else:
            X_minority_balanced = X_minority
            y_minority_balanced = y_minority
        
        # Combine balanced minority with majority
        X_balanced = np.concatenate([X_minority_balanced, X_majority])
        y_balanced = np.concatenate([y_minority_balanced, y_majority])
        
        return X_balanced, y_balanced
