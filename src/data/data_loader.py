"""
Data loading and preprocessing utilities for Cow Lumpy Disease Detection.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Tuple, List, Optional
import yaml


class DataLoader:
    """Class for loading and preprocessing cow disease detection data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.image_size = self.config['data']['image_size']
        self.test_split = self.config['data']['test_split']
        self.val_split = self.config['data']['val_split']
        self.random_seed = self.config['data']['random_seed']
        
        # Class mapping
        self.class_names = ['Healthy Cow', 'Lumpy Cow']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
    def load_images_from_directory(self, healthy_path: str, lumpy_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from directories and create labels.
        
        Args:
            healthy_path: Path to healthy cow images
            lumpy_path: Path to lumpy cow images
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        # Load healthy cow images
        if os.path.exists(healthy_path):
            for filename in os.listdir(healthy_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(healthy_path, filename)
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Healthy cow
        
        # Load lumpy cow images
        if os.path.exists(lumpy_path):
            for filename in os.listdir(lumpy_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(lumpy_path, filename)
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Lumpy cow
        
        return np.array(images), np.array(labels)
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if loading fails
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Image data
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.random_seed, stratify=y
        )
        
        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_split/(1-self.test_split), 
            random_state=self.random_seed, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_info(self, y: np.ndarray) -> dict:
        """
        Get information about the dataset.
        
        Args:
            y: Labels array
            
        Returns:
            Dictionary with dataset information
        """
        unique, counts = np.unique(y, return_counts=True)
        info = {
            'total_samples': len(y),
            'class_distribution': dict(zip([self.class_names[i] for i in unique], counts)),
            'class_balance': counts[0] / counts[1] if len(counts) > 1 else 1.0
        }
        return info
    
    def create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Create a pandas DataFrame with image paths and labels.
        
        Args:
            X: Image data
            y: Labels
            
        Returns:
            DataFrame with image data and labels
        """
        df = pd.DataFrame({
            'image': [f"image_{i}.jpg" for i in range(len(X))],
            'label': y,
            'class_name': [self.class_names[label] for label in y]
        })
        return df


def load_kaggle_dataset(dataset_path: str, config_path: str = "config/config.yaml") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from Kaggle download path.
    
    Args:
        dataset_path: Path to the downloaded Kaggle dataset
        config_path: Path to configuration file
        
    Returns:
        Tuple of (images, labels)
    """
    loader = DataLoader(config_path)
    
    # Common Kaggle dataset structure
    healthy_path = os.path.join(dataset_path, "Healthy")
    lumpy_path = os.path.join(dataset_path, "Lumpy")
    
    # Alternative paths if the above don't exist
    if not os.path.exists(healthy_path):
        healthy_path = os.path.join(dataset_path, "healthy")
    if not os.path.exists(lumpy_path):
        lumpy_path = os.path.join(dataset_path, "lumpy")
    
    return loader.load_images_from_directory(healthy_path, lumpy_path)
