"""
Tests for data loading functionality.
"""

import pytest
import numpy as np
import os
import tempfile
import cv2
from unittest.mock import patch, MagicMock

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_content = """
data:
  image_size: 150
  batch_size: 32
  test_split: 0.2
  val_split: 0.1
  random_seed: 42
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_image_dirs(self):
        """Create temporary directories with test images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            healthy_dir = os.path.join(temp_dir, "healthy")
            lumpy_dir = os.path.join(temp_dir, "lumpy")
            os.makedirs(healthy_dir)
            os.makedirs(lumpy_dir)
            
            # Create test images
            for i in range(3):
                # Healthy cow image
                healthy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(healthy_dir, f"healthy_{i}.jpg"), healthy_img)
                
                # Lumpy cow image
                lumpy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(lumpy_dir, f"lumpy_{i}.jpg"), lumpy_img)
            
            yield temp_dir
    
    def test_data_loader_initialization(self, temp_config):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_config)
        
        assert loader.image_size == 150
        assert loader.test_split == 0.2
        assert loader.val_split == 0.1
        assert loader.random_seed == 42
        assert loader.class_names == ['Healthy Cow', 'Lumpy Cow']
    
    def test_load_images_from_directory(self, temp_config, temp_image_dirs):
        """Test loading images from directories."""
        loader = DataLoader(temp_config)
        
        healthy_path = os.path.join(temp_image_dirs, "healthy")
        lumpy_path = os.path.join(temp_image_dirs, "lumpy")
        
        X, y = loader.load_images_from_directory(healthy_path, lumpy_path)
        
        assert len(X) == 6  # 3 healthy + 3 lumpy
        assert len(y) == 6
        assert X.shape[1:] == (150, 150, 3)  # Resized images
        assert np.array_equal(np.unique(y), [0, 1])  # Binary labels
        assert np.sum(y == 0) == 3  # 3 healthy cows
        assert np.sum(y == 1) == 3  # 3 lumpy cows
    
    def test_load_images_nonexistent_directory(self, temp_config):
        """Test loading from non-existent directories."""
        loader = DataLoader(temp_config)
        
        X, y = loader.load_images_from_directory("nonexistent", "also_nonexistent")
        
        assert len(X) == 0
        assert len(y) == 0
    
    def test_split_data(self, temp_config):
        """Test data splitting functionality."""
        loader = DataLoader(temp_config)
        
        # Create dummy data
        X = np.random.rand(100, 150, 150, 3)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        # Check shapes
        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert len(y_train) + len(y_val) + len(y_test) == 100
        
        # Check proportions (approximately)
        assert abs(len(X_test) / 100 - 0.2) < 0.05  # 20% test
        assert abs(len(X_val) / 80 - 0.1) < 0.05   # 10% of remaining for val
    
    def test_get_data_info(self, temp_config):
        """Test getting data information."""
        loader = DataLoader(temp_config)
        
        y = np.array([0, 0, 0, 1, 1])
        info = loader.get_data_info(y)
        
        assert info['total_samples'] == 5
        assert info['class_distribution']['Healthy Cow'] == 3
        assert info['class_distribution']['Lumpy Cow'] == 2
        assert info['class_balance'] == 3/2
    
    def test_create_dataframe(self, temp_config):
        """Test creating pandas DataFrame."""
        loader = DataLoader(temp_config)
        
        X = np.random.rand(5, 150, 150, 3)
        y = np.array([0, 0, 1, 1, 0])
        
        df = loader.create_dataframe(X, y)
        
        assert len(df) == 5
        assert 'image' in df.columns
        assert 'label' in df.columns
        assert 'class_name' in df.columns
        assert df['label'].tolist() == y.tolist()
    
    def test_preprocess_image_invalid_path(self, temp_config):
        """Test preprocessing invalid image path."""
        loader = DataLoader(temp_config)
        
        result = loader._load_and_preprocess_image("nonexistent.jpg")
        assert result is None
    
    def test_preprocess_image_valid(self, temp_config, temp_image_dirs):
        """Test preprocessing valid image."""
        loader = DataLoader(temp_config)
        
        image_path = os.path.join(temp_image_dirs, "healthy", "healthy_0.jpg")
        result = loader._load_and_preprocess_image(image_path)
        
        assert result is not None
        assert result.shape == (150, 150, 3)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0


def test_load_kaggle_dataset():
    """Test loading Kaggle dataset."""
    with patch('src.data.data_loader.DataLoader') as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load_images_from_directory.return_value = (np.array([]), np.array([]))
        
        from src.data.data_loader import load_kaggle_dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock Kaggle dataset structure
            healthy_dir = os.path.join(temp_dir, "Healthy")
            lumpy_dir = os.path.join(temp_dir, "Lumpy")
            os.makedirs(healthy_dir)
            os.makedirs(lumpy_dir)
            
            X, y = load_kaggle_dataset(temp_dir)
            
            mock_loader.load_images_from_directory.assert_called_once()
            assert len(X) == 0
            assert len(y) == 0
