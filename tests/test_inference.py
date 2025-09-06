"""
Tests for inference functionality.
"""

import pytest
import numpy as np
import tempfile
import os
import cv2
from unittest.mock import patch, MagicMock

from src.inference import CowDiseasePredictor


class TestCowDiseasePredictor:
    """Test cases for CowDiseasePredictor class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_content = """
data:
  image_size: 150
  batch_size: 32
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_model(self):
        """Create a temporary model file for testing."""
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            # Create a mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[0.3, 0.7]])  # Mock prediction
            mock_load_model.return_value = mock_model
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                model_path = f.name
            
            yield model_path, mock_model
            os.unlink(model_path)
    
    @pytest.fixture
    def temp_image(self):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # Create a dummy image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            yield f.name
        os.unlink(f.name)
    
    def test_predictor_initialization(self, temp_config, temp_model):
        """Test CowDiseasePredictor initialization."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            assert predictor.model_path == model_path
            assert predictor.image_size == 150
            assert predictor.class_names == ['Healthy Cow', 'Lumpy Cow']
            assert predictor.model is not None
    
    def test_predictor_initialization_invalid_model(self, temp_config):
        """Test initialization with invalid model path."""
        with patch('tensorflow.keras.models.load_model', side_effect=Exception("Model not found")):
            with pytest.raises(ValueError, match="Error loading model"):
                CowDiseasePredictor("invalid_model.h5", temp_config)
    
    def test_preprocess_image_from_path(self, temp_config, temp_model, temp_image):
        """Test preprocessing image from file path."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            processed = predictor.preprocess_image(temp_image)
            
            assert processed.shape == (1, 150, 150, 3)
            assert processed.dtype == np.float32
            assert processed.max() <= 1.0
            assert processed.min() >= 0.0
    
    def test_preprocess_image_from_array(self, temp_config, temp_model):
        """Test preprocessing image from numpy array."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            # Create dummy image array
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            processed = predictor.preprocess_image(img_array)
            
            assert processed.shape == (1, 150, 150, 3)
            assert processed.dtype == np.float32
    
    def test_preprocess_image_invalid_path(self, temp_config, temp_model):
        """Test preprocessing invalid image path."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            with pytest.raises(FileNotFoundError):
                predictor.preprocess_image("nonexistent.jpg")
    
    def test_predict_single_image(self, temp_config, temp_model, temp_image):
        """Test prediction on single image."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            predicted_class, confidence, probabilities = predictor.predict(temp_image)
            
            assert predicted_class in ['Healthy Cow', 'Lumpy Cow']
            assert 0.0 <= confidence <= 1.0
            assert len(probabilities) == 2
            assert np.allclose(probabilities.sum(), 1.0)
    
    def test_predict_batch(self, temp_config, temp_model, temp_image):
        """Test batch prediction."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            # Create batch of images
            images = [temp_image, temp_image]
            predictions = predictor.predict_batch(images)
            
            assert len(predictions) == 2
            for pred in predictions:
                predicted_class, confidence, probabilities = pred
                assert predicted_class in ['Healthy Cow', 'Lumpy Cow']
                assert 0.0 <= confidence <= 1.0
    
    def test_predict_batch_with_error(self, temp_config, temp_model):
        """Test batch prediction with error handling."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            # Create batch with invalid image
            images = ["nonexistent.jpg", "another_nonexistent.jpg"]
            predictions = predictor.predict_batch(images)
            
            assert len(predictions) == 2
            for pred in predictions:
                predicted_class, confidence, probabilities = pred
                assert predicted_class == "Error"
                assert confidence == 0.0
    
    def test_get_model_info(self, temp_config, temp_model):
        """Test getting model information."""
        model_path, mock_model = temp_model
        mock_model.input_shape = (None, 150, 150, 3)
        mock_model.output_shape = (None, 2)
        mock_model.count_params.return_value = 1000000
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            predictor = CowDiseasePredictor(model_path, temp_config)
            
            info = predictor.get_model_info()
            
            assert info['model_path'] == model_path
            assert info['input_shape'] == (None, 150, 150, 3)
            assert info['output_shape'] == (None, 2)
            assert info['number_of_parameters'] == 1000000
            assert info['class_names'] == ['Healthy Cow', 'Lumpy Cow']
    
    def test_predict_without_model(self, temp_config):
        """Test prediction without loaded model."""
        with patch('tensorflow.keras.models.load_model', return_value=None):
            predictor = CowDiseasePredictor("dummy.h5", temp_config)
            predictor.model = None
            
            with pytest.raises(ValueError, match="Model not loaded"):
                predictor.predict("dummy.jpg")
    
    def test_visualize_prediction(self, temp_config, temp_model, temp_image):
        """Test prediction visualization."""
        model_path, mock_model = temp_model
        
        with patch('tensorflow.keras.models.load_model', return_value=mock_model):
            with patch('matplotlib.pyplot.show') as mock_show:
                predictor = CowDiseasePredictor(model_path, temp_config)
                
                # This should not raise an exception
                predictor.visualize_prediction(temp_image)
                
                # Verify that matplotlib was called
                mock_show.assert_called_once()


def test_main_function():
    """Test main function for command-line interface."""
    with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
        mock_args = MagicMock()
        mock_args.model = "test_model.h5"
        mock_args.image = "test_image.jpg"
        mock_args.config = "test_config.yaml"
        mock_args.save = None
        mock_args.batch = False
        mock_parse_args.return_value = mock_args
        
        with patch('src.inference.CowDiseasePredictor') as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = ("Healthy Cow", 0.8, np.array([0.8, 0.2]))
            mock_predictor_class.return_value = mock_predictor
            
            # Import and run main
            from src.inference import main
            
            # This should not raise an exception
            main()
