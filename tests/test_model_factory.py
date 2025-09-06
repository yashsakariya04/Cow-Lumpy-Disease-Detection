"""
Tests for model factory functionality.
"""

import pytest
import numpy as np
import tempfile
import os

from src.models.model_factory import ModelFactory


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config_content = """
data:
  image_size: 224
  batch_size: 32

model:
  architecture: "custom_cnn"
  num_classes: 2
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 50
  patience: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_model_factory_initialization(self, temp_config):
        """Test ModelFactory initialization."""
        factory = ModelFactory(temp_config)
        
        assert factory.image_size == 224
        assert factory.num_classes == 2
        assert factory.dropout_rate == 0.3
        assert factory.learning_rate == 0.001
    
    def test_create_custom_cnn(self, temp_config):
        """Test creating custom CNN model."""
        factory = ModelFactory(temp_config)
        model = factory.create_custom_cnn()
        
        # Check model structure
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2)
        assert model.count_params() > 0
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0
    
    def test_create_resnet50(self, temp_config):
        """Test creating ResNet50 model."""
        factory = ModelFactory(temp_config)
        model = factory.create_resnet50(pretrained=False)
        
        # Check model structure
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2)
        assert model.count_params() > 0
    
    def test_create_vgg16(self, temp_config):
        """Test creating VGG16 model."""
        factory = ModelFactory(temp_config)
        model = factory.create_vgg16(pretrained=False)
        
        # Check model structure
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2)
        assert model.count_params() > 0
    
    def test_create_efficientnet(self, temp_config):
        """Test creating EfficientNet model."""
        factory = ModelFactory(temp_config)
        model = factory.create_efficientnet(pretrained=False)
        
        # Check model structure
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2)
        assert model.count_params() > 0
    
    def test_create_model_with_architecture(self, temp_config):
        """Test creating model with specific architecture."""
        factory = ModelFactory(temp_config)
        
        # Test custom CNN
        model = factory.create_model('custom_cnn')
        assert model.input_shape == (None, 224, 224, 3)
        
        # Test ResNet50
        model = factory.create_model('resnet50', pretrained=False)
        assert model.input_shape == (None, 224, 224, 3)
    
    def test_create_model_invalid_architecture(self, temp_config):
        """Test creating model with invalid architecture."""
        factory = ModelFactory(temp_config)
        
        with pytest.raises(ValueError, match="Unknown architecture"):
            factory.create_model('invalid_architecture')
    
    def test_get_model_summary(self, temp_config):
        """Test getting model summary."""
        factory = ModelFactory(temp_config)
        model = factory.create_custom_cnn()
        
        summary = factory.get_model_summary(model)
        
        assert isinstance(summary, str)
        assert "Model:" in summary
        assert "Total params:" in summary
    
    def test_model_prediction(self, temp_config):
        """Test model prediction functionality."""
        factory = ModelFactory(temp_config)
        model = factory.create_custom_cnn()
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (1, 2)
        assert np.allclose(prediction.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_model_compilation(self, temp_config):
        """Test that models are properly compiled."""
        factory = ModelFactory(temp_config)
        
        architectures = ['custom_cnn', 'resnet50', 'vgg16', 'efficientnet']
        
        for arch in architectures:
            model = factory.create_model(arch, pretrained=False)
            
            # Check compilation
            assert model.optimizer is not None
            assert model.loss == 'sparse_categorical_crossentropy'
            assert 'accuracy' in [metric.name for metric in model.metrics]
    
    def test_different_image_sizes(self):
        """Test models with different image sizes."""
        config_content = """
data:
  image_size: 150
  batch_size: 32

model:
  architecture: "custom_cnn"
  num_classes: 2
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 50
  patience: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name
        
        try:
            factory = ModelFactory(config_path)
            model = factory.create_custom_cnn()
            
            assert model.input_shape == (None, 150, 150, 3)
        finally:
            os.unlink(config_path)
