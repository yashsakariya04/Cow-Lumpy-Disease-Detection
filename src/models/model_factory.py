"""
Model factory for creating different CNN architectures for cow disease detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any
import yaml


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ModelFactory with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        self.image_size = self.config['data']['image_size']
        self.num_classes = self.model_config['num_classes']
        self.dropout_rate = self.model_config['dropout_rate']
        self.learning_rate = self.model_config['learning_rate']
    
    def create_custom_cnn(self) -> Model:
        """
        Create a custom CNN model (improved version of the original).
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            # Input layer
            layers.Input(shape=(self.image_size, self.image_size, 3)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate),
            
            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model)
    
    def create_resnet50(self, pretrained: bool = True) -> Model:
        """
        Create a ResNet50-based model with transfer learning.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Compiled Keras model
        """
        if pretrained:
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
            base_model.trainable = False
        else:
            base_model = ResNet50(
                weights=None,
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model)
    
    def create_vgg16(self, pretrained: bool = True) -> Model:
        """
        Create a VGG16-based model with transfer learning.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Compiled Keras model
        """
        if pretrained:
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
            base_model.trainable = False
        else:
            base_model = VGG16(
                weights=None,
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model)
    
    def create_efficientnet(self, pretrained: bool = True) -> Model:
        """
        Create an EfficientNet-based model with transfer learning.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Compiled Keras model
        """
        if pretrained:
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
            base_model.trainable = False
        else:
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self._compile_model(model)
    
    def _compile_model(self, model: Model) -> Model:
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            model: Uncompiled Keras model
            
        Returns:
            Compiled Keras model
        """
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_model(self, architecture: str = None, **kwargs) -> Model:
        """
        Create a model based on the specified architecture.
        
        Args:
            architecture: Model architecture name
            **kwargs: Additional arguments for model creation
            
        Returns:
            Compiled Keras model
        """
        if architecture is None:
            architecture = self.model_config['architecture']
        
        model_creators = {
            'custom_cnn': self.create_custom_cnn,
            'resnet50': self.create_resnet50,
            'vgg16': self.create_vgg16,
            'efficientnet': self.create_efficientnet
        }
        
        if architecture not in model_creators:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model_creators[architecture](**kwargs)
    
    def get_model_summary(self, model: Model) -> str:
        """
        Get a string representation of the model summary.
        
        Args:
            model: Keras model
            
        Returns:
            Model summary as string
        """
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
