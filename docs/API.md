# API Documentation

## Data Loading

### DataLoader

```python
from src.data.data_loader import DataLoader

loader = DataLoader(config_path="config/config.yaml")
```

#### Methods

##### `load_images_from_directory(healthy_path, lumpy_path)`
Load images from directories and create labels.

**Parameters:**
- `healthy_path` (str): Path to healthy cow images
- `lumpy_path` (str): Path to lumpy cow images

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (images, labels)

##### `split_data(X, y)`
Split data into train, validation, and test sets.

**Parameters:**
- `X` (np.ndarray): Image data
- `y` (np.ndarray): Labels

**Returns:**
- `Tuple[np.ndarray, ...]`: (X_train, X_val, X_test, y_train, y_val, y_test)

##### `get_data_info(y)`
Get information about the dataset.

**Parameters:**
- `y` (np.ndarray): Labels array

**Returns:**
- `dict`: Dataset information

## Model Factory

### ModelFactory

```python
from src.models.model_factory import ModelFactory

factory = ModelFactory(config_path="config/config.yaml")
```

#### Methods

##### `create_custom_cnn()`
Create a custom CNN model.

**Returns:**
- `Model`: Compiled Keras model

##### `create_resnet50(pretrained=True)`
Create a ResNet50-based model.

**Parameters:**
- `pretrained` (bool): Whether to use pretrained weights

**Returns:**
- `Model`: Compiled Keras model

##### `create_vgg16(pretrained=True)`
Create a VGG16-based model.

**Parameters:**
- `pretrained` (bool): Whether to use pretrained weights

**Returns:**
- `Model`: Compiled Keras model

##### `create_efficientnet(pretrained=True)`
Create an EfficientNet-based model.

**Parameters:**
- `pretrained` (bool): Whether to use pretrained weights

**Returns:**
- `Model`: Compiled Keras model

##### `create_model(architecture=None, **kwargs)`
Create a model based on the specified architecture.

**Parameters:**
- `architecture` (str): Model architecture name
- `**kwargs`: Additional arguments

**Returns:**
- `Model`: Compiled Keras model

## Training

### ModelTrainer

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(config_path="config/config.yaml")
```

#### Methods

##### `train_model(model, train_generator, val_generator, model_name)`
Train the model with data generators.

**Parameters:**
- `model` (Model): Keras model to train
- `train_generator`: Training data generator
- `val_generator`: Validation data generator
- `model_name` (str): Name for saving the model

**Returns:**
- `Dict`: Training history

##### `train_model_with_data(model, X_train, y_train, X_val, y_val, model_name)`
Train the model with numpy arrays.

**Parameters:**
- `model` (Model): Keras model to train
- `X_train` (np.ndarray): Training images
- `y_train` (np.ndarray): Training labels
- `X_val` (np.ndarray): Validation images
- `y_val` (np.ndarray): Validation labels
- `model_name` (str): Name for saving the model

**Returns:**
- `Dict`: Training history

##### `plot_training_history(history, model_name, save_plot=True)`
Plot and save training history.

**Parameters:**
- `history` (Dict): Training history dictionary
- `model_name` (str): Name of the model
- `save_plot` (bool): Whether to save the plot

## Evaluation

### ModelEvaluator

```python
from src.models.evaluator import ModelEvaluator

evaluator = ModelEvaluator(class_names=['Healthy Cow', 'Lumpy Cow'])
```

#### Methods

##### `evaluate_model(model, X_test, y_test)`
Comprehensive model evaluation.

**Parameters:**
- `model` (Model): Trained Keras model
- `X_test` (np.ndarray): Test images
- `y_test` (np.ndarray): Test labels

**Returns:**
- `Dict`: Evaluation results

##### `plot_confusion_matrix(save_path=None, figsize=(8, 6))`
Plot confusion matrix.

**Parameters:**
- `save_path` (str, optional): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

##### `plot_roc_curve(save_path=None, figsize=(8, 6))`
Plot ROC curve.

**Parameters:**
- `save_path` (str, optional): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

##### `generate_evaluation_report(model_name, save_path=None)`
Generate a comprehensive evaluation report.

**Parameters:**
- `model_name` (str): Name of the model
- `save_path` (str, optional): Path to save the report

**Returns:**
- `str`: Report as string

## Inference

### CowDiseasePredictor

```python
from src.inference import CowDiseasePredictor

predictor = CowDiseasePredictor(model_path="models/model.h5")
```

#### Methods

##### `predict(image)`
Make prediction on a single image.

**Parameters:**
- `image` (Union[str, np.ndarray]): Image path or numpy array

**Returns:**
- `Tuple[str, float, np.ndarray]`: (predicted_class, confidence, all_probabilities)

##### `predict_batch(images)`
Make predictions on multiple images.

**Parameters:**
- `images` (List[Union[str, np.ndarray]]): List of image paths or numpy arrays

**Returns:**
- `List[Tuple[str, float, np.ndarray]]`: List of prediction tuples

##### `visualize_prediction(image, save_path=None, figsize=(12, 5))`
Visualize prediction with image and confidence scores.

**Parameters:**
- `image` (Union[str, np.ndarray]): Image path or numpy array
- `save_path` (str, optional): Path to save the visualization
- `figsize` (Tuple[int, int]): Figure size

##### `get_model_info()`
Get information about the loaded model.

**Returns:**
- `dict`: Model information

## Data Augmentation

### DataAugmentation

```python
from src.data.data_augmentation import DataAugmentation

augmentation = DataAugmentation(config_path="config/config.yaml")
```

#### Methods

##### `create_train_generator(X_train, y_train, batch_size=32)`
Create training data generator with augmentation.

**Parameters:**
- `X_train` (np.ndarray): Training images
- `y_train` (np.ndarray): Training labels
- `batch_size` (int): Batch size for training

**Returns:**
- `ImageDataGenerator`: Training data generator

##### `create_val_generator(X_val, y_val, batch_size=32)`
Create validation data generator without augmentation.

**Parameters:**
- `X_val` (np.ndarray): Validation images
- `y_val` (np.ndarray): Validation labels
- `batch_size` (int): Batch size for validation

**Returns:**
- `ImageDataGenerator`: Validation data generator

##### `balance_dataset(X, y)`
Balance the dataset by augmenting the minority class.

**Parameters:**
- `X` (np.ndarray): Input images
- `y` (np.ndarray): Input labels

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (balanced_images, balanced_labels)

## Utilities

### VisualizationUtils

```python
from src.utils.visualization import VisualizationUtils

visualizer = VisualizationUtils()
```

#### Methods

##### `plot_data_distribution(labels, class_names, title, save_path=None)`
Plot the distribution of classes in the dataset.

**Parameters:**
- `labels` (np.ndarray): Array of labels
- `class_names` (List[str]): List of class names
- `title` (str): Plot title
- `save_path` (str, optional): Path to save the plot

##### `plot_sample_images(images, labels, class_names, num_samples=8, title, save_path=None)`
Plot sample images from the dataset.

**Parameters:**
- `images` (np.ndarray): Array of images
- `labels` (np.ndarray): Array of labels
- `class_names` (List[str]): List of class names
- `num_samples` (int): Number of samples to display
- `title` (str): Plot title
- `save_path` (str, optional): Path to save the plot

##### `plot_training_curves(history, metrics=None, save_path=None)`
Plot training curves for multiple metrics.

**Parameters:**
- `history` (Dict): Training history dictionary
- `metrics` (List[str], optional): List of metrics to plot
- `save_path` (str, optional): Path to save the plot

### Helper Functions

```python
from src.utils.helpers import *
```

#### Functions

##### `setup_logging(log_level="INFO", log_file=None)`
Setup logging configuration.

**Parameters:**
- `log_level` (str): Logging level
- `log_file` (str, optional): Optional log file path

**Returns:**
- `logging.Logger`: Configured logger

##### `load_config(config_path)`
Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

##### `save_results(results, file_path)`
Save results to JSON file.

**Parameters:**
- `results` (Dict[str, Any]): Results dictionary
- `file_path` (str): Path to save results

##### `create_experiment_log(experiment_name, config, results, log_dir="experiments")`
Create an experiment log with timestamp.

**Parameters:**
- `experiment_name` (str): Name of the experiment
- `config` (Dict[str, Any]): Configuration used
- `results` (Dict[str, Any]): Results obtained
- `log_dir` (str): Directory to save experiment logs

**Returns:**
- `str`: Path to the experiment log file

##### `set_random_seeds(seed=42)`
Set random seeds for reproducibility.

**Parameters:**
- `seed` (int): Random seed value
