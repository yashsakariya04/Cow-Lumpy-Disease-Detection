"""
Helper utilities for the cow disease detection project.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving config to {config_path}: {e}")


def create_directories(directories: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results(results: Dict[str, Any], file_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        file_path: Path to save results
    """
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving results to {file_path}: {e}")


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Results dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading results from {file_path}: {e}")


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Human-readable file size
    """
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Number of trainable parameters
    """
    return model.count_params()


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Array of labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_model_summary_dict(model) -> Dict[str, Any]:
    """
    Get model summary as dictionary.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary dictionary
    """
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    summary_text = buffer.getvalue()
    
    return {
        'summary_text': summary_text,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


def create_experiment_log(experiment_name: str, config: Dict[str, Any], 
                         results: Dict[str, Any], log_dir: str = "experiments") -> str:
    """
    Create an experiment log with timestamp.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration used
        results: Results obtained
        log_dir: Directory to save experiment logs
        
    Returns:
        Path to the experiment log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = os.path.join(log_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)
    
    # Save results
    results_path = os.path.join(experiment_dir, "results.json")
    save_results(results, results_path)
    
    # Create experiment summary
    summary = {
        'experiment_id': experiment_id,
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'config_file': config_path,
        'results_file': results_path,
        'summary': {
            'best_accuracy': results.get('accuracy', 'N/A'),
            'best_epoch': results.get('best_epoch', 'N/A'),
            'total_epochs': results.get('total_epochs', 'N/A')
        }
    }
    
    summary_path = os.path.join(experiment_dir, "summary.json")
    save_results(summary, summary_path)
    
    return experiment_dir


def validate_image_file(file_path: str) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_bytes': os.path.getsize(image_path),
                'size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
    except Exception as e:
        return {'error': str(e)}


def create_data_report(data_dir: str, output_path: str = "data_report.json"):
    """
    Create a comprehensive data report.
    
    Args:
        data_dir: Directory containing the dataset
        output_path: Path to save the report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_directory': data_dir,
        'total_files': 0,
        'class_distribution': {},
        'file_info': [],
        'errors': []
    }
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                report['total_files'] += 1
                
                # Get class from directory name
                class_name = os.path.basename(root)
                if class_name not in report['class_distribution']:
                    report['class_distribution'][class_name] = 0
                report['class_distribution'][class_name] += 1
                
                # Get image info
                img_info = get_image_info(file_path)
                img_info['file_path'] = file_path
                img_info['class'] = class_name
                report['file_info'].append(img_info)
                
                if 'error' in img_info:
                    report['errors'].append(f"{file_path}: {img_info['error']}")
    
    save_results(report, output_path)
    return report


def print_system_info():
    """Print system information for reproducibility."""
    import platform
    import sys
    import tensorflow as tf
    
    print("System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
