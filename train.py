"""
Main training script for Cow Lumpy Disease Detection.

This script refactors the original Jupyter notebook into a clean, modular training pipeline.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import DataLoader, load_kaggle_dataset
from src.data.data_augmentation import DataAugmentation
from src.models.model_factory import ModelFactory
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.helpers import setup_logging, create_experiment_log, set_random_seeds, print_system_info
from src.utils.visualization import VisualizationUtils


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Cow Disease Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset (Kaggle download path or local directory)')
    parser.add_argument('--model-architecture', type=str, default='custom_cnn',
                       choices=['custom_cnn', 'resnet50', 'vgg16', 'efficientnet'],
                       help='Model architecture to use')
    parser.add_argument('--experiment-name', type=str, default='cow_disease_detection',
                       help='Name for this experiment')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights for transfer learning')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Cow Disease Detection Training")
    
    # Print system info
    print_system_info()
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader(args.config)
        data_augmentation = DataAugmentation(args.config)
        model_factory = ModelFactory(args.config)
        trainer = ModelTrainer(args.config)
        evaluator = ModelEvaluator()
        visualizer = VisualizationUtils()
        
        # Load data
        logger.info("Loading dataset...")
        if os.path.isdir(args.data_path):
            # Local directory
            healthy_path = os.path.join(args.data_path, "Healthy")
            lumpy_path = os.path.join(args.data_path, "Lumpy")
            
            # Try alternative naming
            if not os.path.exists(healthy_path):
                healthy_path = os.path.join(args.data_path, "healthy")
            if not os.path.exists(lumpy_path):
                lumpy_path = os.path.join(args.data_path, "lumpy")
            
            X, y = data_loader.load_images_from_directory(healthy_path, lumpy_path)
        else:
            # Kaggle dataset path
            X, y = load_kaggle_dataset(args.data_path, args.config)
        
        logger.info(f"Loaded {len(X)} images")
        
        # Get data information
        data_info = data_loader.get_data_info(y)
        logger.info(f"Data distribution: {data_info['class_distribution']}")
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Balance dataset if needed
        if data_info['class_balance'] < 0.8 or data_info['class_balance'] > 1.2:
            logger.info("Balancing dataset...")
            X_train, y_train = data_augmentation.balance_dataset(X_train, y_train)
            logger.info(f"Balanced train set: {len(X_train)} images")
        
        # Create model
        logger.info(f"Creating {args.model_architecture} model...")
        model = model_factory.create_model(
            architecture=args.model_architecture,
            pretrained=args.pretrained
        )
        
        # Print model summary
        model_summary = model_factory.get_model_summary(model)
        logger.info(f"Model created with {model.count_params():,} parameters")
        
        # Create data generators
        logger.info("Creating data generators...")
        train_generator = data_augmentation.create_train_generator(
            X_train, y_train, 
            batch_size=data_loader.config['data']['batch_size']
        )
        val_generator = data_augmentation.create_val_generator(
            X_val, y_val,
            batch_size=data_loader.config['data']['batch_size']
        )
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train_model(
            model, train_generator, val_generator, 
            model_name=args.experiment_name
        )
        
        # Plot training history
        logger.info("Plotting training history...")
        trainer.plot_training_history(history, args.experiment_name)
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = evaluator.evaluate_model(model, X_test, y_test)
        
        # Print evaluation results
        logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Test Precision: {results['precision']:.4f}")
        logger.info(f"Test Recall: {results['recall']:.4f}")
        logger.info(f"Test F1-Score: {results['f1_score']:.4f}")
        
        # Generate evaluation plots
        logger.info("Generating evaluation plots...")
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve()
        evaluator.plot_precision_recall_curve()
        
        # Save evaluation results
        evaluator.save_results(args.experiment_name)
        
        # Create comprehensive results
        comprehensive_results = {
            'experiment_name': args.experiment_name,
            'model_architecture': args.model_architecture,
            'pretrained': args.pretrained,
            'data_info': data_info,
            'model_info': {
                'total_parameters': model.count_params(),
                'trainable_parameters': sum([np.prod(w.shape) for w in model.trainable_weights]),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            },
            'training_history': history,
            'evaluation_results': results,
            'best_epoch': trainer.get_best_epoch(history),
            'best_val_accuracy': max(history['val_accuracy'])
        }
        
        # Create experiment log
        experiment_dir = create_experiment_log(
            args.experiment_name, 
            data_loader.config, 
            comprehensive_results
        )
        
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved to: {experiment_dir}")
        logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        logger.info(f"Test accuracy: {results['accuracy']:.4f}")
        
        # Create summary visualization
        visualizer.create_summary_plot(
            data_info, 
            comprehensive_results['model_info'], 
            history,
            save_path=os.path.join(experiment_dir, "summary_plot.png")
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
