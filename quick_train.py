#!/usr/bin/env python3
"""
Quick training script that works with your current dataset structure.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.model_factory import ModelFactory
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.helpers import setup_logging, set_random_seeds

def load_dataset(data_path):
    """Load dataset from the current directory structure."""
    print(f"Loading dataset from: {data_path}")
    
    X = []
    y = []
    image_size = 150
    
    # Load healthy cow images
    healthy_path = os.path.join(data_path, "healthycows")
    if os.path.exists(healthy_path):
        print(f"Loading healthy cow images from: {healthy_path}")
        for filename in os.listdir(healthy_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(healthy_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (image_size, image_size))
                        img = img.astype(np.float32) / 255.0
                        X.append(img)
                        y.append(0)  # Healthy cow
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load lumpy cow images
    lumpy_path = os.path.join(data_path, "lumpycows")
    if os.path.exists(lumpy_path):
        print(f"Loading lumpy cow images from: {lumpy_path}")
        for filename in os.listdir(lumpy_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(lumpy_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (image_size, image_size))
                        img = img.astype(np.float32) / 255.0
                        X.append(img)
                        y.append(1)  # Lumpy cow
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images total")
    print(f"Healthy cows: {np.sum(y == 0)}")
    print(f"Lumpy cows: {np.sum(y == 1)}")
    
    return X, y

def main():
    """Main training function."""
    print("🐄 Quick Training - Cow Lumpy Disease Detection")
    print("=" * 50)
    
    # Setup
    set_random_seeds()
    logger = setup_logging()
    
    # Load dataset
    data_path = "data/raw"
    X, y = load_dataset(data_path)
    
    if len(X) == 0:
        print("❌ No images found! Please check your data directory.")
        return
    
    # Shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    # Split data
    print("Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    print("Creating model...")
    factory = ModelFactory("config/config.yaml")
    model = factory.create_custom_cnn()
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Train model
    print("Starting training...")
    trainer = ModelTrainer("config/config.yaml")
    
    history = trainer.train_model_with_data(
        model, X_train, y_train, X_val, y_val, "quick_training"
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test)
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 TRAINING RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1-Score: {results['f1_score']:.4f}")
    
    # Save evaluation results
    evaluator.save_results("quick_training")
    
    # Plot results
    trainer.plot_training_history(history, "quick_training")
    evaluator.plot_confusion_matrix()
    
    print("\n🎉 Training completed successfully!")
    print(f"Model saved to: models/quick_training_final.h5")
    print(f"Results saved to: results/")

if __name__ == "__main__":
    main()
