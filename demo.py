#!/usr/bin/env python3
"""
Demo script for the enhanced Cow Lumpy Disease Detection project.
This script demonstrates the key features without requiring the full dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_model_creation():
    """Demonstrate model creation capabilities."""
    print("🏗️  Demo: Model Creation")
    print("=" * 50)
    
    try:
        from src.models.model_factory import ModelFactory
        
        # Create model factory
        factory = ModelFactory("config/config.yaml")
        
        # Create different model architectures
        print("Creating Custom CNN model...")
        custom_model = factory.create_custom_cnn()
        print(f"✅ Custom CNN created with {custom_model.count_params():,} parameters")
        
        print("\nCreating ResNet50 model...")
        resnet_model = factory.create_resnet50(pretrained=False)
        print(f"✅ ResNet50 created with {resnet_model.count_params():,} parameters")
        
        print("\nCreating VGG16 model...")
        vgg_model = factory.create_vgg16(pretrained=False)
        print(f"✅ VGG16 created with {vgg_model.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def demo_data_processing():
    """Demonstrate data processing capabilities."""
    print("\n📊 Demo: Data Processing")
    print("=" * 50)
    
    try:
        from src.data.data_loader import DataLoader
        from src.data.data_augmentation import DataAugmentation
        
        # Create data loader
        loader = DataLoader("config/config.yaml")
        print("✅ DataLoader initialized")
        
        # Create data augmentation
        augmentation = DataAugmentation("config/config.yaml")
        print("✅ DataAugmentation initialized")
        
        # Create dummy data for demonstration (larger dataset for proper splitting)
        dummy_images = np.random.rand(100, 150, 150, 3).astype(np.float32)
        dummy_labels = np.random.randint(0, 2, 100)
        
        print(f"✅ Created dummy dataset: {dummy_images.shape[0]} images")
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(dummy_images, dummy_labels)
        print(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Test data augmentation
        train_gen = augmentation.create_train_generator(X_train, y_train, batch_size=4)
        print("✅ Training data generator created")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def demo_inference():
    """Demonstrate inference capabilities."""
    print("\n🔮 Demo: Inference")
    print("=" * 50)
    
    try:
        # Import without streamlit dependency for demo
        import sys
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.models.model_factory import ModelFactory
        from src.inference import CowDiseasePredictor
        
        # Create a simple model for demo
        factory = ModelFactory("config/config.yaml")
        model = factory.create_custom_cnn()
        
        # Save model temporarily
        model_path = "demo_model.h5"
        model.save(model_path)
        
        # Create predictor
        predictor = CowDiseasePredictor(model_path, "config/config.yaml")
        print("✅ CowDiseasePredictor initialized")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Make prediction
        predicted_class, confidence, probabilities = predictor.predict(dummy_image)
        print(f"✅ Prediction: {predicted_class} (Confidence: {confidence:.2%})")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"✅ Model info: {model_info['number_of_parameters']:,} parameters")
        
        # Clean up
        import os
        os.remove(model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📈 Demo: Visualization")
    print("=" * 50)
    
    try:
        from src.utils.visualization import VisualizationUtils
        
        # Create visualizer
        visualizer = VisualizationUtils()
        print("✅ VisualizationUtils initialized")
        
        # Create dummy data
        dummy_labels = np.random.randint(0, 2, 100)
        class_names = ['Healthy Cow', 'Lumpy Cow']
        
        # Test data distribution plot
        print("Creating data distribution plot...")
        visualizer.plot_data_distribution(dummy_labels, class_names, "Demo Data Distribution")
        print("✅ Data distribution plot created")
        
        # Test training curves
        dummy_history = {
            'accuracy': [0.5, 0.6, 0.7, 0.8, 0.85],
            'val_accuracy': [0.4, 0.5, 0.6, 0.7, 0.75],
            'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4]
        }
        
        print("Creating training curves plot...")
        visualizer.plot_training_curves(dummy_history)
        print("✅ Training curves plot created")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def main():
    """Run all demos."""
    print("🐄 Cow Lumpy Disease Detection - Enhanced Project Demo")
    print("=" * 60)
    print("This demo showcases the enhanced features of your project.")
    print("Note: This demo uses dummy data and doesn't require the actual dataset.")
    print()
    
    demos = [
        ("Model Creation", demo_model_creation),
        ("Data Processing", demo_data_processing),
        ("Inference", demo_inference),
        ("Visualization", demo_visualization)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} demo failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Demo Summary:")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 All demos passed! Your enhanced project is working perfectly!")
        print("\n🚀 Next steps:")
        print("1. Download the dataset: python scripts/download_dataset.py")
        print("2. Train a model: python train.py --data-path data/raw")
        print("3. Make predictions: python -m src.inference --model models/model.h5 --image path/to/image.jpg")
        print("4. Web interface: streamlit run src/inference.py")
    else:
        print(f"\n⚠️  {total - passed} demos failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
