#!/usr/bin/env python3
"""
Setup script for the Cow Lumpy Disease Detection project.
This script helps users get started quickly.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "results",
        "experiments"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True


def download_dataset():
    """Download the dataset."""
    print("📥 Downloading dataset...")
    
    try:
        subprocess.run([sys.executable, "scripts/download_dataset.py"], check=True)
        print("✅ Dataset downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        print("You can download it manually later using:")
        print("python scripts/download_dataset.py")
        return False
    
    return True


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        print("✅ All tests passed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    print("🐄 Setting up Cow Lumpy Disease Detection Project")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("⚠️  Please install dependencies manually: pip install -r requirements.txt")
    
    # Download dataset
    print("\n📥 Downloading dataset...")
    dataset_success = download_dataset()
    
    # Run tests
    print("\n🧪 Running tests...")
    test_success = run_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Project setup completed!")
    print("\n📋 Next steps:")
    print("1. If dataset download failed, run: python scripts/download_dataset.py")
    print("2. Start training: python train.py --data-path data/raw")
    print("3. Make predictions: python -m src.inference --model models/model.h5 --image path/to/image.jpg")
    print("4. Web interface: streamlit run src/inference.py")
    
    if not dataset_success:
        print("\n⚠️  Note: Dataset download failed. Please download it manually.")
    
    if not test_success:
        print("\n⚠️  Note: Some tests failed. Please check the test output.")


if __name__ == "__main__":
    main()
