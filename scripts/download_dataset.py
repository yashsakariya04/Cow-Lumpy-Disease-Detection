#!/usr/bin/env python3
"""
Script to download the Cow Lumpy Disease dataset from Kaggle.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub not found. Installing...")
    os.system("pip install kagglehub")
    import kagglehub


def download_dataset(output_dir: str = "data/raw"):
    """
    Download the Cow Lumpy Disease dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading Cow Lumpy Disease dataset from Kaggle...")
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("shivamagarwal29/cow-lumpy-disease-dataset")
        
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy dataset to output directory (optional)
        if dataset_path != output_dir:
            print(f"Copying dataset to {output_dir}...")
            import shutil
            shutil.copytree(dataset_path, output_dir, dirs_exist_ok=True)
            print(f"Dataset copied to: {output_dir}")
        
        # List dataset contents
        print("\nDataset contents:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        return dataset_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download Cow Lumpy Disease dataset')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Directory to save the dataset (default: data/raw)')
    
    args = parser.parse_args()
    
    dataset_path = download_dataset(args.output_dir)
    
    if dataset_path:
        print(f"\n✅ Dataset successfully downloaded!")
        print(f"📁 Location: {dataset_path}")
        print(f"\n🚀 You can now run training with:")
        print(f"   python train.py --data-path {dataset_path}")
    else:
        print("❌ Failed to download dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
