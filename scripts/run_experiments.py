#!/usr/bin/env python3
"""
Script to run multiple experiments with different configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.helpers import setup_logging


def run_experiment(config_path: str, data_path: str, experiment_name: str, 
                  model_architecture: str = None, pretrained: bool = False):
    """
    Run a single experiment.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to dataset
        experiment_name: Name for the experiment
        model_architecture: Model architecture to use
        pretrained: Whether to use pretrained weights
    """
    cmd = [
        "python", "train.py",
        "--config", config_path,
        "--data-path", data_path,
        "--experiment-name", experiment_name
    ]
    
    if model_architecture:
        cmd.extend(["--model-architecture", model_architecture])
    
    if pretrained:
        cmd.append("--pretrained")
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Experiment {experiment_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {experiment_name} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main function to run multiple experiments."""
    parser = argparse.ArgumentParser(description='Run multiple experiments')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--configs', type=str, nargs='+', 
                       default=['config/config.yaml', 'config/custom_config.yaml'],
                       help='Configuration files to use')
    parser.add_argument('--architectures', type=str, nargs='+',
                       default=['custom_cnn', 'resnet50'],
                       help='Model architectures to test')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Define experiments
    experiments = []
    
    for config in args.configs:
        for arch in args.architectures:
            experiment_name = f"{Path(config).stem}_{arch}"
            if args.pretrained:
                experiment_name += "_pretrained"
            
            experiments.append({
                'config': config,
                'architecture': arch,
                'name': experiment_name,
                'pretrained': args.pretrained
            })
    
    print(f"Running {len(experiments)} experiments...")
    
    # Run experiments
    successful = 0
    failed = 0
    
    for exp in experiments:
        success = run_experiment(
            exp['config'],
            args.data_path,
            exp['name'],
            exp['architecture'],
            exp['pretrained']
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Experiment Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved in: experiments/")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
