#!/usr/bin/env python3
"""
Randomize and Train Script for FOD Detection
This script randomizes the dataset and then initiates training
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time

def main():
    """Main function to randomize dataset and initiate training"""
    parser = argparse.ArgumentParser(description='Randomize dataset and train FOD detection model')
    
    # Randomization arguments
    parser.add_argument('--dataset', type=str, default='data/processed/fod_yolo',
                       help='Path to YOLO dataset')
    parser.add_argument('--level', type=str, choices=['normal', 'high', 'extreme'], 
                       default='extreme', help='Randomization level')
    parser.add_argument('--no-class-balance', action='store_false', dest='class_balance',
                       help='Disable class balancing across splits')
    
    # Training arguments
    parser.add_argument('--model', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                      default='n', help='Model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, help='Batch size override')
    parser.add_argument('--skip-randomize', action='store_true', help='Skip randomization step')
    parser.add_argument('--full-training', action='store_true', help='Use full training script instead of quick train')
    
    args = parser.parse_args()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Starting randomization and training process at {timestamp}")
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists() or not (dataset_path / 'data.yaml').exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return 1
    
    # Step 1: Randomize dataset (unless skipped)
    if not args.skip_randomize:
        print("\n=== Step 1: Enhanced Dataset Randomization ===")
        randomize_cmd = [
            "python", "enhance_dataset_randomization.py",
            "--dataset", args.dataset,
            "--level", args.level
        ]
        
        if not args.class_balance:
            randomize_cmd.append("--no-class-balance")
        
        print(f"Running command: {' '.join(randomize_cmd)}")
        result = subprocess.run(randomize_cmd)
        
        if result.returncode != 0:
            print("Error: Dataset randomization failed")
            return 1
        
        print("\nDataset randomization completed successfully!")
    else:
        print("\n=== Skipping dataset randomization as requested ===")
    
    # Step 2: Train model
    if args.full_training:
        print("\n=== Step 2: Running Full Training ===")
        train_cmd = [
            "python", "train_fod_detection.py",
            "--model", f"yolov8{args.model}",
            "--epochs", str(args.epochs)
        ]
        
        if args.batch_size:
            train_cmd.extend(["--batch_size", str(args.batch_size)])
    else:
        print("\n=== Step 2: Running Quick Training ===")
        train_cmd = [
            "python", "quick_train.py",
            "--quick",
            "--model", args.model,
            "--epochs", str(args.epochs)
        ]
        
        if args.batch_size:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
    
    print(f"Running command: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    
    if result.returncode != 0:
        print("Warning: Training process returned non-zero exit code")
    
    print("\nRandomization and training process completed!")
    print(f"Started at: {timestamp}")
    print(f"Completed at: {time.strftime('%Y%m%d_%H%M%S')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 