#!/usr/bin/env python3
"""
Quick Training Script for FOD Detection
Simplified interface for common training tasks
"""

import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime
import subprocess
import os

def quick_train(model_size='n', epochs=10, batch_size=None, dataset_path=None, randomize=False, randomize_level='high'):
    """Quick training with sensible defaults"""
    print(f"Starting quick training with YOLOv8{model_size}")
    print(f"Training for {epochs} epochs (quick test)")
    
    # Set default dataset path if not provided
    if not dataset_path:
        dataset_path = "data/processed/fod_yolo/data.yaml"
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None
    
    # Randomize the dataset if requested
    if randomize:
        print(f"\n🔀 Randomizing dataset with level: {randomize_level}")
        randomize_cmd = [
            "python", "enhance_dataset_randomization.py",
            "--dataset", str(Path(dataset_path).parent),
            "--level", randomize_level
        ]
        
        print(f"Running command: {' '.join(randomize_cmd)}")
        result = subprocess.run(randomize_cmd)
        
        if result.returncode != 0:
            print("Warning: Dataset randomization returned a non-zero exit code")
            choice = input("Continue with training anyway? (y/n): ").lower()
            if choice != 'y':
                return None, None
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Adjust batch size for GPU memory
    if not batch_size:
        if device == 'cuda':
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                
                if gpu_memory < 4:
                    batch_size = 8
                elif gpu_memory < 8:
                    batch_size = 16
                else:
                    batch_size = 32
            except:
                batch_size = 16
        else:
            batch_size = 8
    
    print(f"Using batch size: {batch_size}")
    
    # Load model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Training parameters
    train_args = {
        'data': dataset_path,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': device,
        'project': 'runs/train',
        'name': f'fod_yolov8{model_size}_quick',
        'patience': max(1, epochs // 3),  # Early stopping
        'save_period': max(1, epochs // 5),  # Save checkpoints
        'verbose': True,
        'seed': 42
    }
    
    print(f"\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    try:
        print(f"\nStarting training...")
        results = model.train(**train_args)
        
        # Validate the model
        print("Evaluating model...")
        val_results = model.val()
        
        # Extract metrics
        metrics = {}
        if hasattr(val_results, 'box'):
            metrics = {
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.p.mean()),
                'recall': float(val_results.box.r.mean()),
            }
        
        # Save the model
        output_dir = Path('data/models/quick_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the best model
        best_weights = Path(f'runs/train/fod_yolov8{model_size}_quick/weights/best.pt')
        if best_weights.exists():
            final_model_path = output_dir / f'fod_yolov8{model_size}_best.pt'
            shutil.copy2(best_weights, final_model_path)
            print(f"Best model saved to: {final_model_path}")
        
        print("\nQuick Training Results:")
        if metrics:
            print(f"mAP@0.5: {metrics['map50']:.3f}")
            print(f"mAP@0.5:0.95: {metrics['map50_95']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
        
        return results, metrics
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def list_available_models():
    """List available trained models"""
    models_dir = Path('data/models')
    if not models_dir.exists():
        print("No models directory found.")
        return []
    
    models = list(models_dir.glob('**/*.pt'))
    
    if models:
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
    else:
        print("No trained models found in data/models/")
    
    return models

def validate_existing_model(model_path, dataset_path=None):
    """Validate an existing model"""
    print(f"Validating model: {model_path}")
    
    if not dataset_path:
        dataset_path = "data/processed/fod_yolo/data.yaml"
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return None
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return None
    
    model = YOLO(model_path)
    results = model.val(data=dataset_path)
    
    if hasattr(results, 'box'):
        metrics = {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.p.mean()),
            'recall': float(results.box.r.mean()),
        }
        
        print("\nValidation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        return metrics
    return None

def setup_training_environment():
    """Set up the training environment"""
    print("Setting up FOD detection training environment...")
    
    # Create necessary directories
    directories = [
        'data/models',
        'data/results',
        'runs/train',
        'configs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create basic config file
    config = {
        'model': {
            'name': 'yolov8n',
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            'patience': 20,
            'save_period': 10
        },
        'paths': {
            'data': 'data/processed/fod_yolo/data.yaml',
            'output': 'runs/train',
            'models': 'data/models'
        }
    }
    
    config_path = Path('configs/basic_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Created basic config: {config_path}")
    
    print("\nEnvironment setup complete!")
    print("\nNext steps:")
    print("1. Run: python quick_train.py --quick")
    print("2. Or run full training: python basic_training_script.py")

def main():
    parser = argparse.ArgumentParser(description='Quick FOD Detection Training')
    parser.add_argument('--quick', action='store_true', help='Run quick 10-epoch test')
    parser.add_argument('--model', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                       default='n', help='Model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for quick training')
    parser.add_argument('--batch-size', type=int, help='Batch size override')
    parser.add_argument('--dataset', type=str, help='Path to dataset YAML file')
    parser.add_argument('--validate', type=str, help='Path to model to validate')
    parser.add_argument('--list-models', action='store_true', help='List available trained models')
    parser.add_argument('--setup', action='store_true', help='Set up training environment')
    parser.add_argument('--randomize', action='store_true', help='Randomize dataset before training')
    parser.add_argument('--randomize-level', type=str, choices=['normal', 'high', 'extreme'], 
                       default='high', help='Dataset randomization level')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_training_environment()
    elif args.quick:
        results, metrics = quick_train(
            args.model, 
            args.epochs, 
            args.batch_size, 
            args.dataset,
            randomize=args.randomize,
            randomize_level=args.randomize_level
        )
        if metrics:
            print(f"\n🎉 Quick training completed!")
            print(f"Results: mAP@0.5 = {metrics['map50']:.3f}")
    elif args.validate:
        validate_existing_model(args.validate, args.dataset)
    elif args.list_models:
        list_available_models()
    else:
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  python quick_train.py --setup                  # Set up environment")
        print("  python quick_train.py --quick                  # Quick 10-epoch training")
        print("  python quick_train.py --quick --randomize      # Randomize and train")
        print("  python quick_train.py --validate <model.pt>    # Validate model")

if __name__ == "__main__":
    main()