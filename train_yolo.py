#!/usr/bin/env python3
"""
Basic YOLO training script for FOD detection
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_yolo_data_yaml(dataset_path, output_path):
    """
    Create YOLO format data.yaml file from Pascal VOC dataset
    """
    # First, we need to convert Pascal VOC to YOLO format
    # For now, let's create a basic data.yaml structure
    
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {
            # We'll populate this after analyzing the dataset
            # For now, let's use some common FOD classes
            0: 'person',
            1: 'car',
            2: 'suitcase',
            3: 'sports_ball',
            4: 'bottle',
            5: 'umbrella',
            6: 'stop_sign',
            7: 'airplane',
            8: 'motorcycle',
            9: 'bicycle',
            10: 'bus',
            11: 'boat',
            12: 'snowboard',
            13: 'baseball_bat',
            14: 'tennis_racket',
            15: 'skis',
            16: 'bed'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return data_yaml

def convert_voc_to_yolo(voc_path, yolo_path):
    """
    Convert Pascal VOC format to YOLO format
    This is a placeholder - you'll need to implement the actual conversion
    """
    print(f"Converting Pascal VOC dataset from {voc_path} to YOLO format at {yolo_path}")
    print("Note: This function needs to be implemented based on your specific dataset structure")
    
    # For now, create basic directory structure
    yolo_path.mkdir(parents=True, exist_ok=True)
    (yolo_path / 'train').mkdir(exist_ok=True)
    (yolo_path / 'val').mkdir(exist_ok=True)
    (yolo_path / 'test').mkdir(exist_ok=True)
    
    # Each subdirectory should have 'images' and 'labels' folders
    for split in ['train', 'val', 'test']:
        (yolo_path / split / 'images').mkdir(exist_ok=True)
        (yolo_path / split / 'labels').mkdir(exist_ok=True)

def train_yolo_model(data_yaml_path, model_size='n', epochs=100, img_size=640):
    """
    Train YOLO model with FOD dataset
    """
    print(f"Training YOLOv8{model_size} on FOD dataset...")
    
    # Initialize the model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Training configuration
    train_args = {
        'data': str(data_yaml_path),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': 16,  # Adjust based on your GPU memory
        'workers': 4,
        'device': device,
        'project': 'runs/train',
        'name': f'fod_yolov8{model_size}',
        'save_period': 10,  # Save checkpoint every 10 epochs
    }
    
    # Start training
    results = model.train(**train_args)
    
    # Evaluate on validation set
    print("Evaluating model...")
    results = model.val()
    
    # Save the best model
    model_save_path = Path('data/models/fod_yolov8_best.pt')
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_path))
    
    print(f"Training complete! Best model saved to {model_save_path}")
    return results

def main():
    """Main training function"""
    # Set up paths
    base_dir = Path("~/Desktop/SUAS").expanduser()
    dataset_path = base_dir / "data/raw/fod_a_data"
    yolo_data_path = base_dir / "data/processed/fod_yolo"
    data_yaml_path = yolo_data_path / "data.yaml"
    
    print("Starting YOLO training for FOD detection...")
    print(f"Dataset path: {dataset_path}")
    print(f"YOLO data path: {yolo_data_path}")
    
    # Step 1: Convert Pascal VOC to YOLO format (placeholder)
    print("\nStep 1: Converting dataset format...")
    convert_voc_to_yolo(dataset_path, yolo_data_path)
    
    # Step 2: Create data.yaml file
    print("\nStep 2: Creating data.yaml file...")
    setup_yolo_data_yaml(yolo_data_path, data_yaml_path)
    
    # Step 3: Start training
    print("\nStep 3: Starting model training...")
    # First, let's try with a small model for testing
    results = train_yolo_model(data_yaml_path, model_size='n', epochs=10)
    
    print("\nTraining complete!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()