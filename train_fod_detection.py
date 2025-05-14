#!/usr/bin/env python3
"""
Main FOD Detection Training Script
Complete training system for SUAS 2025
"""

import os
import sys
import yaml
import torch
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import Dict, Optional, Tuple

class FODTrainer:
    """FOD Detection Training Class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the FOD trainer"""
        self.config = self._load_config(config_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration"""
        default_config = {
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
                'save_period': 10,
                'workers': 4
            },
            'paths': {
                'data': 'data/processed/fod_yolo/data.yaml',
                'output': 'runs/train',
                'models': 'data/models'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
        else:
            config = default_config
            
        return config
    
    def validate_dataset(self, data_path: str) -> bool:
        """Validate that the dataset exists and has required structure"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            print(f"Error: Dataset file not found: {data_path}")
            return False
        
        # Load and validate data.yaml
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in data_config]
        
        if missing_keys:
            print(f"Error: Missing keys in data.yaml: {missing_keys}")
            return False
        
        # Check if paths exist
        base_path = Path(data_config.get('path', data_path.parent))
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        if not train_path.exists():
            print(f"Error: Training path not found: {train_path}")
            return False
        
        if not val_path.exists():
            print(f"Error: Validation path not found: {val_path}")
            return False
        
        # Count images
        train_images = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
        val_images = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
        
        print(f"Dataset validation successful:")
        print(f"  Classes: {data_config['nc']}")
        print(f"  Training images: {train_images}")
        print(f"  Validation images: {val_images}")
        
        return True
    
    def train(self, epochs=None, batch_size=None, model_name=None):
        """Execute the training process"""
        print("Starting FOD detection training...")
        
        # Override config with provided parameters
        if epochs:
            self.config['training']['epochs'] = epochs
        if batch_size:
            self.config['training']['batch_size'] = batch_size
        if model_name:
            self.config['model']['name'] = model_name
        
        # Validate dataset
        if not self.validate_dataset(self.config['paths']['data']):
            raise ValueError("Dataset validation failed")
        
        # Initialize model
        model_name = self.config['model']['name']
        print(f"Loading {model_name} model...")
        self.model = YOLO(f'{model_name}.pt')
        
        # Adjust batch size for GPU memory
        batch_size = self._adjust_batch_size()
        
        # Create training arguments
        train_args = {
            'data': self.config['paths']['data'],
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['training']['img_size'],
            'batch': batch_size,
            'lr0': self.config['training']['learning_rate'],
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            'workers': self.config['training']['workers'],
            'device': self.device.type,
            'project': self.config['paths']['output'],
            'name': f"fod_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'verbose': True,
            'seed': 42,
        }
        
        # Print training configuration
        print("\nTraining Configuration:")
        print("=" * 50)
        for key, value in train_args.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        # Start training
        try:
            results = self.model.train(**train_args)
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Save model artifacts
            self._save_model_artifacts(results, metrics)
            
            print("\nTraining completed successfully!")
            return results, metrics
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
    
    def _adjust_batch_size(self) -> int:
        """Automatically adjust batch size based on GPU memory"""
        batch_size = self.config['training']['batch_size']
        
        if self.device.type == 'cuda':
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                
                # Adjust batch size based on GPU memory
                if gpu_memory < 4:  # Less than 4GB
                    batch_size = min(batch_size, 8)
                    print(f"Adjusted batch size to {batch_size} for <4GB GPU")
                elif gpu_memory < 8:  # 4-8GB
                    batch_size = min(batch_size, 16)
                    print(f"Adjusted batch size to {batch_size} for 4-8GB GPU")
                # else: use original batch size
            except Exception as e:
                print(f"Warning: Could not determine GPU memory: {e}")
        
        return batch_size
    
    def _extract_metrics(self, results) -> Dict:
        """Extract key metrics from training results"""
        metrics = {}
        
        # Run validation to get final metrics
        val_results = self.model.val()
        
        if hasattr(val_results, 'box'):
            metrics = {
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.p.mean()),
                'recall': float(val_results.box.r.mean()),
                'final_epoch': results.epoch if hasattr(results, 'epoch') else None,
                'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else None,
            }
        
        return metrics
    
    def _save_model_artifacts(self, results, metrics: Dict):
        """Save trained model and artifacts"""
        models_dir = Path(self.config['paths']['models'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the training run directory
        run_dir = Path(results.save_dir)
        
        # Save best weights
        best_weights = run_dir / 'weights' / 'best.pt'
        if best_weights.exists():
            model_name = f"fod_{self.config['model']['name']}_best.pt"
            final_model_path = models_dir / model_name
            shutil.copy2(best_weights, final_model_path)
            print(f"Best model saved to: {final_model_path}")
        
        # Save training configuration and metrics
        config_save = {
            'training_config': self.config,
            'final_metrics': metrics,
            'training_completed': datetime.now().isoformat(),
            'model_path': str(final_model_path) if best_weights.exists() else None
        }
        
        config_path = models_dir / f"fod_{self.config['model']['name']}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_save, f, default_flow_style=False)
        
        print(f"Configuration saved to: {config_path}")


def create_default_config(output_path: str = "configs/default_training.yaml"):
    """Create a default training configuration file"""
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
            'save_period': 10,
            'workers': 4
        },
        'paths': {
            'data': 'data/processed/fod_yolo/data.yaml',
            'output': 'runs/train',
            'models': 'data/models'
        },
        'competition': {
            'target_fps': 15,
            'min_altitude': 50,
            'delivery_accuracy': 25
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Default configuration created at: {output_path}")
    return output_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train FOD Detection Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--model', type=str, choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='Model size to use')
    parser.add_argument('--create_config', action='store_true', help='Create default configuration file')
    parser.add_argument('--validate_only', type=str, help='Only validate specified model')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_path = create_default_config()
        print(f"Configuration file created at: {config_path}")
        return
    
    # Initialize trainer
    trainer = FODTrainer(config_path=args.config)
    
    # Validate only if requested
    if args.validate_only:
        try:
            trainer.model = YOLO(args.validate_only)
            val_results = trainer.model.val(data=trainer.config['paths']['data'])
            
            if hasattr(val_results, 'box'):
                print("\nValidation Results:")
                print(f"mAP@0.5: {val_results.box.map50:.4f}")
                print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
                print(f"Precision: {val_results.box.p.mean():.4f}")
                print(f"Recall: {val_results.box.r.mean():.4f}")
            return
        except Exception as e:
            print(f"Validation failed: {e}")
            return
    
    # Run training
    try:
        results, metrics = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model
        )
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {trainer.config['model']['name']}")
        if metrics:
            print(f"Final mAP@0.5: {metrics.get('map50', 'N/A'):.4f}")
            print(f"Final mAP@0.5:0.95: {metrics.get('map50_95', 'N/A'):.4f}")
            print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
        print("="*50)
        
        # Check performance
        if metrics and 'map50' in metrics:
            if metrics['map50'] > 0.7:
                print("✅ Excellent performance! (mAP@0.5 > 0.7)")
            elif metrics['map50'] > 0.5:
                print("✅ Good performance! (mAP@0.5 > 0.5)")
            elif metrics['map50'] > 0.3:
                print("⚠️  Fair performance (mAP@0.5 > 0.3)")
            else:
                print("❌ Low performance (mAP@0.5 < 0.3)")
        
        return results, metrics
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, None
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()