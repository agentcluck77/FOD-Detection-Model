#!/usr/bin/env python3
"""
Basic Training Script for FOD Detection - SUAS 2025
This script handles the complete training pipeline for FOD detection using YOLOv8
"""

import os
import sys
import yaml
import torch
import shutil
import argparse
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
                'patience': 50,
                'save_period': 10,
                'workers': 4
            },
            'augmentation': {
                'mosaic': 1.0,
                'mixup': 0.15,
                'copy_paste': 0.3,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.9,
                'flipud': 0.0,
                'fliplr': 0.5
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
    
    def setup_model(self) -> None:
        """Initialize the YOLO model"""
        model_name = self.config['model']['name']
        pretrained = self.config['model']['pretrained']
        
        if pretrained:
            print(f"Loading pretrained {model_name} model...")
            self.model = YOLO(f'{model_name}.pt')
        else:
            print(f"Creating {model_name} model from scratch...")
            self.model = YOLO(f'{model_name}.yaml')
    
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
    
    def adjust_batch_size(self) -> int:
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
    
    def create_training_args(self) -> Dict:
        """Create training arguments dictionary"""
        batch_size = self.adjust_batch_size()
        
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
            'name': f"fod_{self.config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'verbose': True,
            'seed': 42,  # For reproducible results
        }
        
        # Add augmentation parameters
        train_args.update({
            'mosaic': self.config['augmentation']['mosaic'],
            'mixup': self.config['augmentation']['mixup'],
            'copy_paste': self.config['augmentation']['copy_paste'],
            'hsv_h': self.config['augmentation']['hsv_h'],
            'hsv_s': self.config['augmentation']['hsv_s'],
            'hsv_v': self.config['augmentation']['hsv_v'],
            'degrees': self.config['augmentation']['degrees'],
            'translate': self.config['augmentation']['translate'],
            'scale': self.config['augmentation']['scale'],
            'fliplr': self.config['augmentation']['fliplr'],
            'flipud': self.config['augmentation']['flipud'],
        })
        
        return train_args
    
    def create_callbacks(self):
        """Create custom callbacks for training monitoring"""
        from ultralytics.utils.callbacks import default_callbacks
        
        callbacks = default_callbacks.copy()
        
        # Custom callback to save intermediate results
        def save_training_info(trainer):
            """Save training information to JSON"""
            results_path = Path(trainer.save_dir) / 'training_info.json'
            info = {
                'current_epoch': trainer.epoch,
                'best_fitness': float(trainer.best_fitness) if trainer.best_fitness else None,
                'model_name': self.config['model']['name'],
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(results_path, 'w') as f:
                json.dump(info, f, indent=2)
        
        callbacks['on_train_epoch_end'] = save_training_info
        
        return callbacks
    
    def train(self) -> Tuple[object, Dict]:
        """Execute the training process"""
        print("Starting FOD detection training...")
        
        # Setup model
        self.setup_model()
        
        # Validate dataset
        if not self.validate_dataset(self.config['paths']['data']):
            raise ValueError("Dataset validation failed")
        
        # Create training arguments
        train_args = self.create_training_args()
        
        # Print training configuration
        print("\nTraining Configuration:")
        print("=" * 50)
        for key, value in train_args.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        # Start training
        try:
            results = self.model.train(**train_args)
            
            # Get training metrics
            metrics = self._extract_metrics(results)
            
            # Save model
            self._save_model_artifacts(results, metrics)
            
            print("\nTraining completed successfully!")
            return results, metrics
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check GPU memory usage")
            print("2. Verify dataset integrity")
            print("3. Try reducing batch size")
            print("4. Ensure CUDA drivers are up to date")
            raise
    
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
        
        # Save last weights
        last_weights = run_dir / 'weights' / 'last.pt'
        if last_weights.exists():
            model_name = f"fod_{self.config['model']['name']}_last.pt"
            final_model_path = models_dir / model_name
            shutil.copy2(last_weights, final_model_path)
        
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
    
    def validate_model(self, model_path: Optional[str] = None) -> Dict:
        """Validate trained model performance"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        if not model:
            raise ValueError("No model available for validation")
        
        print("Running model validation...")
        val_results = model.val(data=self.config['paths']['data'])
        
        if hasattr(val_results, 'box'):
            metrics = {
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.p.mean()),
                'recall': float(val_results.box.r.mean()),
            }
            
            print("\nValidation Results:")
            print("=" * 30)
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            print("=" * 30)
            
            return metrics
        
        return {}


def create_default_config(output_path: str = "configs/training_config.yaml"):
    """Create a default training configuration file"""
    config = {
        'model': {
            'name': 'yolov8n',  # or 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'learning_rate': 0.01,
            'patience': 50,
            'save_period': 10,
            'workers': 4
        },
        'augmentation': {
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.9,
            'flipud': 0.0,
            'fliplr': 0.5
        },
        'paths': {
            'data': 'data/processed/fod_yolo/data.yaml',
            'output': 'runs/train',
            'models': 'data/models'
        },
        'competition': {
            'target_fps': 15,
            'min_altitude': 50,  # feet
            'delivery_accuracy': 25  # feet
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
    
    # Override config with command line arguments
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.model:
        trainer.config['model']['name'] = args.model
    
    # Validate only if requested
    if args.validate_only:
        try:
            metrics = trainer.validate_model(args.validate_only)
            return metrics
        except Exception as e:
            print(f"Validation failed: {e}")
            return None
    
    # Run training
    try:
        results, metrics = trainer.train()
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {trainer.config['model']['name']}")
        print(f"Final mAP@0.5: {metrics.get('map50', 'N/A'):.4f}")
        print(f"Final mAP@0.5:0.95: {metrics.get('map50_95', 'N/A'):.4f}")
        print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
        print("="*50)
        
        # Check performance against requirements
        if 'map50' in metrics:
            if metrics['map50'] > 0.7:
                print("✅ Excellent performance! (mAP@0.5 > 0.7)")
            elif metrics['map50'] > 0.5:
                print("✅ Good performance! (mAP@0.5 > 0.5)")
            elif metrics['map50'] > 0.3:
                print("⚠️  Fair performance (mAP@0.5 > 0.3)")
                print("   Consider training for more epochs or using a larger model")
            else:
                print("❌ Low performance (mAP@0.5 < 0.3)")
                print("   Try: increase epochs, larger model, check data quality")
        
        return results, metrics
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, None
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return None, None


if __name__ == "__main__":
    # Set up environment
    os.environ['OMP_NUM_THREADS'] = '1'  # Reduce CPU overhead
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    main()