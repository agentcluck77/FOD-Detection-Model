#!/usr/bin/env python3
"""
Enhanced model testing script with model selection capability
"""
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os
import glob

def find_available_models():
    """Find all available YOLO models in the project"""
    model_patterns = [
        "data/models/**/*.pt",
        "runs/**/weights/*.pt",
        "*.pt"
    ]
    
    found_models = []
    for pattern in model_patterns:
        for model_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(model_path):
                found_models.append(model_path)
    
    # Add pretrained models
    pretrained_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    return found_models, pretrained_models

def list_models():
    """List all available models"""
    found_models, pretrained_models = find_available_models()
    
    print("\n=== Available Custom Models ===")
    if found_models:
        for i, model in enumerate(found_models, 1):
            print(f"{i}. {model}")
    else:
        print("No custom models found")
    
    print("\n=== Available Pretrained Models ===")
    for i, model in enumerate(pretrained_models, 1):
        print(f"{i}. {model} (will download if not present)")
    
    return found_models, pretrained_models

def select_model_interactive():
    """Interactive model selection"""
    found_models, pretrained_models = list_models()
    
    print("\nSelect a model:")
    print("0. Enter custom path")
    
    # Custom models
    for i, model in enumerate(found_models, 1):
        print(f"{i}. {model}")
    
    # Pretrained models
    offset = len(found_models)
    for i, model in enumerate(pretrained_models, offset + 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = int(input("\nEnter choice (0 for custom path): "))
            if choice == 0:
                return input("Enter model path: ")
            elif 1 <= choice <= len(found_models):
                return found_models[choice - 1]
            elif len(found_models) < choice <= len(found_models) + len(pretrained_models):
                return pretrained_models[choice - len(found_models) - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def test_model_predictions(model_path, dataset_path, num_test_images=5):
    """Test model predictions on sample images"""
    print(f"\nTesting model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    print(f"Model loaded successfully. Classes: {len(model.names)}")
    print(f"Class names: {list(model.names.values())}")
    
    # Test on training images
    train_images_dir = Path(dataset_path) / "train" / "images"
    val_images_dir = Path(dataset_path) / "val" / "images"
    
    print("\n" + "="*50)
    print("TESTING ON TRAINING IMAGES")
    print("="*50)
    train_results = test_predictions(model, train_images_dir, num_images=num_test_images)
    
    print("\n" + "="*50)
    print("TESTING ON VALIDATION IMAGES")
    print("="*50)
    val_results = test_predictions(model, val_images_dir, num_images=num_test_images)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Train images tested: {len(train_results)}")
    print(f"Val images tested: {len(val_results)}")
    print(f"Train detections: {sum(len(r['detections']) for r in train_results)}")
    print(f"Val detections: {sum(len(r['detections']) for r in val_results)}")
    
    # Check if model is detecting anything
    total_detections = sum(len(r['detections']) for r in train_results + val_results)
    if total_detections == 0:
        print("\n⚠️  WARNING: NO DETECTIONS FOUND!")
        print("This suggests a serious issue with:")
        print("1. Model training (very low mAP)")
        print("2. Label format problems")
        print("3. Class mismatch")
        
        # Test with very low confidence
        print("\nTrying with confidence=0.001...")
        test_predictions(model, train_images_dir, num_images=1, conf_thresh=0.001)
    else:
        print(f"\n✅ Model is making detections (total: {total_detections})")
    
    return model

def test_predictions(model, image_dir, num_images=5, conf_thresh=0.01):
    """Test predictions on images from a directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(ext)))
    
    image_files = image_files[:num_images]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    results = []
    for img_path in image_files:
        print(f"\nTesting: {img_path.name}")
        
        # Run prediction
        predictions = model(str(img_path), conf=conf_thresh, verbose=False)
        
        # Process results
        detections = []
        if predictions[0].boxes is not None:
            boxes = predictions[0].boxes
            print(f"  Detections: {len(boxes)}")
            for i, box in enumerate(boxes):
                conf = box.conf.item()
                cls = int(box.cls.item())
                cls_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                detections.append({
                    'class': cls,
                    'class_name': cls_name,
                    'confidence': conf
                })
                print(f"    {i+1}. {cls_name} (conf: {conf:.3f})")
        else:
            print("  No detections")
        
        # Save annotated image
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"test_{img_path.stem}.jpg"
        annotated = predictions[0].plot()
        cv2.imwrite(str(output_path), annotated)
        print(f"  Saved: {output_path}")
        
        results.append({
            'image': img_path.name,
            'detections': detections
        })
    
    return results

def check_model_info(model_path):
    """Check detailed model information"""
    print(f"\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    model = YOLO(model_path)
    
    # Model architecture info
    print(f"Model type: {type(model.model)}")
    print(f"Model device: {next(model.model.parameters()).device}")
    print(f"Number of classes: {len(model.names)}")
    
    # Training info if available
    try:
        if hasattr(model, 'ckpt') and model.ckpt:
            print(f"Epoch: {model.ckpt.get('epoch', 'N/A')}")
            print(f"Best fitness: {model.ckpt.get('best_fitness', 'N/A')}")
    except Exception as e:
        print(f"Could not retrieve training info: {e}")
    
    # Check if it's a pretrained model or trained on custom data
    if hasattr(model, 'model') and hasattr(model.model, 'nc'):
        print(f"Model classes (nc): {model.model.nc}")
    
    # Display class names
    print(f"Class names: {list(model.names.values())}")
    
    return model

def diagnose_low_map(model_path, dataset_path):
    """Diagnose potential causes of low mAP"""
    print(f"\n" + "="*50)
    print("LOW mAP DIAGNOSIS")
    print("="*50)
    
    # Check data.yaml
    data_yaml = Path(dataset_path) / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, 'r') as f:
            content = f.read()
            print("data.yaml content:")
            print(content)
    else:
        print("⚠️  data.yaml not found!")
    
    # Check label samples
    train_labels = Path(dataset_path) / "train" / "labels"
    if train_labels.exists():
        print(f"\nSample labels from {train_labels}:")
        label_files = list(train_labels.glob("*.txt"))[:3]
        for label_file in label_files:
            print(f"\n{label_file.name}:")
            with open(label_file, 'r') as f:
                lines = f.readlines()[:3]  # First 3 lines
                for line in lines:
                    print(f"  {line.strip()}")
    
    # Validate model
    model = YOLO(model_path)
    
    # Check if model was actually trained on this data
    print(f"\nModel info:")
    print(f"- Classes in model: {len(model.names)}")
    print(f"- Classes in data.yaml: {data_yaml.exists() and 'Check above' or 'N/A'}")
    
    # Simple validation test
    print("\nRunning validation...")
    try:
        val_results = model.val(data=str(data_yaml), verbose=False)
        print(f"Validation mAP50: {val_results.box.map50:.3f}")
        print(f"Validation mAP50-95: {val_results.box.map:.3f}")
    except Exception as e:
        print(f"Validation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test YOLO model performance")
    parser.add_argument("--model", type=str, help="Path to model file (.pt)")
    parser.add_argument("--dataset", type=str, default="data/processed/fod_yolo", 
                       help="Path to dataset directory")
    parser.add_argument("--images", type=int, default=5, 
                       help="Number of test images per split")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--interactive", action="store_true", help="Interactive model selection")
    parser.add_argument("--diagnose", action="store_true", help="Run low mAP diagnosis")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    # Select model
    if args.interactive:
        model_path = select_model_interactive()
    elif args.model:
        model_path = args.model
    else:
        # Try to find a model automatically
        found_models, _ = find_available_models()
        if found_models:
            model_path = found_models[0]
            print(f"Using automatically found model: {model_path}")
        else:
            print("No model specified. Use --model, --interactive, or --list")
            return
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Check dataset
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        return
    
    # Run tests
    check_model_info(model_path)
    
    if args.diagnose:
        diagnose_low_map(model_path, args.dataset)
    
    test_model_predictions(model_path, args.dataset, args.images)
    
    print(f"\nTest complete! Check the 'test_outputs' directory for annotated images.")

if __name__ == "__main__":
    main()
