#!/usr/bin/env python3
"""
Quick Benchmark Script for FOD Detection
Simple interface to run common benchmarks
"""

import argparse
from pathlib import Path
import os
import time

def find_latest_model():
    """Find the most recently trained model"""
    model_paths = [
        "data/models/quick_test/fod_yolov8n_best.pt",
        "data/models/fod_yolov8n_best.pt",
        "runs/train/fod_yolov8n/weights/best.pt",
        "runs/train/fod_yolov8n_quick/weights/best.pt",
        "yolov8n.pt"  # Fallback to pretrained
    ]
    
    for path in model_paths:
        if Path(path).exists():
            return path
    
    return None

def quick_speed_test(model_path, iterations=20):
    """Run a quick speed test"""
    import torch
    from ultralytics import YOLO
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Test 640x640 (competition standard)
    print(f"\n🚀 Running speed test ({iterations} iterations)...")
    dummy_input = torch.rand(1, 3, 640, 640).to(device)
    
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Actual test
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    
    # Print results
    print("\n📊 Results:")
    print(f"  Average inference time: {avg_time*1000:.2f}ms")
    print(f"  Average FPS: {fps:.2f}")
    print(f"  Target FPS: 15")
    
    if fps >= 15:
        print(f"  ✅ SUCCESS: Model meets 15 FPS requirement!")
    else:
        print(f"  ❌ FAIL: Model needs to be faster (current: {fps:.2f} FPS)")
    
    return fps, avg_time

def quick_accuracy_test(model_path, dataset_path):
    """Run a quick accuracy test"""
    from ultralytics import YOLO
    
    print(f"\n🎯 Running accuracy test...")
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=str(Path(dataset_path) / 'data.yaml'))
    
    if hasattr(results, 'box'):
        map50 = float(results.box.map50)
        map50_95 = float(results.box.map)
        precision = float(results.box.p.mean())
        recall = float(results.box.r.mean())
        
        print(f"  mAP@0.5: {map50:.4f}")
        print(f"  mAP@0.5:0.95: {map50_95:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        if map50 > 0.5:
            print(f"  ✅ Good accuracy (mAP@0.5 > 0.5)")
        else:
            print(f"  ⚠️  Low accuracy (mAP@0.5 < 0.5)")
        
        return map50, precision, recall
    else:
        print("  ❌ Could not extract validation metrics")
        return None, None, None

def main():
    """Main function to run quick benchmarks"""
    parser = argparse.ArgumentParser(description='Quick FOD Model Benchmarking')
    parser.add_argument('--model', type=str, help='Path to model file (auto-detects if not provided)')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark (fewer iterations)')
    parser.add_argument('--simple', action='store_true', help='Run only basic speed test')
    parser.add_argument('--accuracy', action='store_true', help='Also run accuracy test')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations for speed test')
    
    args = parser.parse_args()
    
    print("🚀 Quick FOD Model Benchmark")
    print("=" * 40)
    
    # Find model if not provided
    if not args.model:
        args.model = find_latest_model()
        if not args.model:
            print("❌ No model found. Please train a model first or specify --model path")
            print("\nAvailable commands:")
            print("  python quick_train.py --quick  # Train a quick model first")
            return
        else:
            print(f"📦 Auto-detected model: {args.model}")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Set iterations
    if args.quick:
        iterations = max(10, args.iterations // 5)
        print(f"🏃‍♂️ Quick mode: {iterations} iterations")
    else:
        iterations = args.iterations
        print(f"🚶‍♂️ Normal mode: {iterations} iterations")
    
    # Run speed test
    try:
        fps, avg_time = quick_speed_test(args.model, iterations)
        
        # Run accuracy test if requested and dataset available
        if args.accuracy or not args.simple:
            dataset_path = "data/processed/fod_yolo"
            if Path(dataset_path).exists() and (Path(dataset_path) / "data.yaml").exists():
                map50, precision, recall = quick_accuracy_test(args.model, dataset_path)
            else:
                print(f"\n⚠️  Dataset not found at {dataset_path}")
                print("  Run dataset conversion first to enable accuracy testing")
        
        # Summary
        print("\n" + "=" * 40)
        print("📋 Quick Benchmark Summary")
        print("=" * 40)
        print(f"Model: {Path(args.model).name}")
        print(f"Speed: {fps:.2f} FPS ({'✅ PASS' if fps >= 15 else '❌ FAIL'})")
        if 'map50' in locals() and map50 is not None:
            print(f"Accuracy: {map50:.3f} mAP@0.5 ({'✅ GOOD' if map50 > 0.5 else '⚠️ LOW'})")
        print("=" * 40)
        
        # Recommendations
        print("\n💡 Recommendations:")
        if fps < 15:
            print("  - Try smaller model (yolov8n)")
            print("  - Consider reducing input image size")
            print("  - Check for TensorRT optimization")
        
        if 'map50' in locals() and map50 is not None and map50 < 0.5:
            print("  - Train for more epochs")
            print("  - Try larger model (yolov8s/m)")
            print("  - Verify dataset quality")
        
        if fps >= 15 and ('map50' not in locals() or map50 is None or map50 > 0.5):
            print("  - Model looks good for competition!")
            print("  - Consider running full benchmark for detailed analysis")
    
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install: pip install ultralytics torch")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()