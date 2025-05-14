# FOD Detection Model Benchmarking Guide

This guide explains how to use the benchmarking tools to evaluate your trained models.

## Quick Benchmark

The `quick_benchmark.py` script provides a fast assessment of model speed and accuracy.

### Basic Usage

```bash
# Run with default settings
python quick_benchmark.py

# Run quick test with fewer iterations
python quick_benchmark.py --quick

# Specify a specific model file
python quick_benchmark.py --model runs/train/fod_yolov8n_quick/weights/best.pt

# Run speed test only
python quick_benchmark.py --simple

# Specify number of iterations
python quick_benchmark.py --iterations 100
```

The quick benchmark will:
- Auto-detect your latest trained model
- Run inference speed tests
- Measure FPS (Frames Per Second)
- Check if model meets competition requirements (15+ FPS)
- Validate accuracy if dataset is available
- Provide recommendations based on results

## Comprehensive Benchmark

The `fod_benchmark.py` script provides detailed performance analysis.

### Basic Usage

```bash
# Run full benchmark with all tests
python fod_benchmark.py --model runs/train/fod_yolov8n_quick/weights/best.pt

# Run benchmark and specify dataset path
python fod_benchmark.py --model yolov8n.pt --dataset data/processed/fod_yolo

# Save benchmark results to a specific file
python fod_benchmark.py --model yolov8n.pt --output benchmark_results.json

# Skip real image testing
python fod_benchmark.py --model yolov8n.pt --no-images

# Run with fewer iterations for faster results
python fod_benchmark.py --model yolov8n.pt --quick
```

The comprehensive benchmark will:
- Collect detailed system information
- Test multiple image resolutions
- Measure GPU and CPU memory usage
- Validate model on the dataset
- Test on real-world images
- Generate performance visualizations
- Create a detailed benchmark report
- Save results to JSON file

## Interpreting Results

### Speed
- **Target**: 15+ FPS for competition
- **Recommendation**: If speed is low, try smaller models or reduce image size

### Accuracy
- **mAP@0.5**: Overall detection accuracy (higher is better)
- **mAP@0.5:0.95**: More rigorous accuracy metric across multiple IoU thresholds
- **Precision**: Ratio of correct positive predictions to total positive predictions
- **Recall**: Ratio of correct positive predictions to all actual positives

### Memory Usage
- Important for deployment on resource-constrained systems
- Check if model fits within available GPU memory

## Tips for Better Performance

1. Try smaller models (yolov8n) for faster inference
2. Reduce batch size if experiencing OOM errors
3. Use half-precision (FP16) on supported GPUs
4. Consider model optimization (pruning, quantization)
5. Benchmark on hardware similar to what will be used in competition 