#!/usr/bin/env python3
"""
Fixed FOD Detection Benchmarking Script
Comprehensive performance testing for SUAS 2025 competition
"""

import time
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
import psutil
import cpuinfo

# Custom JSON encoder to handle numpy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle single numpy scalars
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

class FODBenchmark:
    """Comprehensive benchmark suite for FOD detection models"""
    
    def __init__(self, model_path: str, dataset_path: str = None):
        """Initialize benchmark with model and optional dataset"""
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.load_model()
        
        # Competition requirements
        self.target_fps = 15
        self.min_altitude = 50  # feet
        self.delivery_accuracy = 25  # feet
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(str(self.model_path))
            print(f"✅ Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def get_system_info(self) -> Dict:
        """Collect system information"""
        print("\n📊 Collecting system information...")
        
        # CPU information
        cpu_info = cpuinfo.get_cpu_info()
        
        # GPU information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'compute_capability': torch.cuda.get_device_capability(0),
            }
        
        # System memory
        memory = psutil.virtual_memory()
        
        system_info = {
            'python_version': cpu_info.get('python_version', 'Unknown'),
            'platform': cpu_info.get('platform', 'Unknown'),
            'cpu': {
                'brand': cpu_info.get('brand_raw', 'Unknown'),
                'arch': cpu_info.get('arch', 'Unknown'),
                'cores': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
            },
            'memory': {
                'total_gb': memory.total / 1024**3,
                'available_gb': memory.available / 1024**3,
                'percent_used': memory.percent
            },
            'gpu': gpu_info,
            'pytorch_version': torch.__version__,
            'device': str(self.device)
        }
        
        self.results['system_info'] = system_info
        return system_info
    
    def benchmark_inference_speed(self, num_iterations: int = 100, image_sizes: List[Tuple[int, int]] = None) -> Dict:
        """Benchmark inference speed at different image sizes"""
        print("\n🚀 Benchmarking inference speed...")
        
        if image_sizes is None:
            image_sizes = [(320, 320), (416, 416), (640, 640), (832, 832)]
        
        speed_results = {}
        
        for width, height in image_sizes:
            print(f"  Testing {width}x{height}...")
            
            # Create dummy input
            dummy_input = torch.rand(1, 3, height, width).to(self.device)
            
            # Warm up
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Actual benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = 1.0 / avg_time
            
            speed_results[f"{width}x{height}"] = {
                'avg_time_ms': float(avg_time * 1000),
                'fps': float(fps),
                'meets_requirement': fps >= self.target_fps
            }
            
            print(f"    Average: {avg_time*1000:.2f}ms, FPS: {fps:.2f}")
        
        self.results['inference_speed'] = speed_results
        return speed_results
    
    def benchmark_memory_usage(self) -> Dict:
        """Monitor GPU and CPU memory usage"""
        print("\n💾 Benchmarking memory usage...")
        
        memory_results = {}
        
        # CPU Memory
        process = psutil.Process()
        cpu_memory_before = process.memory_info().rss / 1024**2  # MB
        
        # GPU Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Run inference
        dummy_input = torch.rand(4, 3, 640, 640).to(self.device)
        
        with torch.no_grad():
            for _ in range(50):
                _ = self.model(dummy_input)
        
        # Measure memory after
        cpu_memory_after = process.memory_info().rss / 1024**2  # MB
        cpu_memory_used = cpu_memory_after - cpu_memory_before
        
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
            gpu_memory_used = gpu_memory_after - gpu_memory_before
            
            memory_results['gpu'] = {
                'memory_before_mb': float(gpu_memory_before),
                'memory_after_mb': float(gpu_memory_after),
                'memory_used_mb': float(gpu_memory_used),
                'memory_peak_mb': float(gpu_memory_peak),
                'memory_total_mb': float(torch.cuda.get_device_properties(0).total_memory / 1024**2)
            }
        
        memory_results['cpu'] = {
            'memory_before_mb': float(cpu_memory_before),
            'memory_after_mb': float(cpu_memory_after),
            'memory_used_mb': float(cpu_memory_used)
        }
        
        self.results['memory_usage'] = memory_results
        print(f"  CPU Memory Used: {cpu_memory_used:.2f} MB")
        if torch.cuda.is_available():
            print(f"  GPU Memory Used: {gpu_memory_used:.2f} MB")
            print(f"  GPU Memory Peak: {gpu_memory_peak:.2f} MB")
        
        return memory_results
    
    def validate_on_dataset(self) -> Dict:
        """Validate model on the test dataset"""
        print("\n🎯 Validating model on dataset...")
        
        if not self.dataset_path or not (self.dataset_path / 'data.yaml').exists():
            print("  ⚠️  Dataset not available, skipping validation")
            return {}
        
        # Run validation
        val_results = self.model.val(data=str(self.dataset_path / 'data.yaml'))
        
        if hasattr(val_results, 'box'):
            validation_results = {
                'map50': float(val_results.box.map50),
                'map50_95': float(val_results.box.map),
                'precision': float(val_results.box.p.mean()),
                'recall': float(val_results.box.r.mean()),
                'fitness': float(val_results.fitness) if hasattr(val_results, 'fitness') else None
            }
            
            print(f"  mAP@0.5: {validation_results['map50']:.4f}")
            print(f"  mAP@0.5:0.95: {validation_results['map50_95']:.4f}")
            print(f"  Precision: {validation_results['precision']:.4f}")
            print(f"  Recall: {validation_results['recall']:.4f}")
            
            self.results['validation'] = validation_results
            return validation_results
        else:
            print("  ❌ Could not extract validation metrics")
            return {}
    
    def test_real_images(self, test_image_dir: str = None, num_images: int = 20) -> Dict:
        """Test on real images if available"""
        print("\n📸 Testing on real images...")
        
        if not test_image_dir:
            # Try to find test images from dataset
            if self.dataset_path:
                test_image_dir = self.dataset_path / 'test' / 'images'
        
        if not test_image_dir or not Path(test_image_dir).exists():
            print("  ⚠️  No test images found, skipping real image test")
            return {}
        
        test_image_dir = Path(test_image_dir)
        image_files = list(test_image_dir.glob('*.jpg')) + list(test_image_dir.glob('*.png'))
        
        if not image_files:
            print("  ⚠️  No images found in test directory")
            return {}
        
        # Limit number of images
        image_files = image_files[:num_images]
        
        real_image_results = {
            'total_images': len(image_files),
            'inference_times': [],
            'detection_counts': [],
            'average_confidence': []
        }
        
        print(f"  Testing on {len(image_files)} images...")
        
        for img_file in image_files:
            # Load and run inference
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            start_time = time.time()
            results = self.model(img)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            real_image_results['inference_times'].append(inference_time)
            
            # Extract detection information
            if len(results) > 0 and results[0].boxes is not None:
                detections = len(results[0].boxes)
                confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []
                avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
            else:
                detections = 0
                avg_conf = 0
            
            real_image_results['detection_counts'].append(detections)
            real_image_results['average_confidence'].append(avg_conf)
        
        # Calculate summary statistics
        real_image_results['summary'] = {
            'avg_inference_time_ms': float(np.mean(real_image_results['inference_times'])),
            'avg_fps': float(1000 / np.mean(real_image_results['inference_times'])),
            'avg_detections_per_image': float(np.mean(real_image_results['detection_counts'])),
            'avg_confidence': float(np.mean(real_image_results['average_confidence'])),
            'min_fps': float(1000 / max(real_image_results['inference_times'])),
            'max_fps': float(1000 / min(real_image_results['inference_times']))
        }
        
        summary = real_image_results['summary']
        print(f"  Average inference time: {summary['avg_inference_time_ms']:.2f}ms")
        print(f"  Average FPS: {summary['avg_fps']:.2f}")
        print(f"  Average detections per image: {summary['avg_detections_per_image']:.1f}")
        print(f"  Average confidence: {summary['avg_confidence']:.3f}")
        
        self.results['real_images'] = real_image_results
        return real_image_results
    
    def generate_benchmark_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        print("\n📊 Generating benchmark report...")
        
        # Competition compliance check
        compliance = {
            'fps_requirement_met': False,
            'memory_efficient': False,
            'accuracy_acceptable': False
        }
        
        # Check FPS requirement (15+ FPS for 640x640)
        if 'inference_speed' in self.results:
            fps_640 = self.results['inference_speed'].get('640x640', {}).get('fps', 0)
            compliance['fps_requirement_met'] = fps_640 >= self.target_fps
        
        # Check memory efficiency (< 4GB GPU memory)
        if 'memory_usage' in self.results and 'gpu' in self.results['memory_usage']:
            gpu_peak = self.results['memory_usage']['gpu']['memory_peak_mb']
            compliance['memory_efficient'] = gpu_peak < 4000  # 4GB in MB
        
        # Check accuracy (mAP@0.5 > 0.5)
        if 'validation' in self.results:
            map50 = self.results['validation'].get('map50', 0)
            compliance['accuracy_acceptable'] = map50 > 0.5
        
        # Create summary
        summary = {
            'model_path': str(self.model_path),
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'competition_compliance': compliance,
            'key_metrics': {}
        }
        
        # Extract key metrics
        if 'inference_speed' in self.results:
            fps_640 = self.results['inference_speed'].get('640x640', {}).get('fps', 0)
            summary['key_metrics']['fps_640x640'] = float(fps_640)
        
        if 'validation' in self.results:
            summary['key_metrics']['map50'] = float(self.results['validation']['map50'])
            summary['key_metrics']['map50_95'] = float(self.results['validation']['map50_95'])
        
        if 'memory_usage' in self.results and 'gpu' in self.results['memory_usage']:
            summary['key_metrics']['gpu_memory_peak_mb'] = float(self.results['memory_usage']['gpu']['memory_peak_mb'])
        
        self.results['summary'] = summary
        return summary
    
    def visualize_results(self, save_plots: bool = True) -> None:
        """Create visualization plots of benchmark results"""
        print("\n📈 Creating visualization plots...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Inference Speed by Image Size
        if 'inference_speed' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            sizes = []
            fps_values = []
            for size, metrics in self.results['inference_speed'].items():
                sizes.append(size)
                fps_values.append(metrics['fps'])
            
            bars = ax1.bar(sizes, fps_values, color=['green' if fps >= self.target_fps else 'red' for fps in fps_values])
            ax1.axhline(y=self.target_fps, color='orange', linestyle='--', label=f'Target: {self.target_fps} FPS')
            ax1.set_ylabel('FPS')
            ax1.set_title('Inference Speed by Image Size')
            ax1.legend()
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, fps in zip(bars, fps_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{fps:.1f}', ha='center', va='bottom')
        
        # 2. Memory Usage
        if 'memory_usage' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            memory_types = []
            memory_values = []
            
            if 'gpu' in self.results['memory_usage']:
                memory_types.extend(['GPU Used', 'GPU Peak'])
                memory_values.extend([
                    self.results['memory_usage']['gpu']['memory_used_mb'],
                    self.results['memory_usage']['gpu']['memory_peak_mb']
                ])
            
            if 'cpu' in self.results['memory_usage']:
                memory_types.append('CPU Used')
                memory_values.append(self.results['memory_usage']['cpu']['memory_used_mb'])
            
            ax2.bar(memory_types, memory_values, color=['blue', 'navy', 'green'])
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title('Memory Usage')
            plt.xticks(rotation=45)
        
        # 3. Validation Metrics
        if 'validation' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
            values = [
                self.results['validation']['map50'],
                self.results['validation']['map50_95'],
                self.results['validation']['precision'],
                self.results['validation']['recall']
            ]
            
            bars = ax3.bar(metrics, values, color='skyblue')
            ax3.set_ylabel('Score')
            ax3.set_title('Validation Metrics')
            ax3.set_ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Real Image Testing Results
        if 'real_images' in self.results and 'summary' in self.results['real_images']:
            ax4 = plt.subplot(2, 3, 4)
            summary = self.results['real_images']['summary']
            
            # Create FPS distribution histogram
            inference_times = self.results['real_images']['inference_times']
            fps_values = [1000/t for t in inference_times]
            
            ax4.hist(fps_values, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            ax4.axvline(summary['avg_fps'], color='red', linestyle='--', linewidth=2, label=f'Average: {summary["avg_fps"]:.1f} FPS')
            ax4.axvline(self.target_fps, color='orange', linestyle='--', linewidth=2, label=f'Target: {self.target_fps} FPS')
            ax4.set_xlabel('FPS')
            ax4.set_ylabel('Count')
            ax4.set_title('Real Image FPS Distribution')
            ax4.legend()
        
        # 5. Competition Compliance Summary
        ax5 = plt.subplot(2, 3, 5)
        if 'summary' in self.results and 'competition_compliance' in self.results['summary']:
            compliance = self.results['summary']['competition_compliance']
            requirements = ['FPS ≥ 15', 'Memory < 4GB', 'mAP@0.5 > 0.5']
            statuses = [
                compliance['fps_requirement_met'],
                compliance['memory_efficient'],
                compliance['accuracy_acceptable']
            ]
            colors = ['green' if status else 'red' for status in statuses]
            
            bars = ax5.bar(requirements, [1]*3, color=colors, alpha=0.7)
            ax5.set_title('Competition Requirements Compliance')
            ax5.set_ylabel('Pass/Fail')
            ax5.set_ylim(0, 1.2)
            
            # Add pass/fail labels
            for bar, status in zip(bars, statuses):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                        'PASS' if status else 'FAIL', ha='center', va='center',
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = Path('data/results/benchmark')
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f'benchmark_results_{time.strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  📊 Plots saved to: {plot_path}")
            
        plt.show()
    
    def save_results(self, output_file: str = None) -> str:
        """Save benchmark results to JSON file with proper serialization"""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path('data/results/benchmark')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'benchmark_results_{timestamp}.json'
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean the results to ensure JSON serialization
        def clean_for_json(data):
            if isinstance(data, dict):
                return {key: clean_for_json(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            elif isinstance(data, (np.integer, np.int32, np.int64)):
                return int(data)
            elif isinstance(data, (np.floating, np.float32, np.float64)):
                return float(data)
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif hasattr(data, 'item'):  # Handle single numpy scalars
                return data.item()
            else:
                return data
        
        # Clean the results
        cleaned_results = clean_for_json(self.results)
        
        # Save with custom encoder
        with open(output_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"📄 Results saved to: {output_file}")
        return str(output_file)
    
    def run_full_benchmark(self, include_real_images: bool = True) -> Dict:
        """Run the complete benchmark suite"""
        print("🚀 Starting Full FOD Detection Benchmark")
        print("=" * 50)
        
        # Collect system information
        self.get_system_info()
        
        # Benchmark inference speed
        self.benchmark_inference_speed()
        
        # Monitor memory usage
        self.benchmark_memory_usage()
        
        # Validate on dataset
        self.validate_on_dataset()
        
        # Test on real images
        if include_real_images:
            self.test_real_images()
        
        # Generate report
        self.generate_benchmark_report()
        
        # Create visualizations (only if not disabled)
        if not hasattr(self, 'skip_plots') or not self.skip_plots:
            self.visualize_results()
        
        # Save results
        results_file = self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary to console"""
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        
        if 'summary' in self.results:
            summary = self.results['summary']
            
            print(f"Model: {summary['model_path']}")
            print(f"Timestamp: {summary['benchmark_timestamp']}")
            
            if 'key_metrics' in summary:
                metrics = summary['key_metrics']
                if 'fps_640x640' in metrics:
                    fps = metrics['fps_640x640']
                    status = "✅ PASS" if fps >= self.target_fps else "❌ FAIL"
                    print(f"Inference Speed (640x640): {fps:.2f} FPS {status}")
                
                if 'map50' in metrics:
                    print(f"mAP@0.5: {metrics['map50']:.4f}")
                
                if 'gpu_memory_peak_mb' in metrics:
                    mem = metrics['gpu_memory_peak_mb']
                    print(f"GPU Memory Peak: {mem:.0f} MB")
            
            print("\nCompetition Requirements:")
            if 'competition_compliance' in summary:
                compliance = summary['competition_compliance']
                print(f"  ✅ FPS ≥ 15: {'PASS' if compliance['fps_requirement_met'] else 'FAIL'}")
                print(f"  ✅ Memory < 4GB: {'PASS' if compliance['memory_efficient'] else 'FAIL'}")
                print(f"  ✅ mAP@0.5 > 0.5: {'PASS' if compliance['accuracy_acceptable'] else 'FAIL'}")
        
        print("=" * 60)


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description='Benchmark FOD Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt)')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory (optional)')
    parser.add_argument('--output', type=str, help='Output file for results (optional)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for speed test')
    parser.add_argument('--no-real-images', action='store_true', help='Skip real image testing')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Validate model file
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Create benchmark instance
    try:
        benchmark = FODBenchmark(
            model_path=args.model,
            dataset_path=args.dataset
        )
    except Exception as e:
        print(f"❌ Failed to initialize benchmark: {e}")
        return
    
    # Run benchmark
    try:
        results = benchmark.run_full_benchmark(
            include_real_images=not args.no_real_images
        )
        
        # Save results if output file specified
        if args.output:
            benchmark.save_results(args.output)
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()