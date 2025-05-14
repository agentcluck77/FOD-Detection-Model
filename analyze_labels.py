#!/usr/bin/env python3
"""
Analyze YOLO label files to diagnose low mAP issues
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def analyze_yolo_labels(dataset_path):
    """Comprehensive analysis of YOLO label files"""
    results = {
        'train': analyze_split(dataset_path, 'train'),
        'val': analyze_split(dataset_path, 'val')
    }
    
    print("\n" + "="*50)
    print("YOLO LABEL ANALYSIS REPORT")
    print("="*50)
    
    for split, data in results.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Total files: {data['total_files']}")
        print(f"  Empty files: {data['empty_files']}")
        print(f"  Files with objects: {data['files_with_objects']}")
        print(f"  Total objects: {data['total_objects']}")
        print(f"  Average objects per image: {data['avg_objects_per_image']:.2f}")
        print(f"  Unique classes: {data['unique_classes']}")
        print(f"  Class distribution: {dict(data['class_counts'].most_common(10))}")
        
        if data['coordinate_issues']:
            print(f"  ⚠️  COORDINATE ISSUES: {len(data['coordinate_issues'])}")
            for issue in data['coordinate_issues'][:5]:
                print(f"    - {issue}")
        
        if data['format_issues']:
            print(f"  ⚠️  FORMAT ISSUES: {len(data['format_issues'])}")
            for issue in data['format_issues'][:5]:
                print(f"    - {issue}")
    
    # Create visualizations
    create_analysis_plots(results)
    
    return results

def analyze_split(dataset_path, split):
    """Analyze a specific split (train/val)"""
    labels_dir = Path(dataset_path) / split / "labels"
    
    data = {
        'total_files': 0,
        'empty_files': 0,
        'files_with_objects': 0,
        'total_objects': 0,
        'class_counts': Counter(),
        'coordinate_issues': [],
        'format_issues': [],
        'objects_per_image': [],
        'unique_classes': set(),
        'bbox_sizes': [],
        'bbox_centers': []
    }
    
    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} does not exist!")
        return data
    
    label_files = list(labels_dir.glob("*.txt"))
    data['total_files'] = len(label_files)
    
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            data['empty_files'] += 1
            continue
        
        objects_in_file = 0
        data['files_with_objects'] += 1
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # Check format
            if len(parts) != 5:
                data['format_issues'].append(f"{label_file.name}:{line_num} - Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate coordinates
                if not (0 <= x_center <= 1):
                    data['coordinate_issues'].append(f"{label_file.name}:{line_num} - Invalid x_center: {x_center}")
                if not (0 <= y_center <= 1):
                    data['coordinate_issues'].append(f"{label_file.name}:{line_num} - Invalid y_center: {y_center}")
                if not (0 < width <= 1):
                    data['coordinate_issues'].append(f"{label_file.name}:{line_num} - Invalid width: {width}")
                if not (0 < height <= 1):
                    data['coordinate_issues'].append(f"{label_file.name}:{line_num} - Invalid height: {height}")
                
                # Collect statistics
                data['total_objects'] += 1
                objects_in_file += 1
                data['class_counts'][class_id] += 1
                data['unique_classes'].add(class_id)
                data['bbox_sizes'].append((width, height))
                data['bbox_centers'].append((x_center, y_center))
                
            except ValueError as e:
                data['format_issues'].append(f"{label_file.name}:{line_num} - Cannot parse numbers: {str(e)}")
        
        data['objects_per_image'].append(objects_in_file)
    
    # Calculate averages
    if data['files_with_objects'] > 0:
        data['avg_objects_per_image'] = data['total_objects'] / data['files_with_objects']
    else:
        data['avg_objects_per_image'] = 0
    
    return data

def create_analysis_plots(results):
    """Create visualization plots for the analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLO Dataset Analysis', fontsize=16)
    
    # Plot 1: Objects per image distribution
    ax1 = axes[0, 0]
    train_objects = results['train']['objects_per_image']
    val_objects = results['val']['objects_per_image']
    
    ax1.hist(train_objects, bins=30, alpha=0.7, label='Train', color='blue')
    ax1.hist(val_objects, bins=30, alpha=0.7, label='Val', color='orange')
    ax1.set_xlabel('Objects per Image')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Objects per Image Distribution')
    ax1.legend()
    
    # Plot 2: Class distribution
    ax2 = axes[0, 1]
    train_classes = results['train']['class_counts']
    class_ids = sorted(train_classes.keys())
    counts = [train_classes[cid] for cid in class_ids]
    
    bars = ax2.bar(class_ids, counts)
    ax2.set_xlabel('Class ID')
    ax2.set_ylabel('Count')
    ax2.set_title('Class Distribution (Train)')
    
    # Color bars differently for better visibility
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Set3(i % 12))
    
    # Plot 3: Bounding box sizes
    ax3 = axes[0, 2]
    bbox_sizes = results['train']['bbox_sizes']
    if bbox_sizes:
        widths = [size[0] for size in bbox_sizes]
        heights = [size[1] for size in bbox_sizes]
        scatter = ax3.scatter(widths, heights, alpha=0.6, s=1)
        ax3.set_xlabel('Width (normalized)')
        ax3.set_ylabel('Height (normalized)')
        ax3.set_title('Bounding Box Sizes')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # Plot 4: Bounding box centers
    ax4 = axes[1, 0]
    bbox_centers = results['train']['bbox_centers']
    if bbox_centers:
        x_centers = [center[0] for center in bbox_centers]
        y_centers = [center[1] for center in bbox_centers]
        hist, xedges, yedges = np.histogram2d(x_centers, y_centers, bins=50)
        im = ax4.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot')
        ax4.set_xlabel('X Center (normalized)')
        ax4.set_ylabel('Y Center (normalized)')
        ax4.set_title('Bounding Box Center Distribution')
        plt.colorbar(im, ax=ax4)
    
    # Plot 5: Data quality summary
    ax5 = axes[1, 1]
    quality_metrics = {
        'Train': {
            'Total Images': results['train']['total_files'],
            'With Objects': results['train']['files_with_objects'],
            'Empty Labels': results['train']['empty_files'],
            'Coord Issues': len(results['train']['coordinate_issues']),
            'Format Issues': len(results['train']['format_issues'])
        },
        'Val': {
            'Total Images': results['val']['total_files'],
            'With Objects': results['val']['files_with_objects'],
            'Empty Labels': results['val']['empty_files'],
            'Coord Issues': len(results['val']['coordinate_issues']),
            'Format Issues': len(results['val']['format_issues'])
        }
    }
    
    ax5.axis('off')
    y_pos = 0.9
    ax5.text(0.1, y_pos, 'DATA QUALITY SUMMARY', fontsize=14, fontweight='bold')
    y_pos -= 0.15
    
    for split, metrics in quality_metrics.items():
        ax5.text(0.1, y_pos, f'{split}:', fontsize=12, fontweight='bold')
        y_pos -= 0.08
        for metric, value in metrics.items():
            ax5.text(0.15, y_pos, f'{metric}: {value}', fontsize=10)
            y_pos -= 0.06
        y_pos -= 0.02
    
    # Plot 6: Issues summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    y_pos = 0.9
    ax6.text(0.1, y_pos, 'POTENTIAL ISSUES', fontsize=14, fontweight='bold')
    y_pos -= 0.15
    
    # Check for common issues
    issues = []
    
    # Check for class imbalance
    train_classes = results['train']['class_counts']
    if train_classes:
        most_common = train_classes.most_common(1)[0][1]
        least_common = train_classes.most_common()[-1][1]
        if most_common > least_common * 100:
            issues.append("⚠️ Severe class imbalance detected")
    
    # Check for small objects
    bbox_sizes = results['train']['bbox_sizes']
    if bbox_sizes:
        small_boxes = sum(1 for w, h in bbox_sizes if w * h < 0.001)
        if small_boxes > len(bbox_sizes) * 0.1:
            issues.append(f"⚠️ {small_boxes} very small objects (< 0.1% of image)")
    
    # Check for edge objects
    bbox_centers = results['train']['bbox_centers']
    if bbox_centers:
        edge_boxes = sum(1 for x, y in bbox_centers 
                        if x < 0.05 or x > 0.95 or y < 0.05 or y > 0.95)
        if edge_boxes > len(bbox_centers) * 0.05:
            issues.append(f"⚠️ {edge_boxes} objects near image edges")
    
    # Check for missing classes
    missing_classes = []
    if train_classes:
        max_class = max(train_classes.keys())
        for i in range(max_class + 1):
            if i not in train_classes:
                missing_classes.append(i)
        if missing_classes:
            issues.append(f"⚠️ Missing class IDs: {missing_classes}")
    
    # Check for total issues
    total_coord_issues = len(results['train']['coordinate_issues']) + len(results['val']['coordinate_issues'])
    total_format_issues = len(results['train']['format_issues']) + len(results['val']['format_issues'])
    
    if total_coord_issues > 0:
        issues.append(f"⚠️ {total_coord_issues} coordinate issues")
    if total_format_issues > 0:
        issues.append(f"⚠️ {total_format_issues} format issues")
    
    if not issues:
        issues.append("✅ No major issues detected")
    
    for issue in issues:
        ax6.text(0.1, y_pos, issue, fontsize=11)
        y_pos -= 0.08
    
    plt.tight_layout()
    plt.savefig('label_analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    dataset_path = "data/processed/fod_yolo"
    results = analyze_yolo_labels(dataset_path)
    
    # Save detailed report
    with open('label_analysis_detailed.txt', 'w') as f:
        f.write("DETAILED LABEL ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        for split, data in results.items():
            f.write(f"{split.upper()} SET DETAILED ISSUES:\n")
            f.write("-" * 30 + "\n")
            
            if data['coordinate_issues']:
                f.write("COORDINATE ISSUES:\n")
                for issue in data['coordinate_issues']:
                    f.write(f"  {issue}\n")
                f.write("\n")
            
            if data['format_issues']:
                f.write("FORMAT ISSUES:\n")
                for issue in data['format_issues']:
                    f.write(f"  {issue}\n")
                f.write("\n")
    
    print("\nAnalysis complete! Check 'label_analysis_report.png' and 'label_analysis_detailed.txt' for results.")
