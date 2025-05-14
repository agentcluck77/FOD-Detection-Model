#!/usr/bin/env python3
"""
FOD Dataset Preprocessing Script
Step 6 of Phase 1: Preprocess the converted YOLO dataset
"""

import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import shutil
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm

def analyze_yolo_dataset(dataset_path):
    """Analyze YOLO dataset and generate quality report"""
    dataset_path = Path(dataset_path)
    
    # Load data.yaml
    with open(dataset_path / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Analyzing dataset at {dataset_path}")
    print(f"Classes: {data_config['nc']}")
    print(f"Class names: {list(data_config['names'].values())[:5]}..." if data_config['nc'] > 5 else f"Class names: {list(data_config['names'].values())}")
    
    # Analyze each split
    stats = {}
    for split in ['train', 'val', 'test']:
        if split in data_config:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                label_files = list(labels_dir.glob('*.txt'))
                
                print(f"\n{split.upper()} split:")
                print(f"  Images: {len(image_files)}")
                print(f"  Labels: {len(label_files)}")
                
                # Check for missing labels or images
                missing_labels = []
                missing_images = []
                
                for img_file in image_files:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        missing_labels.append(img_file.name)
                
                for label_file in label_files:
                    img_file = images_dir / f"{label_file.stem}.jpg"
                    if not img_file.exists():
                        img_file = images_dir / f"{label_file.stem}.png"
                    if not img_file.exists():
                        missing_images.append(label_file.name)
                
                stats[split] = {
                    'images': len(image_files),
                    'labels': len(label_files),
                    'missing_labels': len(missing_labels),
                    'missing_images': len(missing_images)
                }
                
                if missing_labels:
                    print(f"  Missing labels: {len(missing_labels)}")
                if missing_images:
                    print(f"  Missing images: {len(missing_images)}")
    
    # Generate visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Split sizes
    splits = ['train', 'val', 'test']
    split_sizes = [stats.get(split, {}).get('images', 0) for split in splits]
    
    axes[0].bar(splits, split_sizes)
    axes[0].set_title('Dataset Split Sizes')
    axes[0].set_ylabel('Number of Images')
    
    # Missing data
    missing_labels = [stats.get(split, {}).get('missing_labels', 0) for split in splits]
    missing_images = [stats.get(split, {}).get('missing_images', 0) for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[1].bar(x - width/2, missing_labels, width, label='Missing Labels')
    axes[1].bar(x + width/2, missing_images, width, label='Missing Images')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Split')
    axes[1].set_title('Missing Files')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(dataset_path / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save preprocessing report
    report = {
        'dataset_path': str(dataset_path),
        'statistics': stats,
        'data_config': data_config,
        'status': 'analyzed'
    }
    
    with open(dataset_path / 'preprocessing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {dataset_path / 'preprocessing_report.json'}")
    print(f"Visualization saved to: {dataset_path / 'dataset_analysis.png'}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess YOLO dataset')
    parser.add_argument('--dataset', type=str, default='data/processed/fod_yolo',
                       help='Path to YOLO dataset')
    
    args = parser.parse_args()
    
    report = analyze_yolo_dataset(args.dataset)
    print("\nPreprocessing complete!")
