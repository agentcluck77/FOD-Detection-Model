#!/usr/bin/env python3
"""
Enhanced Dataset Randomization Script for FOD Detection
This script further randomizes the existing YOLO dataset without losing any data
by applying advanced shuffling and re-splitting techniques.
"""

import os
import sys
import yaml
import json
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from tqdm import tqdm
import itertools
import hashlib

# Set multiple random seeds for maximum randomness
RANDOM_SEEDS = [42, 101, 987, 12345, 999, 7777, 3141]
np.random.seed(sum(RANDOM_SEEDS))
random.seed(sum(RANDOM_SEEDS) % 2**32)

def enhance_randomization(dataset_path, train_split=0.7, val_split=0.2, test_split=0.1, 
                          randomization_level='extreme', preserve_class_balance=True):
    """
    Enhance randomization of the existing dataset
    
    Args:
        dataset_path: Path to the YOLO dataset
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test split
        randomization_level: How aggressive the randomization should be ('normal', 'high', 'extreme')
        preserve_class_balance: Whether to maintain class balance across splits
    """
    dataset_path = Path(dataset_path)
    
    print(f"🔀 Enhancing randomization of dataset at {dataset_path}")
    print(f"Randomization level: {randomization_level}")
    
    # Verify dataset structure
    if not (dataset_path / 'data.yaml').exists():
        print(f"Error: No data.yaml found at {dataset_path}")
        return False
    
    # Load data configuration
    with open(dataset_path / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get class information
    class_names = data_config.get('names', {})
    num_classes = data_config.get('nc', len(class_names))
    
    print(f"Dataset has {num_classes} classes")
    
    # Collect all images and labels from all splits
    all_images = []
    class_distribution = defaultdict(list)
    
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"Warning: {split} split directories not found")
            continue
        
        image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        # Process each image and its labels
        for img_file in image_files:
            label_file = label_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                print(f"Warning: Missing label for {img_file.name}")
                continue
            
            # Extract class information from label file
            try:
                classes_in_image = set()
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            classes_in_image.add(class_id)
                
                # Store image info with its classes
                img_info = {
                    'image_path': img_file,
                    'label_path': label_file,
                    'classes': classes_in_image,
                    'split': split
                }
                
                all_images.append(img_info)
                
                # Store by primary class (first class mentioned)
                if classes_in_image:
                    primary_class = min(classes_in_image)  # Use the smallest class ID as primary
                    class_distribution[primary_class].append(img_info)
                
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
    
    print(f"Total images collected: {len(all_images)}")
    
    # Apply hyper-randomization based on level
    if randomization_level == 'extreme':
        # Apply multiple shuffling passes with different seeds
        for seed in RANDOM_SEEDS:
            random.seed(seed)
            random.shuffle(all_images)
            
        # Generate unique seeds based on content hashes
        content_seeds = []
        for i, chunk in enumerate(np.array_split(all_images, 10)):
            chunk_hash = hashlib.md5(str([img['image_path'].name for img in chunk]).encode()).hexdigest()
            seed_val = int(chunk_hash, 16) % 2**32
            content_seeds.append(seed_val)
            
        # Apply content-based shuffling
        for seed in content_seeds:
            random.seed(seed)
            random.shuffle(all_images)
            
        # Permutation-based shuffling
        all_images = list(np.random.permutation(all_images))
        
    elif randomization_level == 'high':
        # Two-phase randomization
        random.shuffle(all_images)
        all_images = list(np.random.permutation(all_images))
    else:
        # Standard randomization
        random.shuffle(all_images)
    
    # Split dataset
    if preserve_class_balance:
        print("Maintaining class balance across splits...")
        train_images, val_images, test_images = [], [], []
        
        # Process each class separately to maintain balance
        for class_id, images in class_distribution.items():
            # Apply extreme randomization to each class
            for seed in RANDOM_SEEDS:
                random.seed(seed)
                random.shuffle(images)
            
            # Calculate split sizes
            n_total = len(images)
            n_train = int(n_total * train_split)
            n_val = int(n_total * val_split)
            
            # Split images by class
            train_images.extend(images[:n_train])
            val_images.extend(images[n_train:n_train + n_val])
            test_images.extend(images[n_train + n_val:])
            
        # Extra shuffling of each split
        for seed in RANDOM_SEEDS:
            random.seed(seed)
            random.shuffle(train_images)
            random.shuffle(val_images)
            random.shuffle(test_images)
    else:
        # Simple proportional split
        n_total = len(all_images)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
    
    print(f"New split distribution:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Create temporary directories for reorganized dataset
    temp_dir = dataset_path / f"temp_randomized_{random.randint(1000, 9999)}"
    
    for split in ['train', 'val', 'test']:
        (temp_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Helper function to copy files to new location
    def copy_files_to_split(images, split_name):
        for img_info in tqdm(images, desc=f"Copying {split_name}"):
            # Source paths
            img_path = img_info['image_path']
            label_path = img_info['label_path']
            
            # Target paths
            target_img = temp_dir / split_name / 'images' / img_path.name
            target_label = temp_dir / split_name / 'labels' / label_path.name
            
            # Copy files
            shutil.copy2(img_path, target_img)
            shutil.copy2(label_path, target_label)
    
    # Copy files to temporary location
    copy_files_to_split(train_images, 'train')
    copy_files_to_split(val_images, 'val')
    copy_files_to_split(test_images, 'test')
    
    # Create backup of original dataset
    backup_dir = dataset_path.parent / f"{dataset_path.name}_backup"
    if not backup_dir.exists():
        print(f"Creating backup at {backup_dir}")
        shutil.copytree(dataset_path, backup_dir)
    
    # Replace original dataset with new randomized version
    for split in ['train', 'val', 'test']:
        # Remove original content
        original_img_dir = dataset_path / split / 'images'
        original_label_dir = dataset_path / split / 'labels'
        
        # Clear directories but preserve the directories themselves
        if original_img_dir.exists():
            for f in original_img_dir.glob('*'):
                f.unlink()
        
        if original_label_dir.exists():
            for f in original_label_dir.glob('*'):
                f.unlink()
        
        # Copy new content
        temp_img_dir = temp_dir / split / 'images'
        temp_label_dir = temp_dir / split / 'labels'
        
        for f in temp_img_dir.glob('*'):
            shutil.copy2(f, original_img_dir / f.name)
        
        for f in temp_label_dir.glob('*'):
            shutil.copy2(f, original_label_dir / f.name)
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir)
    
    # Create report
    report = {
        'dataset_path': str(dataset_path),
        'randomization_level': randomization_level,
        'preserve_class_balance': preserve_class_balance,
        'splits': {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        },
        'backup_created_at': str(backup_dir)
    }
    
    # Save report
    with open(dataset_path / 'randomization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Dataset randomization completed successfully!")
    print(f"Report saved to: {dataset_path / 'randomization_report.json'}")
    print(f"Backup saved to: {backup_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Enhance randomization of YOLO dataset')
    parser.add_argument('--dataset', type=str, default='data/processed/fod_yolo',
                        help='Path to YOLO dataset')
    parser.add_argument('--level', type=str, choices=['normal', 'high', 'extreme'], 
                        default='extreme', help='Randomization level')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Training set proportion')
    parser.add_argument('--val', type=float, default=0.2,
                        help='Validation set proportion')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set proportion')
    parser.add_argument('--no-class-balance', action='store_false', dest='class_balance',
                        help='Disable class balancing across splits')
    
    args = parser.parse_args()
    
    # Verify splits sum to 1
    total_split = args.train + args.val + args.test
    if abs(total_split - 1.0) > 0.01:
        print(f"Error: Split proportions must sum to 1.0, got {total_split}")
        return 1
    
    enhance_randomization(
        args.dataset,
        train_split=args.train,
        val_split=args.val,
        test_split=args.test,
        randomization_level=args.level,
        preserve_class_balance=args.class_balance
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 