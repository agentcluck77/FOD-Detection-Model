#!/usr/bin/env python3
"""
Enhanced VOC to YOLO Converter with Better Dataset Splitting
Handles video-like sequences by clustering similar images together before splitting
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import yaml
import cv2
from collections import defaultdict, Counter
import random
import argparse
from tqdm import tqdm
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedVOCToYOLOConverter:
    """Convert Pascal VOC format to YOLO format with intelligent splitting"""
    
    def __init__(self, voc_path, yolo_path):
        self.voc_path = Path(voc_path)
        self.yolo_path = Path(yolo_path)
        self.class_to_id = {}
        self.class_names = []
        self.stats = defaultdict(int)
        
    def parse_voc_annotation(self, xml_file):
        """Parse Pascal VOC XML annotation file with support for float coordinates"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            filename_elem = root.find('filename')
            if filename_elem is None:
                return None, None, None, []
            
            filename = filename_elem.text
            size = root.find('size')
            if size is None:
                return None, None, None, []
            
            width_elem = size.find('width')
            height_elem = size.find('height')
            
            if width_elem is None or height_elem is None:
                return None, None, None, []
            
            # Handle both integer and float values
            width = int(float(width_elem.text))
            height = int(float(height_elem.text))
            
            # Get all objects
            objects = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                
                name = name_elem.text
                if not name:
                    continue
                
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                # Extract bounding box coordinates with float support
                try:
                    xmin_elem = bbox.find('xmin')
                    ymin_elem = bbox.find('ymin')
                    xmax_elem = bbox.find('xmax')
                    ymax_elem = bbox.find('ymax')
                    
                    if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                        continue
                    
                    xmin = int(float(xmin_elem.text))
                    ymin = int(float(ymin_elem.text))
                    xmax = int(float(xmax_elem.text))
                    ymax = int(float(ymax_elem.text))
                    
                    # Validate and clamp coordinates
                    xmin = max(0, min(xmin, width))
                    ymin = max(0, min(ymin, height))
                    xmax = max(xmin, min(xmax, width))
                    ymax = max(ymin, min(ymax, height))
                    
                    # Skip invalid bounding boxes
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    
                    # Check for difficult flag
                    difficult_elem = obj.find('difficult')
                    difficult = int(difficult_elem.text) if difficult_elem is not None and difficult_elem.text else 0
                    
                    objects.append({
                        'name': name,
                        'difficult': difficult,
                        'bbox': {
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax
                        }
                    })
                    
                except (ValueError, TypeError) as e:
                    continue
            
            return filename, width, height, objects
            
        except ET.ParseError as e:
            self.stats['xml_parse_errors'] += 1
            return None, None, None, []
        except Exception as e:
            self.stats['unexpected_errors'] += 1
            return None, None, None, []
    
    def extract_sequence_groups(self, all_files):
        """Group files that are likely from the same video sequence"""
        print("📊 Analyzing image sequences...")
        
        # Extract base names and sequence patterns
        sequence_groups = defaultdict(list)
        
        for xml_file, filename, width, height, objects in all_files:
            # Extract potential sequence identifier
            base_name = Path(filename).stem
            
            # Try to identify sequence patterns (e.g., video_001, video_002, etc.)
            # Look for numbered sequences
            numeric_suffix = ''
            alpha_prefix = ''
            
            # Split by common separators
            parts = base_name.replace('_', '-').replace('.', '-').split('-')
            
            # Reconstruct base and number
            if len(parts) > 1 and parts[-1].isdigit():
                alpha_prefix = '-'.join(parts[:-1])
                numeric_suffix = parts[-1]
            else:
                # If no clear pattern, use first part as group
                alpha_prefix = parts[0] if parts else base_name
            
            # Group by prefix (likely same object/scene)
            sequence_groups[alpha_prefix].append({
                'xml_file': xml_file,
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects,
                'sequence_id': f"{alpha_prefix}_{numeric_suffix}" if numeric_suffix else alpha_prefix
            })
        
        print(f"Found {len(sequence_groups)} potential sequence groups")
        
        # Sort within each group by filename to maintain sequence order
        for group_name in sequence_groups:
            sequence_groups[group_name].sort(key=lambda x: x['filename'])
        
        return sequence_groups
    
    def analyze_group_diversity(self, sequence_groups):
        """Analyze the diversity within sequence groups"""
        print("🔍 Analyzing group diversity...")
        
        group_stats = {}
        
        for group_name, files in sequence_groups.items():
            # Collect class information
            classes_in_group = set()
            total_objects = 0
            
            for file_info in files:
                for obj in file_info['objects']:
                    classes_in_group.add(obj['name'])
                    total_objects += 1
            
            group_stats[group_name] = {
                'num_files': len(files),
                'num_classes': len(classes_in_group),
                'classes': list(classes_in_group),
                'total_objects': total_objects,
                'avg_objects_per_image': total_objects / len(files) if files else 0
            }
        
        # Print statistics
        print("\nSequence group statistics:")
        print(f"{'Group':<20} {'Files':<8} {'Classes':<10} {'Avg Obj/Img':<12}")
        print("-" * 55)
        for group_name, stats in sorted(group_stats.items()):
            print(f"{group_name:<20} {stats['num_files']:<8} {stats['num_classes']:<10} {stats['avg_objects_per_image']:<12.1f}")
        
        return group_stats
    
    def smart_split_by_groups(self, sequence_groups, train_split=0.7, val_split=0.2, test_split=0.1):
        """Split dataset by distributing entire sequence groups"""
        print("\n📏 Performing intelligent sequence-aware splitting...")
        
        # Convert groups to list for easier manipulation
        group_list = list(sequence_groups.items())
        random.shuffle(group_list)  # Shuffle the groups
        
        # Calculate split sizes
        total_groups = len(group_list)
        n_train_groups = int(total_groups * train_split)
        n_val_groups = int(total_groups * val_split)
        
        # Assign groups to splits
        train_groups = group_list[:n_train_groups]
        val_groups = group_list[n_train_groups:n_train_groups + n_val_groups]
        test_groups = group_list[n_train_groups + n_val_groups:]
        
        # Count files in each split
        train_files = []
        val_files = []
        test_files = []
        
        for group_name, files in train_groups:
            train_files.extend(files)
        
        for group_name, files in val_groups:
            val_files.extend(files)
        
        for group_name, files in test_groups:
            test_files.extend(files)
        
        print(f"Split groups: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
        print(f"Split files: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        return train_files, val_files, test_files
    
    def stratified_split_by_class(self, sequence_groups, train_split=0.7, val_split=0.2, test_split=0.1):
        """Split dataset ensuring class distribution is maintained across splits"""
        print("\n🎯 Performing stratified split by class distribution...")
        
        # Group sequences by their primary class
        class_to_groups = defaultdict(list)
        
        for group_name, files in sequence_groups.items():
            # Find the most common class in this group
            class_counts = Counter()
            for file_info in files:
                for obj in file_info['objects']:
                    class_counts[obj['name']] += 1
            
            if class_counts:
                primary_class = class_counts.most_common(1)[0][0]
                class_to_groups[primary_class].append((group_name, files))
        
        # Split each class proportionally
        train_files = []
        val_files = []
        test_files = []
        
        for class_name, groups in class_to_groups.items():
            random.shuffle(groups)
            
            n_groups = len(groups)
            n_train = int(n_groups * train_split)
            n_val = int(n_groups * val_split)
            
            # Add files from each split
            for i, (group_name, files) in enumerate(groups):
                if i < n_train:
                    train_files.extend(files)
                elif i < n_train + n_val:
                    val_files.extend(files)
                else:
                    test_files.extend(files)
            
            print(f"Class '{class_name}': {n_groups} groups -> Train={n_train}, Val={n_val}, Test={n_groups-n_train-n_val}")
        
        return train_files, val_files, test_files
    
    def convert_dataset_with_smart_split(self, train_split=0.7, val_split=0.2, test_split=0.1, split_method='sequence'):
        """Convert Pascal VOC dataset to YOLO format with intelligent splitting"""
        
        # Verify splits sum to 1
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        # Analyze dataset
        all_files, annotations_dir, images_dir = self.analyze_dataset()
        
        if not all_files:
            raise ValueError("No valid annotation files found")
        
        # Group files by sequence
        sequence_groups = self.extract_sequence_groups(all_files)
        
        # Analyze group diversity
        group_stats = self.analyze_group_diversity(sequence_groups)
        
        # Choose splitting method
        if split_method == 'sequence':
            train_files, val_files, test_files = self.smart_split_by_groups(
                sequence_groups, train_split, val_split, test_split
            )
        elif split_method == 'stratified':
            train_files, val_files, test_files = self.stratified_split_by_class(
                sequence_groups, train_split, val_split, test_split
            )
        else:
            # Fallback to random split
            print("\n🎲 Using random split as fallback...")
            all_files_list = []
            for files in sequence_groups.values():
                all_files_list.extend(files)
            random.shuffle(all_files_list)
            
            n_total = len(all_files_list)
            n_train = int(n_total * train_split)
            n_val = int(n_total * val_split)
            
            train_files = all_files_list[:n_train]
            val_files = all_files_list[n_train:n_train + n_val]
            test_files = all_files_list[n_train + n_val:]
        
        # Create YOLO directory structure
        self.yolo_path.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.yolo_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Convert each split
        split_info = {}
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            print(f"\nConverting {split_name} split...")
            
            processed_files = 0
            total_objects = 0
            
            # Extract file info from grouped format
            file_list = files if isinstance(files[0], dict) else [(f['xml_file'], f['filename'], f['width'], f['height'], f['objects']) for f in files]
            
            for file_info in tqdm(file_list, desc=f"Converting {split_name}"):
                if isinstance(file_info, dict):
                    xml_file = file_info['xml_file']
                    filename = file_info['filename']
                    width = file_info['width']
                    height = file_info['height']
                    objects = file_info['objects']
                else:
                    xml_file, filename, width, height, objects = file_info
                
                # Find corresponding image file
                base_name = Path(filename).stem
                img_file = None
                
                # Try different image extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_img = images_dir / f"{base_name}{ext}"
                    if potential_img.exists():
                        img_file = potential_img
                        break
                
                if not img_file:
                    self.stats['conversion_missing_images'] += 1
                    continue
                
                # Verify image dimensions
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    
                    # Use actual image dimensions if different from XML
                    if img_width != width or img_height != height:
                        width, height = img_width, img_height
                
                except Exception as e:
                    continue
                
                # Copy image to YOLO directory
                target_img = self.yolo_path / split_name / 'images' / f"{base_name}.jpg"
                try:
                    shutil.copy2(img_file, target_img)
                except Exception as e:
                    continue
                
                # Convert annotations
                target_label = self.yolo_path / split_name / 'labels' / f"{base_name}.txt"
                
                try:
                    with open(target_label, 'w') as f:
                        for obj in objects:
                            if obj['difficult']:
                                continue  # Skip difficult objects
                            
                            class_id = self.class_to_id[obj['name']]
                            x_center, y_center, bbox_width, bbox_height = self.voc_to_yolo_bbox(
                                obj['bbox'], width, height
                            )
                            
                            # Write in YOLO format
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    
                    processed_files += 1
                    total_objects += len([obj for obj in objects if not obj['difficult']])
                
                except Exception as e:
                    continue
            
            split_info[split_name] = {
                'files': processed_files,
                'objects': total_objects
            }
            
            print(f"  {split_name}: {processed_files} files, {total_objects} objects")
        
        # Create data.yaml file
        data_yaml = {
            'path': str(self.yolo_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = self.yolo_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Save conversion report with split analysis
        report = {
            'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_dataset': str(self.voc_path),
            'target_dataset': str(self.yolo_path),
            'split_method': split_method,
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'splits': split_info,
            'sequence_groups': {
                'total_groups': len(sequence_groups),
                'group_stats': group_stats
            },
            'statistics': dict(self.stats)
        }
        
        report_path = self.yolo_path / 'conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nConversion complete!")
        print(f"YOLO dataset saved to: {self.yolo_path}")
        print(f"Configuration file: {yaml_path}")
        print(f"Conversion report: {report_path}")
        print(f"\nSplit method: {split_method}")
        print(f"Dataset summary:")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Train: {split_info['train']['files']} images")
        print(f"  Val: {split_info['val']['files']} images")
        print(f"  Test: {split_info['test']['files']} images")
        
        return yaml_path
    
    # Include all other methods from the original converter
    def find_dataset_directories(self):
        """Find the Annotations and Images directories"""
        annotations_dir = None
        images_dir = None
        
        # Common directory patterns
        patterns = [
            ("Annotations", "JPEGImages"),
            ("Annotations", "Images"),
            ("annotations", "images"),
            ("labels", "images")
        ]
        
        for ann_pattern, img_pattern in patterns:
            # Search in root directory
            ann_path = self.voc_path / ann_pattern
            img_path = self.voc_path / img_pattern
            
            if ann_path.exists() and img_path.exists():
                annotations_dir = ann_path
                images_dir = img_path
                break
            
            # Search recursively
            for ann_candidate in self.voc_path.rglob(ann_pattern):
                if ann_candidate.is_dir():
                    # Look for corresponding images directory
                    parent = ann_candidate.parent
                    img_candidate = parent / img_pattern
                    if img_candidate.exists() and img_candidate.is_dir():
                        annotations_dir = ann_candidate
                        images_dir = img_candidate
                        break
            
            if annotations_dir and images_dir:
                break
        
        return annotations_dir, images_dir
    
    def analyze_dataset(self):
        """Analyze the Pascal VOC dataset"""
        annotations_dir, images_dir = self.find_dataset_directories()
        
        if not annotations_dir or not images_dir:
            raise ValueError(f"Could not find Annotations or Images directories in {self.voc_path}")
        
        print(f"Found Annotations: {annotations_dir}")
        print(f"Found Images: {images_dir}")
        
        # Collect all data
        all_files = []
        class_counter = Counter()
        
        xml_files = list(annotations_dir.glob("*.xml"))
        print(f"Processing {len(xml_files)} annotation files...")
        
        for xml_file in tqdm(xml_files, desc="Analyzing annotations"):
            filename, width, height, objects = self.parse_voc_annotation(xml_file)
            
            if filename is None:
                continue
            
            # Skip files with no valid objects
            if not objects:
                self.stats['files_no_objects'] += 1
                continue
            
            # Verify image exists
            base_name = Path(filename).stem
            img_found = False
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = images_dir / f"{base_name}{ext}"
                if img_path.exists():
                    img_found = True
                    break
            
            if not img_found:
                self.stats['missing_images'] += 1
                continue
            
            all_files.append((xml_file, filename, width, height, objects))
            
            # Count classes
            for obj in objects:
                class_counter[obj['name']] += 1
        
        # Create class mapping
        self.class_names = sorted(class_counter.keys())
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"\nFound {len(all_files)} valid files with {len(self.class_names)} classes:")
        for i, (name, count) in enumerate(class_counter.most_common()):
            print(f"  {i}: {name} ({count} instances)")
        
        return all_files, annotations_dir, images_dir
    
    def voc_to_yolo_bbox(self, bbox, img_width, img_height):
        """Convert Pascal VOC bbox to YOLO format with validation"""
        # Calculate center and dimensions
        x_center = ((bbox['xmin'] + bbox['xmax']) / 2) / img_width
        y_center = ((bbox['ymin'] + bbox['ymax']) / 2) / img_height
        width = (bbox['xmax'] - bbox['xmin']) / img_width
        height = (bbox['ymax'] - bbox['ymin']) / img_height
        
        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return x_center, y_center, width, height


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert Pascal VOC to YOLO format with smart splitting')
    parser.add_argument('--voc_path', type=str, help='Path to VOC dataset')
    parser.add_argument('--yolo_path', type=str, help='Output path for YOLO dataset')
    parser.add_argument('--train_split', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--split_method', type=str, choices=['sequence', 'stratified', 'random'], 
                       default='sequence', help='Splitting method')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if not args.voc_path:
        # Try common locations
        possible_paths = [
            "data/raw/fod_a_data/VOC2007",
            "FOD-A/FOD-data/VOC2007",
            "data/raw/fod_a_data",
            "FOD-A/FOD-data"
        ]
        
        args.voc_path = None
        for path in possible_paths:
            if Path(path).exists():
                # Verify it has the right structure
                if (Path(path) / "Annotations").exists():
                    args.voc_path = path
                    break
        
        if not args.voc_path:
            print("No VOC dataset found. Please specify --voc_path")
            return False
    
    if not args.yolo_path:
        args.yolo_path = "fod_yolo"
    
    print(f"Input VOC dataset: {args.voc_path}")
    print(f"Output YOLO dataset: {args.yolo_path}")
    print(f"Split method: {args.split_method}")
    
    # Verify VOC path exists
    if not Path(args.voc_path).exists():
        print(f"Error: VOC dataset path does not exist: {args.voc_path}")
        return False
    
    # Create converter and run conversion
    try:
        converter = AdvancedVOCToYOLOConverter(args.voc_path, args.yolo_path)
        yaml_path = converter.convert_dataset_with_smart_split(
            args.train_split, args.val_split, args.test_split, args.split_method
        )
        
        print(f"\n✅ Conversion successful with {args.split_method} splitting!")
        print(f"Use this data.yaml file for training: {yaml_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import time  # Add this import
    success = main()
    exit(0 if success else 1)