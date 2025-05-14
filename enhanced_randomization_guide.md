# Enhanced Dataset Randomization Guide

This guide explains how to use the enhanced dataset randomization features for FOD detection training.

## Overview

The enhanced randomization system improves the training and test set randomization in several ways:

1. **Multi-seed shuffling**: Uses multiple random seeds to shuffle data in different patterns
2. **Content-aware shuffling**: Uses hashes of image filenames to generate additional randomization seeds
3. **Class-balanced splitting**: Maintains class distribution across train/val/test splits
4. **Backup safety**: Creates backups of the original dataset before modification

These enhancements help avoid overfitting and make the model more robust by preventing it from recognizing patterns in the dataset organization.

## Usage Options

### Option 1: Standalone Randomization

Use the `enhance_dataset_randomization.py` script directly:

```bash
# Basic usage with default settings (extreme randomization)
python enhance_dataset_randomization.py --dataset data/processed/fod_yolo

# Customize randomization level
python enhance_dataset_randomization.py --dataset data/processed/fod_yolo --level [normal|high|extreme]

# Disable class balancing if needed
python enhance_dataset_randomization.py --dataset data/processed/fod_yolo --no-class-balance
```

### Option 2: Integrated with Quick Training

Use the updated `quick_train.py` script with randomization flags:

```bash
# Train with default randomization (high level)
python quick_train.py --quick --randomize

# Train with extreme randomization
python quick_train.py --quick --randomize --randomize-level extreme

# Train with customized epochs and model size
python quick_train.py --quick --randomize --randomize-level extreme --model m --epochs 20
```

### Option 3: All-in-One Script

Use the `randomize_and_train.py` script for a streamlined process:

```bash
# Default settings (extreme randomization + quick training)
python randomize_and_train.py

# Full training with extreme randomization
python randomize_and_train.py --full-training

# Customize all parameters
python randomize_and_train.py --level extreme --model m --epochs 50 --batch-size 16 --full-training
```

## Randomization Levels

- **normal**: Basic shuffling of the dataset
- **high**: Two-phase randomization with different methods
- **extreme**: Multiple-pass shuffling with varied seeds and techniques

## Tips for Best Results

1. **Use extreme randomization for final training**: The highest level of randomization helps create the most robust models
2. **Always maintain class balance**: Unless you have a specific reason not to, keeping class balance is important for model performance
3. **Lower batch size with randomization**: If GPU memory is limited, consider lowering batch size when using randomization
4. **Run multiple training cycles**: For best results, run training multiple times with different randomization seeds

## How It Works

The randomization process:

1. Collects all images and labels from the existing train/val/test splits
2. Identifies classes in each image for class-balanced splitting
3. Applies multiple shuffling passes with different random seeds
4. Splits the dataset while preserving class distribution
5. Creates a backup of the original dataset
6. Replaces the original dataset files with the newly randomized ones

## Troubleshooting

If you encounter issues:

- Check the backup directory created in case you need to restore
- Verify the dataset structure after randomization with `python preprocess_fod_dataset.py`
- If training performance decreases, try a different randomization level or restore the original dataset 