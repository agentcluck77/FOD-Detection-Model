# Phase 1 - FOD-A and YOLO
### Set up Environment
- use python 3.9 for compatibility, make sure it is CPython and not PyPy
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
# Create a new environment with CPython (not PyPy)
conda create -n fod_detection python=3.9

# Activate the new environment
conda activate fod_detection

# Check if it's CPython
python -c "import sys; print('Implementation:', sys.implementation.name)"
python -c "import sys; print('Is PyPy:', hasattr(sys, 'pypy_version_info'))"
python --version

# You should see:
# Implementation: cpython
# Is PyPy: False
# Python 3.9.x

# Install PyTorch with conda (recommended for conda environments)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install YOLOv8
pip install ultralytics
```

Verification script
```
# Create verification script
cat > verify_setup.py << 'EOF'
import sys
print(f"Python version: {sys.version}")
print(f"Is PyPy: {hasattr(sys, 'pypy_version_info')}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from ultralytics import YOLO
print("YOLOv8 imported successfully!")

# Quick test
model = YOLO('yolov8n.pt')
print("YOLOv8 model loaded successfully!")
EOF

python verify_setup.py
```

### Download dataset
and place VOC2007 in raw/fod_a_data

### Run fod_dataset_analysis.ipynb

### Run voc_converter_complete.py
convert from VOC to YOLO format
### Run create_preprocessing_script.py

### Run preprocess_fod_dataset.py
`python preprocess_fod_dataset.py --dataset data/processed/fod_yolo
### Run quick_train.py
`python quick_train.py --quick --epochs 10
