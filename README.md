# Book Detection System - Assignment Module #1

## Overview
This project implements a traditional computer vision system for detecting book spines in bookshelf images using SIFT features and homography estimation.

## Project Structure
```
.
├── book_detector_final.py          # Main implementation
├── book_detector_optimized.py      # Enhanced multi-scale version
├── book_detection_solution.ipynb   # Jupyter notebook with complete solution
├── dataset/                        # Input data
│   └── dataset/
│       ├── models/                 # Reference book images (22 books)
│       └── scenes/                 # Bookshelf scenes (29 images)
├── results/                        # Output from basic detector
└── results_optimized/              # Output from optimized detector
```

## Setup
```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install opencv-python numpy matplotlib notebook ipykernel
```

## Usage

### Run the detector on all scenes:
```bash
python book_detector_final.py
```

### Run the optimized multi-scale detector:
```bash
python book_detector_optimized.py
```

### Interactive notebook:
```bash
jupyter notebook book_detection_solution.ipynb
```

## Approach
1. **Feature Extraction**: SIFT features from model books and scenes
2. **Feature Matching**: FLANN-based matcher with Lowe's ratio test
3. **Homography Estimation**: RANSAC-based geometric verification
4. **Detection Validation**: Size, aspect ratio, and bounds checking

## Output Format
For each detected book:
```
Book N - X instance(s) found:
  Instance 1 {top_left: (x,y), top_right: (x,y), bottom_left: (x,y), bottom_right: (x,y), area: Npx}
```

## Results
- Generates detection visualizations with bounding boxes
- Saves text results for each scene
- Creates JSON report with metrics