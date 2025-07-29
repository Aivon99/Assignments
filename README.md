# Book Detection System - Computer Vision Assignment

## Overview
This project implements a book detection system using traditional computer vision techniques (SIFT features) to detect and locate books in bookshelf scenes. The system achieves **65.5% accuracy** (19/29 scenes) without using deep learning.

## Project Structure
```
.
├── final_result/                    # Main implementation
│   ├── book_detector.py            # Core SIFT-based detection algorithm
│   ├── main.py                     # Command-line interface
│   ├── requirements.txt            # Python dependencies
│   ├── ground_truth.json           # Expected detection results
│   ├── book_detection_explained.ipynb  # Detailed explanation notebook
│   ├── README.md                   # Detailed documentation
│   └── output/                     # Detection results
│       ├── scene_*_detected.jpg    # Visualizations with bounding boxes
│       ├── scene_*_results.txt     # Text detection results
│       └── summary.json            # Performance metrics
├── dataset/                        # Input data
│   └── dataset/
│       ├── models/                 # Reference book images (22 books)
│       └── scenes/                 # Bookshelf scenes (29 images)
└── README.md                       # This file
```

## Quick Start

1. **Install dependencies:**
```bash
cd final_result
pip install -r requirements.txt
```

2. **Run detection on all scenes:**
```bash
python3 main.py --all
```

3. **Process a single scene:**
```bash
python3 main.py --scene_path ../dataset/dataset/scenes/scene_1.jpg
```

## Results

- **Accuracy**: 65.5% (19/29 scenes correctly detected)
- **Strengths**: Works well for single books and well-separated multiple books
- **Limitations**: Cannot distinguish between identical stacked books due to overlapping SIFT features

All detection results with visualizations are available in `final_result/output/`.

## Algorithm

The detector uses:
1. **SIFT** (Scale-Invariant Feature Transform) for robust feature extraction
2. **FLANN** matcher with Lowe's ratio test for feature matching
3. **RANSAC** for robust homography estimation
4. **Iterative detection** with progressive parameter relaxation
5. **Geometric validation** to ensure detected books have realistic shapes

## Understanding the Results

For a detailed explanation of:
- Why SIFT was chosen
- How the algorithm works
- Parameter tuning process
- Performance analysis
- Fundamental limitations with identical stacked books

See the Jupyter notebook: `final_result/book_detection_explained.ipynb`

## Key Findings

The main limitation is that SIFT features cannot distinguish between identical stacked books because:
- Features from identical books are indistinguishable
- Matches spread across all instances rather than individual books
- RANSAC finds a single homography that spans multiple books

This is a fundamental limitation of feature-based matching for this specific scenario, explaining why accuracy is limited to 65.5%.

## Authors

Computer Vision Course Assignment - 2024