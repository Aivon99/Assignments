# Book Detection using SIFT Features

A computer vision system for detecting and locating books in scene images using Scale-Invariant Feature Transform (SIFT) features. This implementation uses traditional computer vision techniques without deep learning.

## 🎯 Performance

- **Accuracy**: 65.5% (19/29 scenes)
- **Strengths**: Works well for single books and well-separated multiple books
- **Limitations**: Struggles with tightly stacked identical books due to overlapping SIFT features

## 📋 Requirements

- Python 3.7 or higher
- OpenCV with SIFT support
- NumPy

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd final_result
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy
```

## 📁 Project Structure

```
final_result/
├── book_detector.py          # Core detection algorithm
├── main.py                   # Command-line interface
├── ground_truth.json         # Expected detection results
├── requirements.txt          # Python dependencies
├── book_detection_explained.ipynb  # Detailed explanation notebook
├── README.md                 # This file
└── output/                   # Detection results (generated)
    ├── scene_*_detected.jpg  # Visualizations with bounding boxes
    ├── scene_*_results.txt   # Text detection results
    └── summary.json          # Overall performance metrics
```

## 🔧 Usage

### Process a single scene

```bash
python3 main.py --scene_path ../dataset/dataset/scenes/scene_1.jpg
```

### Process all scenes and evaluate accuracy

```bash
python3 main.py --all
```

### Use custom directories

```bash
python3 main.py --all \
    --model_dir /path/to/models \
    --scene_dir /path/to/scenes \
    --output_dir /path/to/results
```

### Command-line options

- `--scene_path`: Path to a single scene image to process
- `--all`: Process all scenes (0-28) and evaluate accuracy
- `--model_dir`: Directory containing model images (default: `../dataset/dataset/models`)
- `--scene_dir`: Directory containing scene images (default: `../dataset/dataset/scenes`)
- `--output_dir`: Directory to save results (default: `output`)

## 📊 Output Files

The system generates several output files:

1. **Visualizations** (`*_detected.jpg`): Scene images with colored bounding boxes around detected books
2. **Text Results** (`*_results.txt`): Detailed detection information for each scene
3. **Summary** (`summary.json`): Overall performance metrics when using `--all`

## 🔍 Algorithm Overview

The detector uses a sophisticated SIFT-based approach:

1. **Feature Extraction**: Extracts SIFT keypoints and descriptors from all model book covers
2. **Feature Matching**: Uses FLANN-based matcher with Lowe's ratio test (0.7 threshold)
3. **Geometric Verification**: Employs RANSAC to find robust homographies
4. **Iterative Detection**: Removes matched features after each detection to find multiple instances
5. **Progressive Relaxation**: Gradually relaxes parameters for difficult detections

### Key Parameters

- `ratio_test`: 0.7 (Lowe's ratio threshold)
- `min_matches`: 10 (minimum feature matches)
- `ransac_threshold`: 5.0 (RANSAC reprojection error)
- `min_inliers`: 9 (minimum RANSAC inliers)
- `min_area`: 2000 pixels (minimum bounding box area)
- `max_area`: 50000 pixels (maximum bounding box area)

## 📚 Understanding the Results

- **Green checkmark (✓)**: Scene correctly detected (all books found)
- **Red X (✗)**: Scene incorrectly detected (missed or false detections)

Common failure cases:
- Scenes with multiple identical stacked books
- Heavily occluded books
- Books with very few distinctive features

## 🧪 Development and Testing

For detailed insights into the approach, parameter tuning process, and performance analysis, see the Jupyter notebook:

```bash
jupyter notebook book_detection_explained.ipynb
```

## ⚠️ Known Limitations

1. **Identical Stacked Books**: When identical books are stacked together, their SIFT features overlap significantly, making it impossible to distinguish individual instances
2. **Computational Cost**: Testing all 22 models against each scene is computationally expensive
3. **Parameter Sensitivity**: Performance varies with parameter choices; finding universal parameters for all scenarios is challenging

## 🤝 Contributing

To improve the detector within traditional computer vision constraints:
1. Experiment with different feature descriptors (ORB, SURF, AKAZE)
2. Try hybrid approaches combining SIFT with edge detection
3. Implement spatial reasoning for better duplicate detection
4. Optimize parameters for specific scene types

## 📄 License

This project is for educational purposes as part of a computer vision course assignment.