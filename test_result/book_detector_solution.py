#!/usr/bin/env python3
"""
Book Detection System - Ultimate Version
High-performance implementation with multiple instance detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import os
import time
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass 
class Detection:
    """Store detection results"""
    model_id: str
    bbox: Tuple[int, int, int, int]
    corners: List[Tuple[int, int]]
    confidence: float
    num_matches: int
    num_inliers: int
    area: int


class UltimateBookDetector:
    """High-performance book detector with multi-instance support"""
    
    def __init__(self):
        """Initialize with optimal parameters"""
        # SIFT with very low threshold for maximum features
        self.sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=4,  # More octaves
            contrastThreshold=0.01,  # Very low threshold
            edgeThreshold=10,
            sigma=1.6
        )
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # More checks for better matches
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Relaxed thresholds for better detection
        self.ratio_threshold = 0.8  # More permissive ratio test
        self.min_matches = 6  # Lower minimum
        self.min_inliers = 6
        self.ransac_threshold = 5.0
        self.min_confidence = 0.01  # Very low confidence threshold
        
        # Multi-scale detection
        self.scales = [0.8, 1.0, 1.2]
        
        self.models = {}
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """Preprocess with optional enhancement"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhance:
            # CLAHE enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        return gray
    
    def load_models(self, models_dir: str) -> None:
        """Load models with multi-scale features"""
        print("Loading models...")
        
        for filename in sorted(os.listdir(models_dir)):
            if not filename.endswith('.png'):
                continue
            
            filepath = os.path.join(models_dir, filename)
            model_id = filename.replace('.png', '')
            
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            # Store features at multiple scales
            all_features = {}
            
            for scale in self.scales:
                # Resize if needed
                if scale != 1.0:
                    h, w = img.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_img = cv2.resize(img, (new_w, new_h))
                else:
                    scaled_img = img
                
                # Extract features
                processed = self.preprocess_image(scaled_img)
                kp, desc = self.sift.detectAndCompute(processed, None)
                
                # Adjust keypoints to original scale
                if scale != 1.0 and kp is not None:
                    for keypoint in kp:
                        keypoint.pt = (keypoint.pt[0] / scale, keypoint.pt[1] / scale)
                        keypoint.size = keypoint.size / scale
                
                if desc is not None and len(kp) >= self.min_matches:
                    all_features[scale] = (kp, desc)
            
            if all_features:
                self.models[model_id] = {
                    'features': all_features,
                    'shape': img.shape[:2],
                    'image': img
                }
                total_kp = sum(len(f[0]) for f in all_features.values())
                print(f"  {model_id}: {total_kp} features across {len(all_features)} scales")
        
        print(f"Loaded {len(self.models)} models\n")
    
    def detect_single_model(self, model_id: str, model_data: dict,
                           scene_kp: List, scene_desc: np.ndarray,
                           used_indices: Set[int], scene_shape: Tuple) -> List[Detection]:
        """Detect all instances of a single model"""
        detections = []
        
        # Try each scale
        for scale, (model_kp, model_desc) in model_data['features'].items():
            # Filter out used keypoints
            valid_indices = [i for i in range(len(scene_kp)) if i not in used_indices]
            if len(valid_indices) < self.min_matches:
                continue
            
            valid_desc = scene_desc[valid_indices]
            valid_kp = [scene_kp[i] for i in valid_indices]
            
            # Match features
            try:
                matches = self.matcher.knnMatch(model_desc, valid_desc, k=2)
            except:
                continue
            
            # Ratio test
            good = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good.append(m)
            
            if len(good) < self.min_matches:
                continue
            
            # Find homography
            src_pts = np.float32([model_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([valid_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
            
            if M is None:
                continue
            
            # Check inliers
            inliers = mask.ravel().tolist()
            num_inliers = sum(inliers)
            
            if num_inliers < self.min_inliers:
                continue
            
            # Transform corners
            h, w = model_data['shape']
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(corners, M)
            dst_corners = np.int32(dst_corners).reshape(-1, 2)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(dst_corners)
            
            # Validate
            if self.validate_detection(x, y, w, h, dst_corners, scene_shape):
                # Mark used keypoints
                for i, m in enumerate(good):
                    if inliers[i]:
                        actual_idx = valid_indices[m.trainIdx]
                        used_indices.add(actual_idx)
                
                # Calculate confidence
                confidence = (num_inliers / len(good)) * (num_inliers / max(len(model_kp), 1))
                
                if confidence >= self.min_confidence:
                    detection = Detection(
                        model_id=model_id,
                        bbox=(x, y, w, h),
                        corners=dst_corners.tolist(),
                        confidence=confidence,
                        num_matches=len(good),
                        num_inliers=num_inliers,
                        area=w * h
                    )
                    detections.append(detection)
                    
                    # Only keep best detection per scale
                    break
        
        return detections
    
    def validate_detection(self, x: int, y: int, w: int, h: int,
                          corners: np.ndarray, scene_shape: Tuple) -> bool:
        """Validate detection is reasonable"""
        scene_h, scene_w = scene_shape[:2]
        
        # Allow some out of bounds
        margin = 20
        if x < -margin or y < -margin or x + w > scene_w + margin or y + h > scene_h + margin:
            return False
        
        # Size constraints
        if w < 15 or h < 30:  # Minimum size
            return False
        if w > scene_w * 0.4 or h > scene_h * 0.8:  # Maximum size
            return False
        
        # Aspect ratio
        aspect = h / w if w > 0 else 0
        if aspect < 0.5 or aspect > 20:
            return False
        
        # Check if corners form a reasonable quadrilateral
        # Calculate area using shoelace formula
        area_corners = 0.5 * abs(sum(corners[i][0] * corners[(i+1)%4][1] - 
                                    corners[(i+1)%4][0] * corners[i][1] 
                                    for i in range(4)))
        area_bbox = w * h
        
        # Area should be similar
        if area_corners < area_bbox * 0.5 or area_corners > area_bbox * 1.5:
            return False
        
        return True
    
    def detect_all_books(self, scene_path: str, max_per_model: int = 3) -> List[Detection]:
        """Detect all books in scene with multiple instances"""
        # Load scene
        scene_img = cv2.imread(scene_path)
        if scene_img is None:
            return []
        
        # Extract features
        scene_processed = self.preprocess_image(scene_img)
        scene_kp, scene_desc = self.sift.detectAndCompute(scene_processed, None)
        
        if scene_desc is None or len(scene_kp) < self.min_matches:
            return []
        
        all_detections = []
        
        # For each model
        for model_id, model_data in self.models.items():
            used_indices = set()
            
            # Try to find multiple instances
            for _ in range(max_per_model):
                detections = self.detect_single_model(
                    model_id, model_data, scene_kp, scene_desc,
                    used_indices, scene_img.shape
                )
                
                if not detections:
                    break
                
                all_detections.extend(detections)
                
                # Stop if we've used too many keypoints
                if len(used_indices) > len(scene_kp) * 0.7:
                    break
        
        # Remove duplicates/overlaps
        return self.filter_overlapping(all_detections)
    
    def filter_overlapping(self, detections: List[Detection], 
                          iou_threshold: float = 0.5) -> List[Detection]:
        """Remove overlapping detections, keeping best confidence"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for det in detections:
            # Check overlap with already selected
            keep = True
            for selected in filtered:
                iou = self.calculate_iou(det.bbox, selected.bbox)
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(self, scene_path: str, detections: List[Detection],
                           output_path: str) -> None:
        """Visualize detections"""
        img = cv2.imread(scene_path)
        if img is None:
            return
        
        # Color map
        colors = {
            'high': (0, 255, 0),     # Green
            'medium': (0, 255, 255), # Yellow  
            'low': (0, 165, 255)     # Orange
        }
        
        for det in detections:
            # Determine color
            if det.confidence > 0.1:
                color = colors['high']
            elif det.confidence > 0.05:
                color = colors['medium']
            else:
                color = colors['low']
            
            # Draw polygon
            corners = np.array(det.corners)
            cv2.polylines(img, [corners], True, color, 2)
            
            # Draw bbox
            x, y, w, h = det.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            
            # Label
            label = f"{det.model_id} ({det.confidence:.2f})"
            label_y = y - 10 if y > 30 else y + h + 20
            cv2.putText(img, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, img)
    
    def format_results(self, detections: List[Detection]) -> str:
        """Format output as required"""
        grouped = {}
        for det in detections:
            if det.model_id not in grouped:
                grouped[det.model_id] = []
            grouped[det.model_id].append(det)
        
        lines = []
        for model_id in sorted(grouped.keys()):
            instances = grouped[model_id]
            book_num = model_id.replace('model_', 'Book ')
            lines.append(f"{book_num} - {len(instances)} instance(s) found:")
            
            for i, det in enumerate(instances, 1):
                c = det.corners
                lines.append(
                    f"  Instance {i} {{top_left: {tuple(c[0])}, "
                    f"top_right: {tuple(c[1])}, "
                    f"bottom_left: {tuple(c[3])}, "
                    f"bottom_right: {tuple(c[2])}, "
                    f"area: {det.area}px}}"
                )
        
        return '\n'.join(lines)


def main():
    """Main execution"""
    # Clean up old results
    os.system("rm -rf results/")
    
    # Initialize
    detector = UltimateBookDetector()
    
    # Load models
    detector.load_models("dataset/dataset/models")
    
    # Process scenes
    scenes_dir = "dataset/dataset/scenes"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.jpg')])
    
    print(f"Processing {len(scene_files)} scenes...\n")
    
    all_results = []
    total_detections = 0
    total_time = 0
    
    for scene_file in scene_files:
        scene_path = os.path.join(scenes_dir, scene_file)
        scene_name = Path(scene_file).stem
        
        start = time.time()
        
        # Detect
        detections = detector.detect_all_books(scene_path)
        
        # Save visualization
        if detections:
            vis_path = os.path.join(output_dir, f"{scene_name}_detected.jpg")
            detector.visualize_detections(scene_path, detections, vis_path)
        
        # Save text results
        results_text = detector.format_results(detections) if detections else "No books detected"
        with open(os.path.join(output_dir, f"{scene_name}_results.txt"), 'w') as f:
            f.write(results_text)
        
        elapsed = time.time() - start
        total_time += elapsed
        total_detections += len(detections)
        
        print(f"{scene_name}: {len(detections)} detections in {elapsed:.2f}s")
        
        # Store results
        all_results.append({
            'scene': scene_name,
            'detections': len(detections),
            'time': elapsed
        })
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total detections: {total_detections}")
    print(f"Average per scene: {total_detections/len(scene_files):.1f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per scene: {total_time/len(scene_files):.2f}s")
    
    # Save summary
    summary = {
        'total_scenes': len(scene_files),
        'total_detections': total_detections,
        'average_per_scene': total_detections/len(scene_files),
        'total_time': total_time,
        'results': all_results
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()