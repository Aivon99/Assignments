"""
Book Detection System using SIFT Features

This module implements a book detection system that can identify and locate
multiple instances of books in scene images using SIFT feature matching.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional


class BookDetector:
    """
    SIFT-based book detector for multiple instance detection.
    
    This detector uses Scale-Invariant Feature Transform (SIFT) to match
    book covers against scene images. It supports detecting multiple instances
    of the same book through iterative matching with progressive parameter relaxation.
    
    Attributes:
        model_dir: Directory containing model book images
        models: Dictionary of loaded model data with SIFT features
        sift: SIFT feature detector instance
        params: Detection parameters
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the book detector.
        
        Args:
            model_dir: Path to directory containing model images (model_0.png to model_21.png)
        """
        self.model_dir = model_dir
        self.models = {}
        self.sift = cv2.SIFT_create()
        
        # Detection parameters (tuned for optimal performance)
        self.params = {
            'ratio_test': 0.7,        # Lowe's ratio test threshold
            'min_matches': 10,        # Minimum number of feature matches
            'ransac_threshold': 5.0,  # RANSAC reprojection error threshold
            'min_inliers': 9,         # Minimum RANSAC inliers
            'min_area': 2000,         # Minimum valid bounding box area
            'max_area': 50000         # Maximum valid bounding box area
        }
        
        # FLANN matcher parameters
        self.flann_params = {
            'index': {'algorithm': 1, 'trees': 5},  # FLANN_INDEX_KDTREE
            'search': {'checks': 50}
        }
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all model images and precompute their SIFT features."""
        for i in range(22):
            model_name = f'model_{i}'
            model_path = os.path.join(self.model_dir, f'{model_name}.png')
            
            if os.path.exists(model_path):
                img = cv2.imread(model_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = self.sift.detectAndCompute(img, None)
                    self.models[model_name] = {
                        'keypoints': kp,
                        'descriptors': des,
                        'shape': img.shape,
                        'image': img
                    }
                    
    def detect_books(self, scene_path: str) -> Dict[str, List[Dict]]:
        """
        Detect all books in a scene image.
        
        Args:
            scene_path: Path to the scene image
            
        Returns:
            Dictionary mapping model names to list of detected instances.
            Each instance contains:
                - top_left: (x, y) coordinates
                - top_right: (x, y) coordinates
                - bottom_right: (x, y) coordinates
                - bottom_left: (x, y) coordinates
                - area: Bounding box area in pixels
        """
        # Load and process scene image
        scene_img = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
        if scene_img is None:
            return {}
            
        scene_kp, scene_des = self.sift.detectAndCompute(scene_img, None)
        if scene_des is None:
            return {}
        
        # Initialize FLANN matcher
        flann = cv2.FlannBasedMatcher(
            self.flann_params['index'], 
            self.flann_params['search']
        )
        
        # Detect books for each model
        detections = {}
        for model_name, model_data in self.models.items():
            if model_data['descriptors'] is None:
                continue
                
            instances = self._detect_model_instances(
                model_data, model_name, scene_kp, scene_des, flann
            )
            
            if instances:
                detections[model_name] = instances
        
        return detections
    
    def _detect_model_instances(
        self, 
        model_data: Dict,
        model_name: str,
        scene_kp: List,
        scene_des: np.ndarray,
        flann: cv2.FlannBasedMatcher
    ) -> List[Dict]:
        """
        Detect multiple instances of a specific model in the scene.
        
        Uses iterative matching: after detecting each instance, its matches
        are removed to allow detection of additional instances.
        """
        instances = []
        max_instances = 5  # Maximum instances to detect per model
        
        # Initial feature matching
        matches = flann.knnMatch(model_data['descriptors'], scene_des, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.params['ratio_test'] * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.params['min_matches']:
            return instances
        
        # Iteratively detect instances
        remaining_matches = good_matches.copy()
        
        for attempt in range(8):  # Maximum attempts with progressive relaxation
            if len(remaining_matches) < 4:  # Minimum for homography
                break
                
            # Progressive parameter relaxation for each attempt
            relaxation_params = self._get_relaxed_params(attempt)
            
            # Find instance with current parameters
            instance = self._find_single_instance(
                model_data, scene_kp, remaining_matches, relaxation_params
            )
            
            if instance is None:
                break
                
            # Check for duplicate detection
            if not self._is_duplicate(instance, instances):
                instances.append(instance)
                
                if len(instances) >= max_instances:
                    break
            
            # Remove matches used in this detection
            remaining_matches = self._remove_used_matches(
                remaining_matches, instance['mask']
            )
        
        return instances
    
    def _get_relaxed_params(self, attempt: int) -> Dict:
        """Get progressively relaxed parameters for each detection attempt."""
        if attempt == 0:
            return {
                'min_matches': 10,
                'min_inliers': 9,
                'min_area': 2000
            }
        elif attempt == 1:
            return {
                'min_matches': 6,
                'min_inliers': 5,
                'min_area': 3000
            }
        elif attempt == 2:
            return {
                'min_matches': 4,
                'min_inliers': 4,
                'min_area': 2000
            }
        else:
            return {
                'min_matches': 3,
                'min_inliers': 3,
                'min_area': 1000
            }
    
    def _find_single_instance(
        self,
        model_data: Dict,
        scene_kp: List,
        matches: List,
        relaxation_params: Dict
    ) -> Optional[Dict]:
        """Find a single instance using homography estimation."""
        if len(matches) < relaxation_params['min_matches']:
            return None
            
        # Extract matched points
        src_pts = np.float32([
            model_data['keypoints'][m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        dst_pts = np.float32([
            scene_kp[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        M, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.params['ransac_threshold']
        )
        
        if M is None or mask is None:
            return None
        
        # Check if we have enough inliers
        inliers = mask.ravel().sum()
        if inliers < relaxation_params['min_inliers']:
            return None
        
        # Transform model corners to scene coordinates
        h, w = model_data['shape']
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        try:
            transformed_corners = cv2.perspectiveTransform(corners, M)
            corners_2d = transformed_corners.reshape(4, 2)
        except:
            return None
        
        # Validate detection geometry
        if not self._validate_detection(corners_2d, relaxation_params['min_area']):
            return None
        
        return {
            'top_left': tuple(corners_2d[0].astype(int)),
            'top_right': tuple(corners_2d[1].astype(int)),
            'bottom_right': tuple(corners_2d[2].astype(int)),
            'bottom_left': tuple(corners_2d[3].astype(int)),
            'area': int(cv2.contourArea(corners_2d)),
            'mask': mask
        }
    
    def _validate_detection(self, corners: np.ndarray, min_area: int) -> bool:
        """Validate detection based on geometric constraints."""
        # Check area
        area = cv2.contourArea(corners)
        if area < min_area or area > self.params['max_area']:
            return False
        
        # Check aspect ratio and shape distortion
        edges = [
            np.linalg.norm(corners[1] - corners[0]),  # Top edge
            np.linalg.norm(corners[2] - corners[3]),  # Bottom edge
            np.linalg.norm(corners[3] - corners[0]),  # Left edge
            np.linalg.norm(corners[2] - corners[1])   # Right edge
        ]
        
        # Check if opposite edges have similar lengths (not too distorted)
        width_ratio = max(edges[0], edges[1]) / (min(edges[0], edges[1]) + 1e-6)
        height_ratio = max(edges[2], edges[3]) / (min(edges[2], edges[3]) + 1e-6)
        
        max_distortion = 3.0
        return width_ratio < max_distortion and height_ratio < max_distortion
    
    def _is_duplicate(self, new_instance: Dict, existing_instances: List[Dict]) -> bool:
        """Check if a detection overlaps with existing instances."""
        new_center = self._get_center(new_instance)
        duplicate_threshold = 50  # Pixel distance threshold
        
        for instance in existing_instances:
            existing_center = self._get_center(instance)
            distance = np.linalg.norm(new_center - existing_center)
            
            if distance < duplicate_threshold:
                return True
                
        return False
    
    def _get_center(self, instance: Dict) -> np.ndarray:
        """Calculate center point of a detection."""
        corners = np.array([
            instance['top_left'],
            instance['top_right'],
            instance['bottom_right'],
            instance['bottom_left']
        ])
        return np.mean(corners, axis=0)
    
    def _remove_used_matches(self, matches: List, mask: np.ndarray) -> List:
        """Remove matches that were used in a successful detection."""
        mask_list = mask.ravel().tolist()
        return [m for i, m in enumerate(matches) if mask_list[i] == 0]