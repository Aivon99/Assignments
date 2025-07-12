#!/usr/bin/env python3
"""
Main script for book detection using SIFT features.

This script provides a command-line interface for detecting books in scene images
using the BookDetector class. It supports both single scene processing and batch
processing of all scenes with accuracy evaluation.

Usage:
    python3 main.py --scene_path path/to/scene.jpg
    python3 main.py --all  # Process all scenes
"""

import cv2
import numpy as np
import os
import json
import argparse
from typing import Dict, List
from book_detector import BookDetector


def visualize_detections(scene_path: str, detections: Dict[str, List[Dict]], output_path: str) -> str:
    """
    Draw bounding boxes on the scene image to visualize detections.
    
    Args:
        scene_path: Path to the original scene image
        detections: Detection results from BookDetector
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Load scene image
    scene_img = cv2.imread(scene_path)
    if scene_img is None:
        return output_path
    
    # Define colors for different models (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Sky blue
    ]
    
    # Draw detections
    color_idx = 0
    for model_name, instances in detections.items():
        color = colors[color_idx % len(colors)]
        
        for instance in instances:
            # Draw bounding box
            pts = np.array([
                instance['top_left'],
                instance['top_right'],
                instance['bottom_right'],
                instance['bottom_left']
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(scene_img, [pts], True, color, 2)
            
            # Add label with model name
            label_pos = (instance['top_left'][0], instance['top_left'][1] - 5)
            cv2.putText(scene_img, model_name, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        color_idx += 1
    
    # Save visualization
    cv2.imwrite(output_path, scene_img)
    return output_path


def process_single_scene(detector: BookDetector, scene_path: str, output_dir: str) -> Dict:
    """
    Process a single scene image for book detection.
    
    Args:
        detector: BookDetector instance
        scene_path: Path to the scene image
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing detection results
    """
    scene_name = os.path.basename(scene_path).split('.')[0]
    
    # Detect books
    print(f"Processing {scene_name}...")
    detections = detector.detect_books(scene_path)
    
    # Count detected books
    detected_models = []
    for model_name, instances in detections.items():
        detected_models.extend([model_name] * len(instances))
    
    # Prepare results with JSON-safe types
    results = {
        'scene': scene_name,
        'detected_count': len(detected_models),
        'detected_models': detected_models,
        'detections': {}
    }
    
    # Convert numpy types to native Python types for JSON serialization
    for model_name, instances in detections.items():
        results['detections'][model_name] = []
        for inst in instances:
            # Remove 'mask' field if present (not needed in output)
            safe_instance = {
                'top_left': [int(inst['top_left'][0]), int(inst['top_left'][1])],
                'top_right': [int(inst['top_right'][0]), int(inst['top_right'][1])],
                'bottom_right': [int(inst['bottom_right'][0]), int(inst['bottom_right'][1])],
                'bottom_left': [int(inst['bottom_left'][0]), int(inst['bottom_left'][1])],
                'area': int(inst['area'])
            }
            results['detections'][model_name].append(safe_instance)
    
    # Save text results
    text_path = os.path.join(output_dir, f'{scene_name}_results.txt')
    with open(text_path, 'w') as f:
        f.write(f"Scene: {scene_name}\n")
        f.write(f"Detected: {len(detected_models)} books\n")
        f.write(f"Models: {detected_models}\n\n")
        f.write("Detection details:\n")
        for model_name, instances in detections.items():
            f.write(f"  {model_name}: {len(instances)} instance(s)\n")
            for i, inst in enumerate(instances):
                f.write(f"    Instance {i+1}: Area = {inst['area']} pixels\n")
    
    # Save visualization
    vis_path = os.path.join(output_dir, f'{scene_name}_detected.jpg')
    visualize_detections(scene_path, detections, vis_path)
    
    print(f"  Detected {len(detected_models)} book(s)")
    
    return results


def evaluate_accuracy(detected_models: List[str], expected_models: List[str]) -> bool:
    """
    Evaluate if detection matches ground truth.
    
    Args:
        detected_models: List of detected model names
        expected_models: List of expected model names from ground truth
        
    Returns:
        True if detection matches ground truth, False otherwise
    """
    return sorted(detected_models) == sorted(expected_models)


def process_all_scenes(detector: BookDetector, scene_dir: str, output_dir: str) -> None:
    """
    Process all scenes and evaluate overall accuracy.
    
    Args:
        detector: BookDetector instance
        scene_dir: Directory containing scene images
        output_dir: Directory to save results
    """
    # Load ground truth
    ground_truth_path = os.path.join(os.path.dirname(__file__), 'ground_truth.json')
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    ground_truth = ground_truth_data['ground_truth']
    
    # Process results
    all_results = {}
    correct = 0
    total = 0
    
    print("\nProcessing all scenes...")
    print("-" * 50)
    
    # Process scenes 0-28
    for scene_num in range(29):
        scene_name = f'scene_{scene_num}'
        scene_path = os.path.join(scene_dir, f'{scene_name}.jpg')
        
        if not os.path.exists(scene_path):
            continue
        
        # Process scene
        results = process_single_scene(detector, scene_path, output_dir)
        
        # Evaluate against ground truth
        detected = results['detected_models']
        expected = ground_truth.get(scene_name, [])
        match = evaluate_accuracy(detected, expected)
        
        if match:
            correct += 1
        total += 1
        
        # Add evaluation to results
        results['expected'] = expected
        results['match'] = match
        all_results[scene_name] = results
        
        # Print evaluation
        status = "✓" if match else "✗"
        print(f"  {status} Expected: {len(expected)}, Got: {len(detected)}")
    
    # Calculate and display final statistics
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("-" * 50)
    print(f"\nFinal Results:")
    print(f"  Correct scenes: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Save summary
    summary = {
        'total_scenes': total,
        'correct': correct,
        'accuracy': f"{accuracy:.1f}%",
        'results': all_results
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    """Main entry point for the book detection script."""
    parser = argparse.ArgumentParser(
        description='Book detection using SIFT features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single scene
  python3 main.py --scene_path ../dataset/dataset/scenes/scene_1.jpg
  
  # Process all scenes and evaluate accuracy
  python3 main.py --all
  
  # Use custom directories
  python3 main.py --all --model_dir /path/to/models --output_dir results
        """
    )
    
    # Command options
    parser.add_argument('--scene_path', type=str, 
                        help='Path to a single scene image to process')
    parser.add_argument('--all', action='store_true', 
                        help='Process all scenes and evaluate accuracy')
    
    # Directory options
    parser.add_argument('--model_dir', type=str, 
                        default='../dataset/dataset/models',
                        help='Directory containing model images (default: ../dataset/dataset/models)')
    parser.add_argument('--scene_dir', type=str, 
                        default='../dataset/dataset/scenes',
                        help='Directory containing scene images (default: ../dataset/dataset/scenes)')
    parser.add_argument('--output_dir', type=str, 
                        default='output',
                        help='Directory to save results (default: output)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.scene_path and not args.all:
        parser.error("Please specify either --scene_path or --all")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    print("Initializing book detector...")
    try:
        detector = BookDetector(args.model_dir)
        print(f"Loaded {len(detector.models)} model(s)")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return 1
    
    # Process based on mode
    try:
        if args.all:
            process_all_scenes(detector, args.scene_dir, args.output_dir)
        else:
            if not os.path.exists(args.scene_path):
                print(f"Error: Scene file not found: {args.scene_path}")
                return 1
            process_single_scene(detector, args.scene_path, args.output_dir)
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())