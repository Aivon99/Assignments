#!/usr/bin/env python3
"""
Simple script to run book detection on all scenes
"""

from book_detector_solution import UltimateBookDetector
import os

def main():
    # Initialize detector
    detector = UltimateBookDetector()
    
    # Load models
    models_dir = "../dataset/dataset/models"
    detector.load_models(models_dir)
    
    # Process all scenes
    scenes_dir = "../dataset/dataset/scenes"
    output_dir = "results"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.jpg')])
    
    print(f"Processing {len(scene_files)} scenes...\n")
    
    total_detections = 0
    
    for scene_file in scene_files:
        scene_path = os.path.join(scenes_dir, scene_file)
        scene_name = scene_file.replace('.jpg', '')
        
        # Detect books
        detections = detector.detect_all_books(scene_path)
        total_detections += len(detections)
        
        # Save visualization if detections found
        if detections:
            vis_path = os.path.join(output_dir, f"{scene_name}_detected.jpg")
            detector.visualize_detections(scene_path, detections, vis_path)
        
        # Save text results
        results_text = detector.format_results(detections) if detections else "No books detected"
        with open(os.path.join(output_dir, f"{scene_name}_results.txt"), 'w') as f:
            f.write(results_text)
        
        print(f"{scene_name}: {len(detections)} books detected")
    
    print(f"\nTotal: {total_detections} books detected")
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()