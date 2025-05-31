import cv2
import numpy as np
from typing import List, Tuple

def detect_book_rectangles_hough(shelf_image_path: str, reference_book_path: str, 
                                output_path: str = "hough_detected.jpg") -> List[Tuple]:
    """
    Detect rectangular regions matching reference book dimensions using Hough transform
    
    Args:
        shelf_image_path: Path to the shelf image
        reference_book_path: Path to the reference book image
        output_path: Path to save the result image with bounding boxes
    
    Returns:
        List of detected rectangles as (x, y, width, height)
    """
    
    # Load images
    shelf_img = cv2.imread(shelf_image_path)
    reference_img = cv2.imread(reference_book_path)
    
    if shelf_img is None or reference_img is None:
        raise ValueError("Could not load one or both images")
    
    # Get reference book dimensions
    ref_height, ref_width = reference_img.shape[:2]
    ref_aspect_ratio = ref_height / ref_width
    
    print(f"Reference book dimensions: {ref_width}x{ref_height}, aspect ratio: {ref_aspect_ratio:.2f}")
    
    # Convert to grayscale for processing
    shelf_gray = cv2.cvtColor(shelf_img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing for better edge detection
    blurred = cv2.GaussianBlur(shelf_gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    
    # Morphological operations to connect broken edges
    kernel = np.ones((2,2), np.uint8)
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Hough Line Detection
    lines = cv2.HoughLinesP(
        edges_cleaned,
        rho=1,                    # Distance resolution in pixels
        theta=np.pi/180,          # Angle resolution in radians
        threshold=30,             # Minimum votes for a line
        minLineLength=ref_height//3,  # Minimum line length
        maxLineGap=10             # Maximum gap between line segments
    )
    
    if lines is None:
        print("No lines detected")
        return []
    
    print(f"Detected {len(lines)} lines")
    
    # Separate vertical and horizontal lines
    vertical_lines = []
    horizontal_lines = []
    output = shelf_img.copy()


    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # lines[i] is [[x1, y1, x2, y2]]
            cv2.line(output, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    cv2.imwrite('output_with_lines.jpg', output)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 == 0:  # Perfectly vertical
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
        
        # Classify lines based on angle (books are typically vertical)
        if angle > 70:  # Nearly vertical
            vertical_lines.append(line[0])
        elif angle < 20:  # Nearly horizontal
            horizontal_lines.append(line[0])
    


    print(f"Vertical lines: {len(vertical_lines)}, Horizontal lines: {len(horizontal_lines)}")
    
    # Find rectangular regions from line intersections
    candidate_rectangles = find_rectangles_from_lines(
        vertical_lines, horizontal_lines, ref_width, ref_height, ref_aspect_ratio
    )
    
    print(f"Found {len(candidate_rectangles)} candidate rectangles")
    
    # Draw results on image
    result_img = shelf_img.copy()
    
    for i, (x, y, w, h) in enumerate(candidate_rectangles):
        # Draw bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = f"Candidate {i+1}"
        cv2.putText(result_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, result_img)
    print(f"Result saved to {output_path}")
    
    return candidate_rectangles

def find_rectangles_from_lines(vertical_lines: List, horizontal_lines: List, 
                              ref_width: int, ref_height: int, ref_aspect_ratio: float) -> List[Tuple]:
    """
    Find rectangular regions that match book proportions from detected lines
    """
    rectangles = []
    tolerance = 0.4  # Aspect ratio tolerance
    min_area = (ref_width * ref_height) * 0.2  # Minimum 20% of reference size
    max_area = (ref_width * ref_height) * 3.0  # Maximum 300% of reference size
    
    # Group nearby vertical lines
    vertical_groups = group_parallel_lines(vertical_lines, is_vertical=True, distance_threshold=15)
    
    # For each pair of vertical line groups, try to form rectangles
    for i in range(len(vertical_groups)):
        for j in range(i + 1, len(vertical_groups)):
            left_lines = vertical_groups[i]
            right_lines = vertical_groups[j]
            
            # Get average x coordinates for left and right edges

            left_x = int(np.mean([line[0] for line in left_lines] + [line[2] for line in left_lines]))

            right_x = int(np.mean([line[0] for line in right_lines] + [line[2] for line in right_lines]))

            if left_x > right_x:
                left_x, right_x = right_x, left_x
            
            width = right_x - left_x
            
            # Check if width is reasonable for a book
            if width < ref_width * 0.3 or width > ref_width * 2.5:
                continue
            
            # Find vertical extent from the lines
            all_y_coords = []
            for line_group in [left_lines, right_lines]:
                for line in line_group:
                    all_y_coords.extend([line[1], line[3]])
            
            if not all_y_coords:
                continue
                
            top_y = min(all_y_coords)
            bottom_y = max(all_y_coords)
            height = bottom_y - top_y
            
            # Check aspect ratio
            if height <= 0:
                continue
                
            aspect_ratio = height / width
            
            # Filter by aspect ratio and area
            if abs(aspect_ratio - ref_aspect_ratio) / ref_aspect_ratio <= tolerance:
                area = width * height
                if min_area <= area <= max_area:
                    rectangles.append((left_x, top_y, width, height))
    
    # Remove overlapping rectangles
    #rectangles = remove_overlapping_rectangles(rectangles)
    
    return rectangles

def group_parallel_lines(lines: List, is_vertical: bool = True, distance_threshold: int = 15) -> List[List]:
    """
    Group parallel lines that are close to each other
    """
    if not lines:
        return []
    
    groups = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
            
        group = [line1]
        used[i] = True
        
        for j, line2 in enumerate(lines):
            if used[j] or i == j:
                continue
            
            # Calculate distance between parallel lines
            if is_vertical:
                dist = abs(line1[0] - line2[0])  # Distance between x coordinates
            else:
                dist = abs(line1[1] - line2[1])  # Distance between y coordinates
            
            if dist <= distance_threshold:
                group.append(line2)
                used[j] = True
        
        groups.append(group)
    
    return groups

def remove_overlapping_rectangles(rectangles: List[Tuple], overlap_threshold: float = 0.5) -> List[Tuple]:
    """
    Remove overlapping rectangles, keeping the larger ones
    """
    if not rectangles:
        return []
    
    # Sort by area (largest first)
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)
    
    filtered = []
    
    for rect in rectangles:
        x1, y1, w1, h1 = rect
        keep = True
        
        for kept_rect in filtered:
            x2, y2, w2, h2 = kept_rect
            
            # Calculate intersection area
            intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                              max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            
            # Calculate union area
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - intersection_area
            
            # Check overlap ratio
            if union_area > 0:
                overlap_ratio = intersection_area / union_area
                if overlap_ratio > overlap_threshold:
                    keep = False
                    break
        
        if keep:
            filtered.append(rect)
    
    return filtered

# Example usage
if __name__ == "__main__":
    
    shelf_path = "dataset/dataset/scenes/scene_7.jpg"
    reference_path = "dataset/dataset/models/model_20.png"
    try:
        rectangles = detect_book_rectangles_hough(shelf_path, reference_path)
        
        print(f"\nHough Detection Results:")
        print(f"Found {len(rectangles)} candidate rectangles")
        
        for i, (x, y, w, h) in enumerate(rectangles):
            print(f"Rectangle {i+1}: x={x}, y={y}, width={w}, height={h}, area={w*h}px")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the image paths!")
        