"""
Hough Line Transform based masking for industrial tote detection.
Detects vertical and horizontal rails and creates feature-level masks.
"""

import cv2
import numpy as np
import math


def compute_hough_feature_mask(image, grid_size):
    """
    Compute a feature-level mask based on Hough line detection of tote rails.
    
    Args:
        image: Input image (RGB numpy array)
        grid_size: Tuple (height, width) indicating number of patches
        
    Returns:
        mask: Boolean array of shape (grid_size[0] * grid_size[1],) indicating which patches are inside rails
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    
    # Edge detection with automatic Canny threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    
    # Define search regions
    roi_w_left = int(W * 0.31)
    roi_w_right = int(W * 0.28) 
    roi_h_top = int(H * 0.08)
    roi_h_bottom = int(H * 0.13)
    
    def find_vertical_lines(roi_edges, is_left, offset_x):
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, 
                                minLineLength=H // 8, maxLineGap=60)
        if lines is None:
            return None
        
        best_line = None
        best_x_score = -1 if is_left else 99999
     
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            if 87 < angle < 93:  # Vertical line
                avg_x = (x1 + x2) / 2 + offset_x
                if (is_left and avg_x > best_x_score) or (not is_left and avg_x < best_x_score):
                    best_x_score = avg_x
                    best_line = (x1, y1, x2, y2)
        
        return best_line
    
    def find_horizontal_lines(roi_edges, is_top, offset_y):
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, 
                                minLineLength=W // 6, maxLineGap=60)
        if lines is None:
            return None
        
        best_line = None
        best_y_score = -1 if is_top else 99999
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            if angle < 3 or angle > 177:  # Horizontal line
                avg_y = (y1 + y2) / 2 + offset_y
                if (is_top and avg_y > best_y_score) or (not is_top and avg_y < best_y_score):
                    best_y_score = avg_y
                    best_line = (x1, y1, x2, y2)
        
        return best_line
    
    # Find lines
    l_line = find_vertical_lines(edges[:, 0:roi_w_left], True, 0)
    r_line = find_vertical_lines(edges[:, W-roi_w_right:W], False, W-roi_w_right)
    t_line = find_horizontal_lines(edges[0:roi_h_top, :], True, 0)
    b_line = find_horizontal_lines(edges[H-roi_h_bottom:H, :], False, H-roi_h_bottom)
    
    def extrapolate_line(line, offset_x=0):
        if line is None: 
            return None
        x1, y1, x2, y2 = line
        x1, x2 = x1 + offset_x, x2 + offset_x
        if x2 == x1: 
            return (x1, 0), (x1, H)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (int((0 - intercept) / slope), 0), (int((H - intercept) / slope), H)
    
    def extrapolate_horizontal_line(line, offset_y=0):
        if line is None: 
            return None
        x1, y1, x2, y2 = line
        y1, y2 = y1 + offset_y, y2 + offset_y
        if y2 == y1: 
            return (0, y1), (W, y1)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (0, int(intercept)), (W, int(slope * W + intercept))
    
    # Extrapolate lines or use fallback positions
    l_pts = extrapolate_line(l_line) or ((int(W*0.12), 0), (int(W*0.12), H))
    r_pts = extrapolate_line(r_line, W-roi_w_right) or ((int(W*0.88), 0), (int(W*0.88), H))
    t_pts = extrapolate_horizontal_line(t_line) or ((0, int(H*0.02)), (W, int(H*0.02)))
    b_pts = extrapolate_horizontal_line(b_line, H-roi_h_bottom) or ((0, int(H*0.92)), (W, int(H*0.92)))
    
    # Create pixel-level mask
    pixel_mask = np.zeros((H, W), dtype=bool)
    
    def get_x_at_y(pts, y):
        (x1, y1), (x2, y2) = pts
        if x1 == x2: 
            return x1
        slope = (x2 - x1) / (y2 - y1)
        return int(x1 + slope * (y - y1))
    
    def get_y_at_x(pts, x):
        (x1, y1), (x2, y2) = pts
        if y1 == y2: 
            return y1
        slope = (y2 - y1) / (x2 - x1)
        return int(y1 + slope * (x - x1))
    
    # Mark pixels inside rails as True
    for y in range(H):
        left_x = get_x_at_y(l_pts, y)
        right_x = get_x_at_y(r_pts, y)
        pixel_mask[y, left_x:right_x] = True
    
    # Apply top/bottom constraints
    for x in range(W):
        top_y = get_y_at_x(t_pts, x)
        bottom_y = get_y_at_x(b_pts, x)
        pixel_mask[:top_y, x] = False
        pixel_mask[bottom_y:, x] = False
    
    # Convert pixel mask to patch mask
    patch_h = H // grid_size[0]
    patch_w = W // grid_size[1]
    
    patch_mask = np.zeros(grid_size[0] * grid_size[1], dtype=bool)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            patch_y_start = i * patch_h
            patch_y_end = (i + 1) * patch_h
            patch_x_start = j * patch_w
            patch_x_end = (j + 1) * patch_w
            
            # Patch is considered inside if its center is inside rails
            center_y = (patch_y_start + patch_y_end) // 2
            center_x = (patch_x_start + patch_x_end) // 2
            
            patch_mask[i * grid_size[1] + j] = pixel_mask[center_y, center_x]
    
    return patch_mask


