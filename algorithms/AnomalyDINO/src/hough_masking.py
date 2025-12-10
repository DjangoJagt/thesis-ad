"""
Hough Line Transform based masking for industrial tote detection.
Detects vertical and horizontal rails and masks everything outside them black.
"""

import cv2
import numpy as np
import math


def robust_industrial_crop(image, output_size=None, wall_ratio=0.06, debug=False):
    """
    Detect vertical and horizontal rails using Hough Transform and mask everything outside them black.
    
    Args:
        image: Input image (BGR or RGB numpy array)
        output_size: Not used (kept for API compatibility)
        wall_ratio: Not used (kept for API compatibility)
        debug: If True, returns both masked image and debug visualization
        
    Returns:
        masked_image: Image with pixels outside rails set to black
        debug_img: (optional) Debug visualization showing detected lines
    """
    H, W = image.shape[:2]
    # Assume input is already in correct color space (RGB from detection.py, BGR from masking.py)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    debug_img = image.copy()
    
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
    
    if debug:
        # Draw search region boundaries
        cv2.line(debug_img, (roi_w_left, 0), (roi_w_left, H), (255, 0, 0), 1)
        cv2.line(debug_img, (W-roi_w_right, 0), (W-roi_w_right, H), (255, 0, 0), 1)
        cv2.line(debug_img, (0, roi_h_top), (W, roi_h_top), (255, 0, 0), 1)
        cv2.line(debug_img, (0, H-roi_h_bottom), (W, H-roi_h_bottom), (255, 0, 0), 1)
    
    def find_vertical_lines(roi_edges, is_left, offset_x):
        """Find best vertical line (87-93°) in region of interest."""
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
                if debug:
                    cv2.line(debug_img, (x1 + offset_x, y1), (x2 + offset_x, y2), (0, 255, 255), 2)
                
                if (is_left and avg_x > best_x_score) or (not is_left and avg_x < best_x_score):
                    best_x_score = avg_x
                    best_line = (x1, y1, x2, y2)
        
        return best_line
    
    def find_horizontal_lines(roi_edges, is_top, offset_y):
        """Find best horizontal line (<3° or >177°) in region of interest."""
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
                if debug:
                    cv2.line(debug_img, (x1, y1 + offset_y), (x2, y2 + offset_y), (255, 255, 0), 2)
                
                if (is_top and avg_y > best_y_score) or (not is_top and avg_y < best_y_score):
                    best_y_score = avg_y
                    best_line = (x1, y1, x2, y2)
        
        return best_line
    
    # Find left and right vertical lines
    l_line = find_vertical_lines(edges[:, 0:roi_w_left], True, 0)
    r_line = find_vertical_lines(edges[:, W-roi_w_right:W], False, W-roi_w_right)
    
    # Find top and bottom horizontal lines
    t_line = find_horizontal_lines(edges[0:roi_h_top, :], True, 0)
    b_line = find_horizontal_lines(edges[H-roi_h_bottom:H, :], False, H-roi_h_bottom)
    
    def extrapolate_line(line, offset_x=0):
        """Extrapolate vertical line to full image height."""
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
        """Extrapolate horizontal line to full image width."""
        if line is None: 
            return None
        x1, y1, x2, y2 = line
        y1, y2 = y1 + offset_y, y2 + offset_y
        if y2 == y1: 
            return (0, y1), (W, y1)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (0, int(intercept)), (W, int(slope * W + intercept))
    
    # Extrapolate lines or use fallback positions (matches masking.py)
    l_pts = extrapolate_line(l_line) or ((int(W*0.12), 0), (int(W*0.12), H))
    r_pts = extrapolate_line(r_line, W-roi_w_right) or ((int(W*0.88), 0), (int(W*0.88), H))
    t_pts = extrapolate_horizontal_line(t_line) or ((0, int(H*0.02)), (W, int(H*0.02)))
    b_pts = extrapolate_horizontal_line(b_line, H-roi_h_bottom) or ((0, int(H*0.92)), (W, int(H*0.92)))
    
    if debug:
        # Draw final rails on debug image (green lines)
        cv2.line(debug_img, l_pts[0], l_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, r_pts[0], r_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, t_pts[0], t_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, b_pts[0], b_pts[1], (0, 255, 0), 2)
    
    # Create masked image: set pixels outside rails to black
    masked_image = image.copy()
    
    def get_x_at_y(pts, y):
        """Get X coordinate of line at given Y."""
        (x1, y1), (x2, y2) = pts
        if x1 == x2: 
            return x1
        slope = (x2 - x1) / (y2 - y1)
        return int(x1 + slope * (y - y1))
    
    def get_y_at_x(pts, x):
        """Get Y coordinate of line at given X."""
        (x1, y1), (x2, y2) = pts
        if y1 == y2: 
            return y1
        slope = (y2 - y1) / (x2 - x1)
        return int(y1 + slope * (x - x1))
    
    # Mask left side (x < left_line)
    for y in range(H):
        left_x = get_x_at_y(l_pts, y)
        if left_x > 0:
            masked_image[y, :left_x] = 0
    
    # Mask right side (x > right_line)
    for y in range(H):
        right_x = get_x_at_y(r_pts, y)
        if right_x < W:
            masked_image[y, right_x:] = 0
    
    # Mask top side (y < top_line)
    for x in range(W):
        top_y = get_y_at_x(t_pts, x)
        if top_y > 0:
            masked_image[:top_y, x] = 0
    
    # Mask bottom side (y > bottom_line)
    for x in range(W):
        bottom_y = get_y_at_x(b_pts, x)
        if bottom_y < H:
            masked_image[bottom_y:, x] = 0
    
    if debug:
        return masked_image, debug_img
    else:
        return masked_image
