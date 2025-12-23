"""
Hough Line Transform based masking for industrial tote detection.
Detects vertical and horizontal rails and creates feature-level masks.
"""

import cv2
import numpy as np
import math


def compute_hough_feature_mask(image, grid_size, patch_size=14):
    """
    Compute a feature-level mask based on Hough line detection of tote rails.
    Runs Hough at full image resolution, then downsamples mask to match grid_size.
    Uses trapezoidal ROI for vertical-only detection (wider at top, narrower at bottom).
    
    Args:
        image: Input image at full resolution (RGB or BGR numpy array)
        grid_size: Tuple (height, width) indicating number of patches in feature grid
        patch_size: DINOv2 patch size (default 14 for ViT-S/14)
        
    Returns:
        mask: Boolean array of shape (grid_size[0] * grid_size[1],) indicating which patches are inside rails
    """
    H, W = image.shape[:2]
    
    # Compute target dimensions for downsampling (from grid_size)
    H_target = grid_size[0] * patch_size
    W_target = grid_size[1] * patch_size
    
    # Convert to grayscale
    if image.ndim == 3:
        # Detect if BGR or RGB by checking which channel is dominant
        ch_means = image.reshape(-1, 3).mean(axis=0)
        if ch_means[2] > ch_means[0]:  # Red channel higher than Blue suggests BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Edge detection with automatic Canny threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Trapezoidal ROI configuration (wider at top, narrower at bottom)
    top_w_pct = 0.42
    bot_w_pct = 0.13
    t_w = int(W * top_w_pct)
    b_w = int(W * bot_w_pct)
    
    # Define ROI Polygons (Trapezoids)
    left_roi_poly = np.array([[0, 0], [t_w, 0], [b_w, H], [0, H]], np.int32)
    right_roi_poly = np.array([[W, 0], [W - t_w, 0], [W - b_w, H], [W, H]], np.int32)
    
    def apply_poly_mask(edge_img, poly):
        """Returns edge image with everything OUTSIDE the poly set to black."""
        mask = np.zeros_like(edge_img)
        cv2.fillPoly(mask, [poly], 255)
        return cv2.bitwise_and(edge_img, mask)
    
    # Create masked edge maps for Hough Transform
    l_edges_masked = apply_poly_mask(edges, left_roi_poly)
    r_edges_masked = apply_poly_mask(edges, right_roi_poly)
    
    def find_vertical_lines(masked_edges):
        return cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=600,
            minLineLength=H // 4,
            maxLineGap=20
        )
    
    def get_best_vertical(lines, is_left):
        if lines is None:
            return None
        best_line = None
        best_x_score = -1 if is_left else 99999
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Normalize Top-to-Bottom
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            valid_angle = False
            
            if is_left:
                if 96 < angle < 120:
                    valid_angle = True
            else:
                if 60 < angle < 84:
                    valid_angle = True
            
            if valid_angle:
                avg_x = (x1 + x2) / 2
                if is_left:
                    if avg_x > best_x_score:
                        best_x_score = avg_x
                        best_line = (x1, y1, x2, y2)
                else:
                    if avg_x < best_x_score:
                        best_x_score = avg_x
                        best_line = (x1, y1, x2, y2)
        return best_line
    
    # Find vertical lines only
    l_candidates = find_vertical_lines(l_edges_masked)
    r_candidates = find_vertical_lines(r_edges_masked)
    
    l_line = get_best_vertical(l_candidates, True)
    r_line = get_best_vertical(r_candidates, False)
    
    def extrapolate_vertical(line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if x2 == x1:
            return (x1, 0), (x1, H)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x_top = int((0 - intercept) / slope)
        x_bot = int((H - intercept) / slope)
        return (x_top, 0), (x_bot, H)
    
    # Extrapolate vertical lines or use fallback positions
    l_pts = extrapolate_vertical(l_line) or ((0, 0), (0, H))
    r_pts = extrapolate_vertical(r_line) or ((W, 0), (W, H))
    
    # Create pixel-level mask at full resolution (vertical only)
    pixel_mask_full = np.zeros((H, W), dtype=bool)
    
    def get_x_at_y(pts, y):
        (x1, y1), (x2, y2) = pts
        if x1 == x2:
            return x1
        slope = (x2 - x1) / (y2 - y1)
        return int(x1 + slope * (y - y1))
    
    # Mark pixels inside vertical rails as True at full resolution
    for y in range(H):
        left_x = get_x_at_y(l_pts, y)
        right_x = get_x_at_y(r_pts, y)
        # Clamp within image bounds
        left_x = max(0, min(W - 1, left_x))
        right_x = max(0, min(W - 1, right_x))
        # Ensure proper ordering
        if left_x > right_x:
            left_x, right_x = right_x, left_x
        pixel_mask_full[y, left_x:right_x + 1] = True
    
    # Downsample pixel mask to target grid dimensions
    # Use bilinear interpolation: convert bool to float, downsample, then threshold
    pixel_mask_float = pixel_mask_full.astype(np.float32)
    pixel_mask_downsampled = cv2.resize(pixel_mask_float, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
    # Threshold at 0.5: a patch is masked if >50% of its area was masked at full resolution
    pixel_mask_target = pixel_mask_downsampled > 0.5
    
    # Convert downsampled pixel mask to patch mask
    patch_h = H_target // grid_size[0]
    patch_w = W_target // grid_size[1]
    
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
            
            # Safely clamp to avoid out-of-bounds access
            center_y = min(center_y, H_target - 1)
            center_x = min(center_x, W_target - 1)
            
            patch_mask[i * grid_size[1] + j] = pixel_mask_target[center_y, center_x]
    
    return patch_mask


def apply_hough_pixel_masking(image):
    """
    Apply Hough line detection and set pixels outside the detected vertical rails to black.
    Uses trapezoidal ROI for vertical-only detection (wider at top, narrower at bottom).
    This is used for reference images before rotation augmentation.
    
    Args:
        image: Input image (RGB numpy array)
        
    Returns:
        masked_image: Image with pixels outside vertical rails set to black (RGB numpy array)
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    
    # Edge detection with automatic Canny threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Trapezoidal ROI configuration (wider at top, narrower at bottom)
    top_w_pct = 0.42
    bot_w_pct = 0.13
    t_w = int(W * top_w_pct)
    b_w = int(W * bot_w_pct)
    
    # Define ROI Polygons (Trapezoids)
    left_roi_poly = np.array([[0, 0], [t_w, 0], [b_w, H], [0, H]], np.int32)
    right_roi_poly = np.array([[W, 0], [W - t_w, 0], [W - b_w, H], [W, H]], np.int32)
    
    def apply_poly_mask(edge_img, poly):
        """Returns edge image with everything OUTSIDE the poly set to black."""
        mask = np.zeros_like(edge_img)
        cv2.fillPoly(mask, [poly], 255)
        return cv2.bitwise_and(edge_img, mask)
    
    # Create masked edge maps for Hough Transform
    l_edges_masked = apply_poly_mask(edges, left_roi_poly)
    r_edges_masked = apply_poly_mask(edges, right_roi_poly)
    
    def find_vertical_lines(masked_edges):
        return cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=600,
            minLineLength=H // 4,
            maxLineGap=20
        )
    
    def get_best_vertical(lines, is_left):
        if lines is None:
            return None
        best_line = None
        best_x_score = -1 if is_left else 99999
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Normalize Top-to-Bottom
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            valid_angle = False
            
            if is_left:
                if 96 < angle < 120:
                    valid_angle = True
            else:
                if 60 < angle < 84:
                    valid_angle = True
            
            if valid_angle:
                avg_x = (x1 + x2) / 2
                if is_left:
                    if avg_x > best_x_score:
                        best_x_score = avg_x
                        best_line = (x1, y1, x2, y2)
                else:
                    if avg_x < best_x_score:
                        best_x_score = avg_x
                        best_line = (x1, y1, x2, y2)
        return best_line
    
    # Find vertical lines only
    l_candidates = find_vertical_lines(l_edges_masked)
    r_candidates = find_vertical_lines(r_edges_masked)
    
    l_line = get_best_vertical(l_candidates, True)
    r_line = get_best_vertical(r_candidates, False)
    
    def extrapolate_vertical(line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if x2 == x1:
            return (x1, 0), (x1, H)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x_top = int((0 - intercept) / slope)
        x_bot = int((H - intercept) / slope)
        return (x_top, 0), (x_bot, H)
    
    # Extrapolate vertical lines or use fallback positions
    l_pts = extrapolate_vertical(l_line) or ((0, 0), (0, H))
    r_pts = extrapolate_vertical(r_line) or ((W, 0), (W, H))
    
    # Create masked image using polygon masking (vertical only)
    masked_image = image.copy()
    
    # Left polygon mask
    pts = np.array([[0, 0], [0, H], l_pts[1], l_pts[0]])
    cv2.fillPoly(masked_image, [pts], (0, 0, 0))
    
    # Right polygon mask
    pts = np.array([r_pts[0], r_pts[1], [W, H], [W, 0]])
    cv2.fillPoly(masked_image, [pts], (0, 0, 0))
    
    return masked_image


