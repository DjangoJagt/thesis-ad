import cv2
import numpy as np
import math

def robust_industrial_crop(image, debug_mode = False):
    """
    1. Detects vertical/horizontal rails using Hough Transform.
    2. Masks everything outside them black using polygon filling (faster than loops).
    3. CROPS the image to the bounding box of the non-masked area.
    
    Returns: 
        final_image (cropped & masked), 
        debug_image (visualization of lines)
    """
    if image is None:
        raise ValueError("Image provided to robust_industrial_crop is None")

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_img = image.copy()
    
    # --- STEP 1: EDGE DETECTION ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    
    # --- STEP 2: ROI DEFINITIONS ---
    # We look for rails in specific strips of the image to avoid product interference
    roi_w_left = int(W * 0.31)
    roi_w_right = int(W * 0.25)
    roi_h_top = int(H * 0.08)
    roi_h_bottom = int(H * 0.13)

    # --- STEP 3: HOUGH TRANSFORM HELPERS ---
    def find_line(roi_edges, is_vertical, offset_x=0, offset_y=0):
        # Tune parameters for long rail lines
        if is_vertical:
            min_len = H // 7
            max_gap = 60
            theta = np.pi / 180
            angle_min, angle_max = 88, 92
        else:
            min_len = W // 6
            max_gap = 80
            theta = np.pi / 180
            # Horizontal lines are near 0 or 180 degrees
            angle_limit = 3 

        lines = cv2.HoughLinesP(roi_edges, 1, theta, threshold=50, 
                                minLineLength=min_len, maxLineGap=max_gap)
        if lines is None:
            return None
        
        best_line = None
        # We want the "innermost" lines (closest to image center)
        # Init with worst possible score
        best_score = -1 if (offset_x == 0 and offset_y == 0) else 99999

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            valid = False
            if is_vertical and (angle_min < angle < angle_max):
                score = (x1 + x2) / 2 + offset_x
                valid = True
            elif not is_vertical and (angle < angle_limit or angle > (180 - angle_limit)):
                score = (y1 + y2) / 2 + offset_y
                valid = True
            
            if valid:
                # Logic: We want to shrink the ROI. 
                # For Top/Left lines: maximize the coordinate (move right/down).
                # For Bottom/Right lines: minimize the coordinate (move left/up).
                is_start_side = (offset_x == 0 and offset_y == 0)
                
                if (is_start_side and score > best_score) or (not is_start_side and score < best_score):
                    best_score = score
                    best_line = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
                    
        return best_line

    # --- STEP 4: DETECT LINES ---
    # Slicing the edge map to only look in relevant regions
    l_line = find_line(edges[:, 0:roi_w_left], True, offset_x=0)
    r_line = find_line(edges[:, W-roi_w_right:W], True, offset_x=W-roi_w_right)
    t_line = find_line(edges[0:roi_h_top, :], False, offset_y=0)
    b_line = find_line(edges[H-roi_h_bottom:H, :], False, offset_y=H-roi_h_bottom)

    # --- STEP 5: EXTRAPOLATE TO FULL IMAGE EDGES ---
    def get_full_line(line, is_vertical):
        if line is None: return None
        x1, y1, x2, y2 = line
        if is_vertical:
            if x2 == x1: return (x1, 0), (x1, H) # Perfectly vertical
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return (int(-b/m), 0), (int((H-b)/m), H) # Top intersect, Bottom intersect
        else:
            if y2 == y1: return (0, y1), (W, y1) # Perfectly horizontal
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return (0, int(b)), (W, int(m*W + b)) # Left intersect, Right intersect

    # Defaults if no lines found (fall back to safe margins)
    l_pts = get_full_line(l_line, True) or ((int(W*0.12), 0), (int(W*0.05), H))
    r_pts = get_full_line(r_line, True) or ((int(W*0.88), 0), (int(W*0.88), H))
    t_pts = get_full_line(t_line, False) or ((0, int(H*0.02)), (W, int(H*0.02)))
    b_pts = get_full_line(b_line, False) or ((0, int(H*0.92)), (W, int(H*0.92)))

    # --- STEP 6: FAST POLYGON MASKING ---
    mask = np.full((H, W), 255, dtype=np.uint8) # Start White
    
    # Helper to draw black polygon
    def mask_region(pts):
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 0)

    # Mask Left (Polygon from x=0 to line)
    mask_region([(0,0), l_pts[0], l_pts[1], (0, H)])
    # Mask Right (Polygon from line to x=W)
    mask_region([r_pts[0], (W,0), (W,H), r_pts[1]])
    # Mask Top (Polygon from y=0 to line)
    mask_region([(0,0), (W,0), (W, t_pts[1][1]), (0, t_pts[0][1])])
    # Mask Bottom (Polygon from line to y=H)
    mask_region([(0, b_pts[0][1]), (W, b_pts[1][1]), (W,H), (0,H)])

    # Apply mask to image (everything outside rails becomes Black)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # --- STEP 7: STRAIGHT CROP TO CONTENT ---
    # Find the bounding box of the white area in the mask
    # This automatically finds the min_x, max_x, min_y, max_y of valid data
    ys, xs = np.where(mask > 5)
    
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: Mask killed everything (rare), return original
        return image, debug_img

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    pad_x = int(0.01 * W)   # ~1.5% width
    pad_y = int(0.01 * H)    # ~2% height

    min_x = max(0, min_x - pad_x)
    max_x = min(W, max_x + pad_x)
    min_y = max(0, min_y - pad_y)
    max_y = min(H, max_y + pad_y)

    final_image = masked_img[min_y:max_y+1, min_x:max_x+1]
    
    # Debug visualization
    if debug_mode:
        cv2.line(debug_img, l_pts[0], l_pts[1], (0, 0, 255), 2)
        cv2.line(debug_img, r_pts[0], r_pts[1], (0, 0, 255), 2)
        cv2.line(debug_img, t_pts[0], t_pts[1], (0, 0, 255), 2)
        cv2.line(debug_img, b_pts[0], b_pts[1], (0, 0, 255), 2)
        cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    return final_image, debug_img


def extract_floor_tiles(cropped_image):
    """
    Takes the already-cropped tote image and:
    1. Removes 15% from left/right (the walls).
    2. Removes 10% from top/bottom (the extreme edges).
    3. Splits the remaining 'floor area' into two vertical tiles.
    """
    h, w = cropped_image.shape[:2]
    
    # Shave off the walls (adjust these % based on your camera angle)
    right_margin = int(w * 0.14)
    left_margin = int(w * 0.04)
    top_margin = int(h * 0.10)
    bottom_margin = int(h * 0.12)
    
    floor_area = cropped_image[top_margin:h-bottom_margin, 
                               left_margin:w-right_margin]
    
    # Split into Top Half and Bottom Half
    fh, fw = floor_area.shape[:2]
    mid = fh // 2
    
    tile_top = floor_area[0:mid, :]
    tile_bottom = floor_area[mid:fh, :]
    
    return tile_top, tile_bottom


def extract_floor_tiles_4x(cropped_image):
    """
    Takes the already-cropped tote image and:
    1. Removes 15% from left/right (the walls).
    2. Removes 10% from top/bottom (the extreme edges).
    3. Splits the remaining 'floor area' into four tiles: top-left, top-right, bottom-left, bottom-right.
    
    Returns:
        tile_top_left, tile_top_right, tile_bottom_left, tile_bottom_right
    """
    h, w = cropped_image.shape[:2]
    
    # Shave off the walls (adjust these % based on your camera angle)
    right_margin = int(w * 0.14)
    left_margin = int(w * 0.04)
    top_margin = int(h * 0.10)
    bottom_margin = int(h * 0.12)
    
    floor_area = cropped_image[top_margin:h-bottom_margin, 
                               left_margin:w-right_margin]
    
    # Split into 4 tiles: top-left, top-right, bottom-left, bottom-right
    fh, fw = floor_area.shape[:2]
    mid_h = fh // 2
    mid_w = fw // 2
    
    tile_top_left = floor_area[0:mid_h, 0:mid_w]
    tile_top_right = floor_area[0:mid_h, mid_w:fw]
    tile_bottom_left = floor_area[mid_h:fh, 0:mid_w]
    tile_bottom_right = floor_area[mid_h:fh, mid_w:fw]
    
    return tile_top_left, tile_top_right, tile_bottom_left, tile_bottom_right 