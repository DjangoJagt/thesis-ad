import argparse
from pathlib import Path
import cv2
import numpy as np
import math

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_DIRS = ["10074656", "10468098", "11421515", "90006036", "10075934", "10760941"]
DEFAULT_SUBPATH = Path("test") / "issue"
# DEFAULT_SUBPATH = Path("train") / "good"

def compute_hough_feature_mask(image, grid_size):
    """
    Compute a feature-level mask based on Hough line detection of tote rails.
    Returns a boolean mask indicating which patches are inside the rails.
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    
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
    
    return patch_mask, pixel_mask, (l_pts, r_pts, t_pts, b_pts)


def robust_industrial_crop(image, output_size=None, wall_ratio=0.06):
    """Detect vertical rails using Hough Transform and mask everything outside them black."""
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_img = image.copy()
    
    # Edge detection with automatic Canny threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    
    # Detect vertical lines in left and right regions
    roi_w_left = int(W * 0.31)
    roi_w_right = int(W * 0.28) 
    cv2.line(debug_img, (roi_w_left, 0), (roi_w_left, H), (255, 0, 0), 1)
    cv2.line(debug_img, (W-roi_w_right, 0), (W-roi_w_right, H), (255, 0, 0), 1)
    
    # Detect horizontal lines in top and bottom regions
    roi_h_top = int(H * 0.08)
    roi_h_bottom = int(H * 0.13)
    cv2.line(debug_img, (0, roi_h_top), (W, roi_h_top), (255, 0, 0), 1)
    cv2.line(debug_img, (0, H-roi_h_bottom), (W, H-roi_h_bottom), (255, 0, 0), 1)
    
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
                cv2.line(debug_img, (x1 + offset_x, y1), (x2 + offset_x, y2), (0, 255, 255), 2)
                
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
                cv2.line(debug_img, (x1, y1 + offset_y), (x2, y2 + offset_y), (255, 255, 0), 2)
                
                if (is_top and avg_y > best_y_score) or (not is_top and avg_y < best_y_score):
                    best_y_score = avg_y
                    best_line = (x1, y1, x2, y2)
        
        return best_line
    
    # Find left and right lines
    l_line = find_vertical_lines(edges[:, 0:roi_w_left], True, 0)
    r_line = find_vertical_lines(edges[:, W-roi_w_right:W], False, W-roi_w_right)
    
    # Find top and bottom lines
    t_line = find_horizontal_lines(edges[0:roi_h_top, :], True, 0)
    b_line = find_horizontal_lines(edges[H-roi_h_bottom:H, :], False, H-roi_h_bottom)
    
    # Extrapolate lines to full height
    def extrapolate_line(line, offset_x=0):
        if line is None: return None
        x1, y1, x2, y2 = line
        x1, x2 = x1 + offset_x, x2 + offset_x
        if x2 == x1: return (x1, 0), (x1, H)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (int((0 - intercept) / slope), 0), (int((H - intercept) / slope), H)
    
    def extrapolate_horizontal_line(line, offset_y=0):
        if line is None: return None
        x1, y1, x2, y2 = line
        y1, y2 = y1 + offset_y, y2 + offset_y
        if y2 == y1: return (0, y1), (W, y1)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (0, int(intercept)), (W, int(slope * W + intercept))
    
    l_pts = extrapolate_line(l_line) or ((int(W*0.12), 0), (int(W*0.12), H))
    r_pts = extrapolate_line(r_line, W-roi_w_right) or ((int(W*0.88), 0), (int(W*0.88), H))
    t_pts = extrapolate_horizontal_line(t_line) or ((0, int(H*0.02)), (W, int(H*0.02)))
    b_pts = extrapolate_horizontal_line(b_line, H-roi_h_bottom) or ((0, int(H*0.92)), (W, int(H*0.92)))
    
    # Draw rails on debug image
    cv2.line(debug_img, l_pts[0], l_pts[1], (0, 255, 0), 2)
    cv2.line(debug_img, r_pts[0], r_pts[1], (0, 255, 0), 2)
    cv2.line(debug_img, t_pts[0], t_pts[1], (0, 255, 0), 2)
    cv2.line(debug_img, b_pts[0], b_pts[1], (0, 255, 0), 2)
    
    # Draw angle range visualization lines (purple) starting from blue line midpoints
    left_mid_x = roi_w_left
    left_mid_y = H // 2
    top_mid_x = W // 2
    top_mid_y = roi_h_top
    
    # Length of the line to draw (make it long enough to cover the screen)
    line_len = 1000 

    # --- Vertical Lines (75° and 105°) ---
    # In OpenCV: 90° is straight down. 
    # So 75° leans slightly right, 105° leans slightly left (or vice versa depending on your origin)
    # We use standard trig: x = x0 + L*cos(a), y = y0 + L*sin(a)

    # 75 Degree Line
    angle_75 = math.radians(87) 
    x_end_75 = int(left_mid_x + line_len * math.cos(angle_75))
    y_end_75 = int(left_mid_y + line_len * math.sin(angle_75))
    cv2.line(debug_img, (left_mid_x, left_mid_y), (x_end_75, y_end_75), (255, 0, 255), 2)

    # 105 Degree Line
    angle_105 = math.radians(93)
    x_end_105 = int(left_mid_x + line_len * math.cos(angle_105))
    y_end_105 = int(left_mid_y + line_len * math.sin(angle_105))
    cv2.line(debug_img, (left_mid_x, left_mid_y), (x_end_105, y_end_105), (255, 0, 255), 2)


    # --- Horizontal Lines (8° and 172°) ---
    # In OpenCV: 0° is straight Right.

    # 8 Degree Line (Slightly down-right)
    angle_8 = math.radians(3)
    x_end_8 = int(top_mid_x + line_len * math.cos(angle_8))
    y_end_8 = int(top_mid_y + line_len * math.sin(angle_8))
    cv2.line(debug_img, (top_mid_x, top_mid_y), (x_end_8, y_end_8), (255, 0, 255), 2)

    # 172 Degree Line (Slightly down-left)
    # Note: 172° points to the Left. 
    angle_172 = math.radians(177)
    x_end_172 = int(top_mid_x + line_len * math.cos(angle_172))
    y_end_172 = int(top_mid_y + line_len * math.sin(angle_172))
    cv2.line(debug_img, (top_mid_x, top_mid_y), (x_end_172, y_end_172), (255, 0, 255), 2)

    # Create output: original image with everything outside the vertical rails masked black
    masked_image = image.copy()
    
    def get_x_at_y(pts, y):
        (x1, y1), (x2, y2) = pts
        if x1 == x2: return x1
        slope = (x2 - x1) / (y2 - y1)
        return int(x1 + slope * (y - y1))
    
    def get_y_at_x(pts, x):
        (x1, y1), (x2, y2) = pts
        if y1 == y2: return y1
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
    
    return masked_image, debug_img

# --- BOILERPLATE (Same as before) ---
def find_first_image(directory, suffixes=VALID_IMAGE_SUFFIXES):
    directory = Path(directory)
    if not directory.exists(): return None
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in suffixes:
            return candidate
    return None

def preview_sample_images(base_dir, class_dirs=DEFAULT_CLASS_DIRS, subpath=DEFAULT_SUBPATH, 
                         output_dir=None, wall_ratio=0.06, grid_size=(32, 32)):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir or base_dir / "featuremask_previews")
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        image_path = find_first_image(base_dir / class_dir / subpath) or find_first_image(base_dir / class_dir)
        if image_path is None:
            print(f"[WARN] No image found in {class_dir}")
            continue

        print(f"Processing: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None: continue

        # Test feature-level masking
        patch_mask, pixel_mask, rail_lines = compute_hough_feature_mask(image, grid_size)
        l_pts, r_pts, t_pts, b_pts = rail_lines
        
        # Create visualization
        H, W = image.shape[:2]
        debug_img = image.copy()
        
        # Draw rails
        cv2.line(debug_img, l_pts[0], l_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, r_pts[0], r_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, t_pts[0], t_pts[1], (0, 255, 0), 2)
        cv2.line(debug_img, b_pts[0], b_pts[1], (0, 255, 0), 2)
        
        # Visualize patch mask
        patch_h = H // grid_size[0]
        patch_w = W // grid_size[1]
        
        patch_viz = image.copy()
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                patch_idx = i * grid_size[1] + j
                if not patch_mask[patch_idx]:
                    # Draw red rectangle on excluded patches
                    y1 = i * patch_h
                    y2 = (i + 1) * patch_h
                    x1 = j * patch_w
                    x2 = (j + 1) * patch_w
                    cv2.rectangle(patch_viz, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        # Blend with original
        patch_viz = cv2.addWeighted(image, 0.6, patch_viz, 0.4, 0)
        
        # Draw grid lines
        for i in range(grid_size[0] + 1):
            y = i * patch_h
            cv2.line(patch_viz, (0, y), (W, y), (255, 255, 255), 1)
        for j in range(grid_size[1] + 1):
            x = j * patch_w
            cv2.line(patch_viz, (x, 0), (x, H), (255, 255, 255), 1)
        
        # Create pixel mask visualization (for comparison)
        pixel_viz = np.zeros_like(image)
        pixel_viz[pixel_mask] = image[pixel_mask]
        
        # Save outputs
        cv2.imwrite(str(output_dir / f"{class_dir}_{image_path.stem}_rails.jpg"), debug_img)
        cv2.imwrite(str(output_dir / f"{class_dir}_{image_path.stem}_patches.jpg"), patch_viz)
        cv2.imwrite(str(output_dir / f"{class_dir}_{image_path.stem}_pixel_mask.jpg"), pixel_viz)
        
        print(f"[OK] Patches inside rails: {patch_mask.sum()}/{len(patch_mask)} ({100*patch_mask.sum()/len(patch_mask):.1f}%)")
        print(f"[OK] Saved visualizations to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path(".")) 
    parser.add_argument("--wall-ratio", type=float, default=0.06)
    parser.add_argument("--grid-size", type=int, nargs=2, default=[32, 32], help="Grid size for patches (height width)")
    args = parser.parse_args()
    preview_sample_images(base_dir=args.base_dir, wall_ratio=args.wall_ratio, grid_size=tuple(args.grid_size))

if __name__ == "__main__":
    main()