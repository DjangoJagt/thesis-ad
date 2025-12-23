import argparse
from pathlib import Path
import cv2
import numpy as np
import math

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_DIRS = ["12461215"]#["10512371", "11299303", "11463862", "11478313", "11738660", "11780914", "11801970", "11810623", "12077488", "12461215"]
DEFAULT_SUBPATH = Path("test") / "issue"

def robust_iphone_crop(image, wall_ratio=0.06):
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_img = image.copy()
    
    # 1. Edge Detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # ---------------------------------------------------------
    # NEW TRAPEZOIDAL ROI LOGIC
    # ---------------------------------------------------------
    # Instead of slicing [:, 0:200], we create a polygon mask.
    # This allows us to look "Wider" at the top and "Narrower" at the bottom.
    
    # Config: How wide (%) to search at Top vs Bottom
    top_w_pct = 0.42      # Search outer 35% at the top (Wide)
    bot_w_pct = 0.13      # Search outer 15% at the bottom (Narrow to avoid products)
    
    # Calculate pixel widths
    t_w = int(W * top_w_pct)
    b_w = int(W * bot_w_pct)
    
    # Define ROI Polygons (Trapezoids)
    # Left Trapezoid: (0,0) -> (t_w, 0) -> (b_w, H) -> (0, H)
    left_roi_poly = np.array([
        [0, 0], [t_w, 0], [b_w, H], [0, H]
    ], np.int32)
    
    # Right Trapezoid: (W,0) -> (W-t_w, 0) -> (W-b_w, H) -> (W, H)
    right_roi_poly = np.array([
        [W, 0], [W - t_w, 0], [W - b_w, H], [W, H]
    ], np.int32)

    # Horizontal/top/bottom logic removed; only vertical crop is kept

    # --- Visualizing the TRAPEZOIDS in Blue ---
    cv2.polylines(debug_img, [left_roi_poly], True, (255, 0, 0), 2)
    cv2.polylines(debug_img, [right_roi_poly], True, (255, 0, 0), 2)

    def apply_poly_mask(edge_img, poly):
        """Returns edge image with everything OUTSIDE the poly set to black."""
        mask = np.zeros_like(edge_img)
        cv2.fillPoly(mask, [poly], 255)
        return cv2.bitwise_and(edge_img, mask)

    # Create Masked Edge Maps for Hough Transform
    # This replaces the simple slicing "edges[:, 0:w]"
    l_edges_masked = apply_poly_mask(edges, left_roi_poly)
    r_edges_masked = apply_poly_mask(edges, right_roi_poly)
    
    # Top/bottom masked edge maps removed

    def find_lines(masked_edges):
        return cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=600,      #400
            minLineLength=H//4,    #H//6
            maxLineGap=20             #20
        )

    # ---------------------------------------------------------
    # ANGLE LOGIC (Keep Split Angles)
    # ---------------------------------------------------------
    def get_best_vertical(lines, is_left, offset_x=0): # offset_x not really needed for Poly but kept for compat
        if lines is None: return None
        best_line = None
        best_x_score = -1 if is_left else  99999
        # best_x_score = 99999 if is_left else  -1
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Normalize Top-to-Bottom
            if y1 > y2: x1, y1, x2, y2 = x2, y2, x1, y1
            
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            valid_angle = False
            
            if is_left:
                # Left Wall: Expect \ (70-95 deg)
                if 96 < angle < 120: valid_angle = True
            else:
                # Right Wall: Expect / (85-110 deg)
                if 60 < angle < 84: valid_angle = True
            
            if valid_angle:
                # Debug draw in Cyan
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                
                # Logic: Find OUTERMOST lines
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

    # Horizontal logic removed

    # EXECUTE SEARCH
    l_candidates = find_lines(l_edges_masked)
    r_candidates = find_lines(r_edges_masked)
    # Horizontal candidates removed

    l_line = get_best_vertical(l_candidates, True)
    r_line = get_best_vertical(r_candidates, False)
    # Horizontal best lines removed

    # ---------------------------------------------------------
    # EXTRAPOLATION
    # ---------------------------------------------------------
    def extrapolate_vertical(line):
        if line is None: return None
        x1, y1, x2, y2 = line
        if x2 == x1: return (x1, 0), (x1, H)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x_top = int((0 - intercept) / slope)
        x_bot = int((H - intercept) / slope)
        return (x_top, 0), (x_bot, H)
    
    # Horizontal extrapolation removed

    # Fallbacks now use the TRAPEZOID defaults if nothing found
    l_pts = extrapolate_vertical(l_line) or ((0, 0), (0, H))
    r_pts = extrapolate_vertical(r_line) or ((W, 0), (W, H))
    # Horizontal final points removed
    
    # Draw Green Final Lines
    for pts in [l_pts, r_pts]:
        cv2.line(debug_img, pts[0], pts[1], (0, 255, 0), 3)

    # Apply Polygon Mask
    masked_image = image.copy()
    
    # Left
    pts = np.array([[0,0], [0,H], l_pts[1], l_pts[0]])
    cv2.fillPoly(masked_image, [pts], (0,0,0))
    # Right
    pts = np.array([r_pts[0], r_pts[1], [W,H], [W,0]])
    cv2.fillPoly(masked_image, [pts], (0,0,0))
    # Top/bottom masking removed

    return masked_image, debug_img

# --- UPDATED BOILERPLATE FOR BATCH PROCESSING ---

def process_all_images(base_dir, class_dirs=DEFAULT_CLASS_DIRS, 
                       output_dir=None, wall_ratio=0.06):
    base_dir = Path(base_dir)
    # Output to a folder named 'masked_results' (or whatever you prefer)
    output_dir = Path(output_dir or base_dir / "all_masked")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0

    for class_dir in class_dirs:
        # Define where to look. 
        # using rglob("*") finds ALL files recursively (train/good, test/issue, etc.)
        search_path = base_dir / class_dir
        
        if not search_path.exists():
            print(f"[SKIP] Directory not found: {search_path}")
            continue

        print(f"\nScanning: {search_path}...")
        
        # Gather all valid images in this folder and subfolders
        all_images = [
            p for p in search_path.rglob("*") 
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
        ]

        if not all_images:
            print(f"  -> No images found.")
            continue

        print(f"  -> Found {len(all_images)} images. Processing...")

        for image_path in all_images:
            # Load
            image = cv2.imread(str(image_path))
            if image is None: 
                print(f"  [ERR] Could not load {image_path.name}")
                continue

            # Process
            masked, debug = robust_iphone_crop(image)
            
            # Save 
            # We prefix the filename with the class_id to keep them unique
            # e.g. "10512371_image_001_masked.jpg"
            save_name_base = f"{class_dir}_{image_path.stem}"
            
            cv2.imwrite(str(output_dir / f"{save_name_base}_masked.jpg"), masked)
            # Uncomment next line if you also want to save the debug lines for ALL images
            cv2.imwrite(str(output_dir / f"{save_name_base}_debug.jpg"), debug)
            
            total_processed += 1
            
            # Optional: Print progress every 10 images to avoid spamming console
            if total_processed % 10 == 0:
                print(f"  ...processed {total_processed} images total")

    print(f"\nDONE! Processed {total_processed} images.")
    print(f"Results saved to: {output_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=None)
    args = parser.parse_args()
    base_dir = args.base_dir or Path(__file__).parent
    
    # Call the new batch processing function
    process_all_images(base_dir=base_dir)

if __name__ == "__main__":
    main()

# # --- BOILERPLATE (Same as before) ---
# def find_first_image(directory, suffixes=VALID_IMAGE_SUFFIXES):
#     directory = Path(directory)
#     if not directory.exists(): return None
#     for candidate in sorted(directory.iterdir()):
#         if candidate.is_file() and candidate.suffix.lower() in suffixes:
#             return candidate
#     return None

# def preview_sample_images(base_dir, class_dirs=DEFAULT_CLASS_DIRS, subpath=DEFAULT_SUBPATH, 
#                          output_dir=None, wall_ratio=0.06):
#     base_dir = Path(base_dir)
#     output_dir = Path(output_dir or base_dir / "trapezoid_masked_previews")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for class_dir in class_dirs:
#         image_path = find_first_image(base_dir / class_dir / subpath) or find_first_image(base_dir / class_dir)
#         if image_path is None:
#             print(f"[WARN] No image found in {class_dir}")
#             continue

#         print(f"Processing: {image_path.name}")
#         image = cv2.imread(str(image_path))
#         if image is None: continue

#         masked, debug = robust_iphone_crop(image)
        
#         cv2.imwrite(str(output_dir / f"{image_path.stem}_masked.jpg"), masked)
#         cv2.imwrite(str(output_dir / f"{image_path.stem}_debug.jpg"), debug)
#         print(f"Saved to {output_dir}")

# def main():
#     parser = argparse.ArgumentParser()
#     # If not provided, default to the script's own directory (dataset_cropped)
#     parser.add_argument("--base-dir", type=Path, default=None)
#     args = parser.parse_args()
#     base_dir = args.base_dir or Path(__file__).parent
#     preview_sample_images(base_dir=base_dir)

# if __name__ == "__main__":
#     main()