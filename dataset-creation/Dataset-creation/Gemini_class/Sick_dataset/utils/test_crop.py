"""
Test script for visualizing floor tile extraction.
Compares 2-tile vs 4-tile extraction for verification.

Usage:
    python test_crop.py --image <path_to_image> [--output <output_dir>]
    
Example:
    python test_crop.py --image /path/to/tote_image.jpg --output ./tile_test_results
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from crop import robust_industrial_crop, extract_floor_tiles, extract_floor_tiles_4x





def save_individual_tiles(tiles_2, tiles_4, output_dir):
    """
    Saves individual tiles for detailed inspection.
    
    Args:
        tiles_2: tuple of (tile_top, tile_bottom)
        tiles_4: tuple of (tile_tl, tile_tr, tile_bl, tile_br)
        output_dir: Directory to save the tiles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tile_top, tile_bottom = tiles_2
    tile_tl, tile_tr, tile_bl, tile_br = tiles_4
    
    # Save 2-tile version
    cv2.imwrite(str(output_dir / "2tile_top.png"), tile_top)
    cv2.imwrite(str(output_dir / "2tile_bottom.png"), tile_bottom)
    print(f"✓ 2-tile images saved to {output_dir}/")
    
    # Save 4-tile version
    cv2.imwrite(str(output_dir / "4tile_top_left.png"), tile_tl)
    cv2.imwrite(str(output_dir / "4tile_top_right.png"), tile_tr)
    cv2.imwrite(str(output_dir / "4tile_bottom_left.png"), tile_bl)
    cv2.imwrite(str(output_dir / "4tile_bottom_right.png"), tile_br)
    print(f"✓ 4-tile images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize floor tile extraction (2-tile vs 4-tile)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the tote image to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tile_test_results",
        help="Output directory for results (default: ./tile_test_results)"
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    output_dir = Path(args.output)
    
    # Validate input
    if not image_path.exists():
        print(f"✗ Error: Image file not found: {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"✗ Error: Could not read image: {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Step 1: Crop the image
    print("\n--- Step 1: Applying robust industrial crop ---")
    try:
        cropped_img, debug_img = robust_industrial_crop(img, debug_mode=True)
        print(f"✓ Cropped image shape: {cropped_img.shape}")
    except Exception as e:
        print(f"✗ Error during cropping: {e}")
        return
    
    # Step 2: Extract 2 tiles
    print("\n--- Step 2: Extracting 2 tiles ---")
    try:
        tiles_2 = extract_floor_tiles(cropped_img)
        tile_top, tile_bottom = tiles_2
        print(f"✓ Top tile shape: {tile_top.shape}")
        print(f"✓ Bottom tile shape: {tile_bottom.shape}")
    except Exception as e:
        print(f"✗ Error during 2-tile extraction: {e}")
        return
    
    # Step 3: Extract 4 tiles
    print("\n--- Step 3: Extracting 4 tiles ---")
    try:
        tiles_4 = extract_floor_tiles_4x(cropped_img)
        tile_tl, tile_tr, tile_bl, tile_br = tiles_4
        print(f"✓ Top-Left tile shape: {tile_tl.shape}")
        print(f"✓ Top-Right tile shape: {tile_tr.shape}")
        print(f"✓ Bottom-Left tile shape: {tile_bl.shape}")
        print(f"✓ Bottom-Right tile shape: {tile_br.shape}")
    except Exception as e:
        print(f"✗ Error during 4-tile extraction: {e}")
        return
    
    # Step 4: Save results
    print("\n--- Step 4: Saving results ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cropped image
    cv2.imwrite(str(output_dir / "cropped_image.png"), cropped_img)
    cv2.imwrite(str(output_dir / "cropped_debug.png"), debug_img)
    print(f"✓ Cropped images saved")
    
    # Save individual tiles
    save_individual_tiles(tiles_2, tiles_4, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - cropped_image.png: The final cropped tote image")
    print(f"  - cropped_debug.png: Debug visualization with detected rails")
    print(f"  - 2tile_top.png, 2tile_bottom.png: 2-tile extraction")
    print(f"  - 4tile_top_left.png, 4tile_top_right.png, 4tile_bottom_left.png, 4tile_bottom_right.png: 4-tile extraction")


if __name__ == "__main__":
    main()
