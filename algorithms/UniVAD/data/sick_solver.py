#!/usr/bin/env python3
"""
Generate meta.json for SICK dataset with the same structure as Cognex.
Structure: sick_data/<product_id>/train/good/*.jpg
           sick_data/<product_id>/test/good/*.jpg
           sick_data/<product_id>/test/issue/*.jpg
"""

import os
import json
import glob
import argparse


def generate_sick_meta(root_dir="./sick_data", output_path="./sick_data/meta.json"):
    """Generate meta.json for SICK dataset."""
    
    meta = {
        "train": {},
        "test": {}
    }
    
    # Get all product directories
    product_dirs = [d for d in os.listdir(root_dir) 
                   if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    product_dirs.sort()
    
    print(f"Found {len(product_dirs)} product categories")
    
    for product_id in product_dirs:
        product_path = os.path.join(root_dir, product_id)
        
        # Initialize product entries
        meta["train"][product_id] = []
        meta["test"][product_id] = []
        
        # Process train/good images
        train_good_pattern = os.path.join(product_path, "train/good/*.jpg")
        train_images = glob.glob(train_good_pattern)
        train_images.extend(glob.glob(train_good_pattern.replace(".jpg", ".JPG")))
        train_images.extend(glob.glob(train_good_pattern.replace(".jpg", ".png")))
        train_images.extend(glob.glob(train_good_pattern.replace(".jpg", ".PNG")))
        
        for img_path in sorted(train_images):
            rel_path = os.path.relpath(img_path, root_dir)
            meta["train"][product_id].append({
                "img_path": rel_path,
                "mask_path": "",
                "cls_name": product_id,
                "specie_name": "good",
                "anomaly": 0
            })
        
        # Process test/good images
        test_good_pattern = os.path.join(product_path, "test/good/*.jpg")
        test_good_images = glob.glob(test_good_pattern)
        test_good_images.extend(glob.glob(test_good_pattern.replace(".jpg", ".JPG")))
        test_good_images.extend(glob.glob(test_good_pattern.replace(".jpg", ".png")))
        test_good_images.extend(glob.glob(test_good_pattern.replace(".jpg", ".PNG")))
        
        for img_path in sorted(test_good_images):
            rel_path = os.path.relpath(img_path, root_dir)
            meta["test"][product_id].append({
                "img_path": rel_path,
                "mask_path": "",
                "cls_name": product_id,
                "specie_name": "good",
                "anomaly": 0
            })
        
        # Process test/issue images (anomalies)
        test_issue_pattern = os.path.join(product_path, "test/issue/*.jpg")
        test_issue_images = glob.glob(test_issue_pattern)
        test_issue_images.extend(glob.glob(test_issue_pattern.replace(".jpg", ".JPG")))
        test_issue_images.extend(glob.glob(test_issue_pattern.replace(".jpg", ".png")))
        test_issue_images.extend(glob.glob(test_issue_pattern.replace(".jpg", ".PNG")))
        
        for img_path in sorted(test_issue_images):
            rel_path = os.path.relpath(img_path, root_dir)
            meta["test"][product_id].append({
                "img_path": rel_path,
                "mask_path": "",
                "cls_name": product_id,
                "specie_name": "issue",
                "anomaly": 1
            })
        
        print(f"  {product_id}: {len(meta['train'][product_id])} train, "
              f"{len(test_good_images)} test/good, {len(test_issue_images)} test/issue")
    
    # Save meta.json
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Generated {output_path}")
    
    # Print summary
    total_train = sum(len(v) for v in meta["train"].values())
    total_test = sum(len(v) for v in meta["test"].values())
    print(f"Total: {total_train} train images, {total_test} test images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/sick_data", 
                        help="Root directory of SICK dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for meta.json (default: <root>/meta.json)")
    args = parser.parse_args()
    
    output_path = args.output or os.path.join(args.root, "meta.json")
    generate_sick_meta(args.root, output_path)
