import os
import shutil
from pathlib import Path

# --- PATH CONFIGURATION (Adjusted for your screenshot structure) ---
# We assume you run this from the Gemini_class/Sick_dataset/scripts
ROOT_DIR = Path(__file__).parent.parent.parent  # Goes up to Dataset-creation/
FLAGGED_FILE = ROOT_DIR.parent / "Sick" / "flagged_quality_issues_2.txt"
STAGING_DIR = ROOT_DIR / "Sick_dataset" / "sick_data_staging"
REVIEW_DIR = ROOT_DIR / "Sick_dataset" / "sick_data_staging_issues"

def create_issue_review_set():
    # 1. Get unique SKU/Article-IDs from the text file
    flagged_skus = set()
    if not FLAGGED_FILE.exists():
        print(f"Error: Could not find {FLAGGED_FILE}")
        return

    with open(FLAGGED_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                flagged_skus.add(parts[0]) # The Article-ID/SKU

    print(f"Found {len(flagged_skus)} products with at least one flag.")

    # 2. Setup the review folder
    if REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR) # Clear old reviews
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Copy EVERY photo for those SKUs
    copied_count = 0
    for sku in flagged_skus:
        sku_source = STAGING_DIR / sku
        sku_target = REVIEW_DIR / sku

        if sku_source.exists():
            shutil.copytree(sku_source, sku_target)
            copied_count += 1
            print(f"Prepared {sku} for review.")
        else:
            print(f"Warning: SKU folder {sku} not found in staging.")

    print(f"\nFinished! {copied_count} product folders moved to {REVIEW_DIR}")

if __name__ == "__main__":
    create_issue_review_set()