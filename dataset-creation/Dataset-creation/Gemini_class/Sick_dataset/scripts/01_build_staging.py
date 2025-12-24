#!/usr/bin/env python
import re
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator, ValidationError
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---- CONFIG ----
QUALITY_ISSUE_DIR = Path("quality-issue")
STAGING_DIR = Path("sick_data_staging")
MANIFEST_PATH = Path("manifest.csv")

# CHANGED: Updated directory name
REFERENCE_IMG_DIR = Path("template_images") 

FILENAME_REGEX = re.compile(
    r"^(?P<date>\d{8})-(?P<tote_id>\d+)-(?P<sku>\d+)-(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})\.png$"
)


class ManifestRow(BaseModel):
    """
    Data model for the manifest CSV. 
    Matches the input requirements for the Gemini Defect Detector.
    """
    original_filepath: str
    staging_filepath: str
    
    # CHANGED: Updated field name
    sku_template_path: str 
    
    filename: str
    sku: str
    tote_id: str
    date: str           # YYYY-MM-DD
    time: str           # HH:MM:SS
    timestamp_iso: str  # ISO 8601

    # Optional metadata
    product_name: Optional[str] = None
    GT: Optional[str] = None      # Ground Truth label    

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("time")
    @classmethod
    def validate_time(cls, v):
        datetime.strptime(v, "%H:%M:%S")
        return v

    @field_validator("timestamp_iso")
    @classmethod
    def validate_ts(cls, v):
        datetime.fromisoformat(v)
        return v


def parse_filename(filename: str):
    m = FILENAME_REGEX.match(filename)
    if not m:
        return None
    groups = m.groupdict()
    
    # Date/Time Parsing
    date_raw = groups["date"]
    date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    time = f"{groups['hour']}:{groups['minute']}:{groups['second']}"
    ts = f"{date}T{time}"

    # Tote ID Logic
    raw_tote_id = groups["tote_id"]
    clean_tote_id = raw_tote_id[1:] if len(raw_tote_id) > 1 else raw_tote_id

    return {
        "sku": groups["sku"],
        "tote_id": clean_tote_id,
        "date": date,
        "time": time,
        "timestamp_iso": ts,
    }

def get_template_image_path(sku: str) -> str:
    """
    Constructs the path where the template image *should* be.
    """
    # CHANGED: Updated extension to .png
    # Example: template_images/12345.png
    ref_path = REFERENCE_IMG_DIR / f"{sku}.png"
    return str(ref_path)

def build_staging(dry_run: bool = False):
    """Build staging directory and manifest from quality-issue images."""
    logger.info(f"Starting staging build (dry_run={dry_run})")
    
    if not dry_run:
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(QUALITY_ISSUE_DIR.glob("*.png"))
    logger.info(f"Found {len(image_files)} PNG files in {QUALITY_ISSUE_DIR}")
    
    rows = []
    skipped = 0
    copied = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            logger.warning(f"Skipping file with unexpected name: {img_path.name}")
            skipped += 1
            continue

        sku = parsed["sku"]
        sku_dir = STAGING_DIR / sku
        dest_path = sku_dir / img_path.name
        
        # CHANGED: Get template path
        ref_path = get_template_image_path(sku)

        if not dry_run:
            sku_dir.mkdir(parents=True, exist_ok=True)
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copied += 1

        data = {
            "original_filepath": str(img_path),
            "staging_filepath": str(dest_path),
            
            # CHANGED: Updated key name
            "sku_template_path": ref_path,
            
            "filename": img_path.name,
            **parsed,
        }

        try:
            row = ManifestRow(**data)
            rows.append(row.model_dump())
        except ValidationError as e:
            logger.error(f"Validation failed for {img_path.name}: {e}")
            skipped += 1

    if not rows:
        logger.warning("No valid files found.")
        return

    logger.info(f"Successfully processed {len(rows)} files")
    
    df_new = pd.DataFrame(rows)
    
    if dry_run:
        return

    # Logic to handle existing manifest
    # NOTE: Since we changed column names, it is safer to overwrite or verify
    # If you deleted the file before running, this block acts as "Create new"
    if MANIFEST_PATH.exists():
        df_old = pd.read_csv(MANIFEST_PATH)
        # Only merge if columns match, otherwise we might get mess
        if "sku_reference_path" in df_old.columns:
            logger.warning("Old manifest schema detected. Overwriting completely to update columns.")
            df_new.to_csv(MANIFEST_PATH, index=False)
        else:
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=["staging_filepath"])
            df_merged.to_csv(MANIFEST_PATH, index=False)
    else:
        df_new.to_csv(MANIFEST_PATH, index=False)
        logger.info(f"Created manifest.csv with {len(df_new)} rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    build_staging(dry_run=args.dry_run)