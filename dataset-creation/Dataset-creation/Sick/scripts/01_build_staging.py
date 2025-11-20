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


FILENAME_REGEX = re.compile(
    r"^(?P<date>\d{8})-(?P<tote_id>\d+)-(?P<sku>\d+)-(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})\.png$"
)


class ManifestRow(BaseModel):
    original_filepath: str
    staging_filepath: str
    filename: str
    sku: str
    tote_id: str
    date: str           # YYYY-MM-DD
    time: str           # HH:MM:SS
    timestamp_iso: str  # ISO 8601

    # placeholders voor latere verrijking
    expected_qty: Optional[int] = None
    complaint_flag: Optional[bool] = None
    shopper_flag: Optional[bool] = None
    product_name: Optional[str] = None
    label: Optional[str] = None      # 'good' / 'issue' (later)
    split: Optional[str] = None      # 'train' / 'test' (later)

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
    date_raw = groups["date"]
    date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    time = f"{groups['hour']}:{groups['minute']}:{groups['second']}"
    ts = f"{date}T{time}"
    return {
        "sku": groups["sku"],
        "tote_id": groups["tote_id"],
        "date": date,
        "time": time,
        "timestamp_iso": ts,
    }


def build_staging(dry_run: bool = False):
    """Build staging directory and manifest from quality-issue images.
    
    Args:
        dry_run: If True, only show what would be done without making changes
    """
    logger.info(f"Starting staging build (dry_run={dry_run})")
    
    if not dry_run:
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all PNG files
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
        
        if not dry_run:
            sku_dir.mkdir(parents=True, exist_ok=True)
            # kopiëren zodat originele data intact blijft
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copied += 1

        data = {
            "original_filepath": str(img_path),
            "staging_filepath": str(dest_path),
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
    logger.info(f"Skipped {skipped} files (invalid name or validation error)")
    if not dry_run:
        logger.info(f"Copied {copied} new files to staging")

    df_new = pd.DataFrame(rows)
    
    # Show stats
    logger.info("\n" + "="*50)
    logger.info("STATISTICS")
    logger.info("="*50)
    logger.info(f"Total images processed: {len(df_new)}")
    logger.info(f"Unique SKUs: {df_new['sku'].nunique()}")
    logger.info(f"Unique totes: {df_new['tote_id'].nunique()}")
    logger.info(f"Date range: {df_new['date'].min()} to {df_new['date'].max()}")
    logger.info("\nTop 10 SKUs by image count:")
    logger.info(f"\n{df_new['sku'].value_counts().head(10)}")
    logger.info("="*50 + "\n")

    if dry_run:
        logger.info("DRY RUN - No files written")
        return

    if MANIFEST_PATH.exists():
        df_old = pd.read_csv(MANIFEST_PATH)
        logger.info(f"Existing manifest has {len(df_old)} rows")
        # merge op staging_filepath om duplicaten te vermijden
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        original_len = len(df_merged)
        df_merged = df_merged.drop_duplicates(subset=["staging_filepath"])
        duplicates_removed = original_len - len(df_merged)
        df_merged.to_csv(MANIFEST_PATH, index=False)
        logger.info(f"Updated manifest: added {len(df_new)} new, removed {duplicates_removed} duplicates")
        logger.info(f"Total rows in manifest: {len(df_merged)}")
    else:
        df_new.to_csv(MANIFEST_PATH, index=False)
        logger.info(f"Created manifest.csv with {len(df_new)} rows")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build staging directory from quality-issue images")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    try:
        build_staging(dry_run=args.dry_run)
        logger.info("✓ Staging build completed successfully")
    except Exception as e:
        logger.error(f"✗ Error during staging build: {e}", exc_info=True)
        raise
