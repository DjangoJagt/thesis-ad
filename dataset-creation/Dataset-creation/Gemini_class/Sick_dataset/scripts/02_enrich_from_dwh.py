#!/usr/bin/env python
"""
Script to enrich manifest.csv with DWH data.
FOCUS: Only adds Product Names (Context) for AI analysis.
"""
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

# --- CONFIGURATION ---
try:
    from picnic.tools import config_loader
    from picnic.database.database_client import DatabaseClientFactory
    PICNIC_AVAILABLE = True
except ImportError:
    PICNIC_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SICK_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SICK_DIR.parent / "config"
MANIFEST_PATH = SICK_DIR / "manifest.csv"
CACHE_DIR = SICK_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Load .env variables
env_file = SICK_DIR.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()


def get_dwh_client():
    """Initialize DWH client."""
    if not PICNIC_AVAILABLE:
        raise ImportError("Picnic modules not available. Use --no-dwh.")
    config = config_loader.load_config(config_dir=CONFIG_DIR)
    return DatabaseClientFactory.from_config(config).get_client()


def get_article_info(dwh_client, article_ids: List[str], cache: bool = True) -> pd.DataFrame:
    """Fetch Product Name and Category."""
    cache_file = CACHE_DIR / "article_info.parquet"
    
    # Load Cache
    if cache and cache_file.exists():
        cached_df = pd.read_parquet(cache_file)
        # Filter for what we need
        needed = set(article_ids)
        available = set(cached_df['article_id'].unique())
        if needed.issubset(available):
            return cached_df[cached_df['article_id'].isin(needed)]

    # Fetch from DWH if missing
    logger.info(f"Fetching info for {len(article_ids)} SKUs...")
    formatted_ids = ",".join([f"'{aid}'" for aid in article_ids])
    
    query = f"""
    SELECT DISTINCT
        article_id::STRING as article_id,
        art_supply_chain_name as product_name,
        art_p_cat_lev_1 as category
    FROM dim.dm_article
    WHERE article_id IN ({formatted_ids})
    """
    
    result = dwh_client.select(query=query)
    df = pd.DataFrame(result.as_dicts())
    
    if cache:
        df.to_parquet(cache_file)
        
    return df


def enrich_manifest(df: pd.DataFrame, dwh_client, use_dwh: bool = True) -> pd.DataFrame:
    """Main enrichment logic."""
    if not use_dwh:
        logger.warning("Skipping DWH enrichment.")
        return df

    # Standardize types
    df['article_id'] = df['sku'].astype(str)
    
    # 1. ADD PRODUCT NAMES (Context for AI)
    unique_skus = df['article_id'].unique().tolist()
    product_info = get_article_info(dwh_client, unique_skus)
    
    # Merge Product Info
    # Drop existing if re-running
    if 'product_name' in df.columns: 
        df.drop(columns=['product_name', 'category'], inplace=True, errors='ignore')
        
    df = df.merge(product_info, on='article_id', how='left')
    
    # Cleanup of columns we don't need for the AI script
    if 'article_id' in df.columns:
        df.drop(columns=['article_id'], inplace=True)
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load
    if not MANIFEST_PATH.exists():
        logger.error("Manifest not found!")
        return
    df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded {len(df)} rows.")

    # Enrich
    try:
        dwh = get_dwh_client()
        df_enriched = enrich_manifest(df, dwh, use_dwh=True)
    except Exception as e:
        logger.error(f"DWH Error: {e}")
        return

    # Stats
    logger.info(f"Enriched: {df_enriched['product_name'].notna().sum()} Product Names")

    # Save
    if not args.dry_run:
        df_enriched.to_csv(MANIFEST_PATH, index=False)
        logger.info("Saved to manifest.csv")

if __name__ == "__main__":
    main()