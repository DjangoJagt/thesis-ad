#!/usr/bin/env python
"""
Script to enrich manifest.csv with DWH data:
1. Expected quantity from stock events
2. Quality issues (customer complaints) 
3. Shopper flags (internal issues)
4. Product names
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import os

import pandas as pd

try:
    from picnic.tools import config_loader
    from picnic.database.database_client import DatabaseClientFactory
    PICNIC_AVAILABLE = True
except ImportError:
    PICNIC_AVAILABLE = False

# Load .env file if it exists
from pathlib import Path as EnvPath
env_file = EnvPath(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SICK_DIR = Path(__file__).resolve().parent.parent
SQL_DIR = SICK_DIR / "sql"
MANIFEST_PATH = SICK_DIR / "manifest.csv"
CACHE_DIR = SICK_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_manifest() -> pd.DataFrame:
    """Load the manifest CSV."""
    df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded manifest with {len(df)} rows")
    return df


def save_manifest(df: pd.DataFrame) -> None:
    """Save the enriched manifest."""
    # Create backup first
    backup_path = MANIFEST_PATH.with_suffix('.csv.backup')
    if MANIFEST_PATH.exists():
        import shutil
        shutil.copy2(MANIFEST_PATH, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    df.to_csv(MANIFEST_PATH, index=False)
    logger.info(f"Saved enriched manifest to {MANIFEST_PATH}")


def get_dwh_client():
    """Initialize DWH client."""
    if not PICNIC_AVAILABLE:
        raise ImportError("Picnic modules not available. Install them or use --no-dwh flag.")
    
    config = config_loader.load_config(config_dir=Path("config"))
    dwh = DatabaseClientFactory.from_config(config).get_client()
    logger.info("✓ DWH client initialized")
    return dwh


def get_quality_issues(dwh_client, start_date: datetime, 
                       cache: bool = True) -> pd.DataFrame:
    """
    Get quality issues (customer complaints) per tote/SKU.
    Returns DataFrame with: stock_tote_barcode, article_id, total_quality_issues
    """
    cache_file = CACHE_DIR / "quality_issues.parquet"
    
    if cache and cache_file.exists():
        logger.info(f"Loading quality issues from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    logger.info("Fetching quality issues from DWH...")
    
    sql_path = SQL_DIR / "quality_issues.sql"
    if not sql_path.exists():
        logger.warning(f"SQL file not found: {sql_path}. Creating template.")
        create_quality_issues_sql(sql_path)
        return pd.DataFrame(columns=['stock_tote_barcode', 'article_id', 'total_quality_issues'])
    
    result = dwh_client.select(
        query_path=sql_path,
        start_date=start_date.strftime("%Y-%m-%d %H:%M:%S")
    )
    df = pd.DataFrame(result.as_dicts())
    
    if cache and len(df) > 0:
        df.to_parquet(cache_file)
        logger.info(f"Cached {len(df)} quality issue records")
    
    return df


# OUDE VERSIE:
# def get_shopper_issues(dwh_client, start_date: datetime, end_date: datetime, cache: bool = True) -> pd.DataFrame:

# NIEUWE VERSIE (Zonder end_date):
def get_shopper_issues(dwh_client, start_date: datetime, cache: bool = True) -> pd.DataFrame:
    """
    Get shopper-detected issues per tote/SKU.
    Returns DataFrame with: stock_tote_barcode, article_id, issue_type, number_of_issues, last_decant_event_timestamp
    """
    cache_file = CACHE_DIR / "shopper_issues.parquet"
    
    if cache and cache_file.exists():
        logger.info(f"Loading shopper issues from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    logger.info("Fetching shopper issues from DWH...")
    
    sql_path = SQL_DIR / "shopper_issues.sql"
    if not sql_path.exists():
        create_shopper_issues_sql(sql_path) # Zorg dat deze functie ook is geupdate!
        return pd.DataFrame(columns=['stock_tote_barcode', 'article_id', 'issue_type', 'number_of_issues', 'last_decant_event_timestamp'])
    
    result = dwh_client.select(
        query_path=sql_path,
        start_date=start_date.strftime("%Y-%m-%d %H:%M:%S")
        # end_date parameter verwijderd
    )
    df = pd.DataFrame(result.as_dicts())
    
    if cache and len(df) > 0:
        df.to_parquet(cache_file)
        logger.info(f"Cached {len(df)} shopper issue records")
    
    return df

def get_article_info(dwh_client, article_ids: List[str], cache: bool = True) -> pd.DataFrame:
    """
    Get product names and categories for a list of article IDs.
    Uses chunking to avoid SQL limits with large lists.
    """
    cache_file = CACHE_DIR / "article_info.parquet"
    
    # 1. Load Cache if available
    cached_df = pd.DataFrame()
    if cache and cache_file.exists():
        cached_df = pd.read_parquet(cache_file)
        cached_df['article_id'] = cached_df['article_id'].astype(str)
    
    # 2. Determine which IDs are missing
    unique_requested = set(str(x) for x in article_ids if pd.notna(x))
    
    if not cached_df.empty:
        available_ids = set(cached_df['article_id'].unique())
        missing_ids = list(unique_requested - available_ids)
    else:
        missing_ids = list(unique_requested)
    
    # 3. Fetch missing IDs from DWH
    if missing_ids:
        logger.info(f"Fetching article info for {len(missing_ids)} new SKUs...")
        
        chunk_size = 1000
        new_results = []
        
        for i in range(0, len(missing_ids), chunk_size):
            chunk = missing_ids[i:i + chunk_size]
            formatted_ids = ",".join([f"'{aid}'" for aid in chunk])
            
            query = f"""
            SELECT 
                article_id::STRING as article_id,
                art_supply_chain_name as product_name,
                art_p_cat_lev_1 as category
            FROM dim.dm_article
            WHERE article_id IN ({formatted_ids})
            """
            
            try:
                # FIX: Verander 'query_string' naar 'query'
                result = dwh_client.select(query=query)
                chunk_df = pd.DataFrame(result.as_dicts())
                new_results.append(chunk_df)
            except Exception as e:
                logger.error(f"Error fetching chunk {i}-{i+chunk_size}: {e}")

        # 4. Combine results
        if new_results:
            new_df = pd.concat(new_results, ignore_index=True)
            
            if not cached_df.empty:
                final_df = pd.concat([cached_df, new_df], ignore_index=True).drop_duplicates(subset=['article_id'])
            else:
                final_df = new_df
            
            if cache:
                final_df.to_parquet(cache_file)
                logger.info(f"Updated cache with {len(new_df)} new records")
                
            cached_df = final_df
    else:
        logger.info("All article info found in cache")

    # Filter to return only what was requested
    if not cached_df.empty:
        return cached_df[cached_df['article_id'].isin(unique_requested)].copy()
    else:
        return pd.DataFrame(columns=['article_id', 'product_name', 'category'])


# def get_product_names(dwh_client, article_ids: List[str], cache: bool = True) -> Dict[str, str]:
#     """
#     Get product names for article IDs.
#     Returns dict: {article_id: product_name}
#     """
#     cache_file = CACHE_DIR / "product_names.json"
    
#     if cache and cache_file.exists():
#         logger.info(f"Loading product names from cache: {cache_file}")
#         with open(cache_file, 'r') as f:
#             return json.load(f)
    
#     logger.info(f"Fetching product names for {len(article_ids)} unique SKUs...")
    
#     # Simple query to get product names
#     query = """
#     SELECT 
#         article_id,
#         art_supply_chain_name as product_name
#     FROM PICNIC_NL_PROD.DIM.DM_ARTICLE
#     WHERE article_id IN ({})
#     """.format(','.join(f"'{aid}'" for aid in article_ids))
    
#     result = dwh_client.execute(query)
#     df = pd.DataFrame(result.as_dicts())
    
#     product_dict = dict(zip(df['article_id'].astype(str), df['product_name']))
    
#     if cache:
#         with open(cache_file, 'w') as f:
#             json.dump(product_dict, f, indent=2)
#         logger.info(f"Cached {len(product_dict)} product names")
    
#     return product_dict


def enrich_manifest(df: pd.DataFrame, dwh_client, use_dwh: bool = True, 
                    use_cache: bool = True) -> pd.DataFrame:
    """
    Main enrichment function with Time-Based matching logic.
    """
    # 1. Prepare Manifest Dates
    if 'timestamp_iso' in df.columns:
        df['photo_datetime'] = pd.to_datetime(df['timestamp_iso'])
    elif 'timestamp' in df.columns:
        df['photo_datetime'] = pd.to_datetime(df['timestamp'])
    else:
        logger.warning("No timestamp column found. Using 'date' (midnight). Matches will likely fail!")
        df['photo_datetime'] = pd.to_datetime(df['date'])

    min_date = df['photo_datetime'].min()
    start_date = min_date - timedelta(days=30) 
    
    logger.info(f"Querying DWH starting from: {start_date.date()}")
    
    # Standardize IDs
    df['tote_barcode'] = df['tote_id'].astype(str)
    df['article_id'] = df['sku'].astype(str)
    df['row_id'] = range(len(df))
    
    # ---------------------------------------------------------
    # 0. CLEAN MANIFEST (De Grote Schoonmaak)
    # ---------------------------------------------------------
    # We verwijderen ALLE kolommen die uit de DWH komen om duplicaten (_x, _y, _dwh) te voorkomen.
    cols_to_reset = [
        # Flags & Totalen
        'total_quality_issues', 'complaint_flag', 
        'shopper_flag', 'number_of_issues',
        
        # Specifieke Quality Metrics (Labels voor je ML model)
        'mix_up_qty', 
        'damaged_items_qty', 
        'dirty_items_qty', 
        'freshness_issue_items_qty', 
        'underripe_items_qty', 
        'overripe_items_qty', 
        'spoiled_items_qty',
        
        # Mogelijke vervuiling van vorige runs
        'mix_up_qty_dwh', 'damaged_items_qty_dwh', 'dirty_items_qty_dwh',
        'freshness_issue_items_qty_dwh', 'underripe_items_qty_dwh', 
        'overripe_items_qty_dwh', 'spoiled_items_qty_dwh',
        'total_quality_issues_dwh', 'number_of_issues_shopper'
    ]
    # Verwijder ze als ze bestaan
    df.drop(columns=[c for c in cols_to_reset if c in df.columns], inplace=True)

    if use_dwh:
        # A. Get Product Info (Name + Category)
        logger.info("\n" + "="*50)
        logger.info("Step 0: Fetching product info")
        logger.info("="*50)
        
        unique_skus = df['article_id'].unique().tolist()
        article_df = get_article_info(dwh_client, unique_skus, cache=use_cache)
        
        if len(article_df) > 0:
            # Clean up columns if they exist from previous runs
            cols_to_drop = ['product_name', 'category']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            
            # Merge
            df = df.merge(
                article_df,
                on='article_id',
                how='left'
            )
            logger.info(f"✓ Added product info for {df['product_name'].notna().sum()} rows")
        else:
            logger.warning("No article info found")

        # B. Quality Issues
        logger.info("\n" + "="*50)
        logger.info("Step 1: Quality Issues (Customer)")
        logger.info("="*50)
        
        quality_df = get_quality_issues(dwh_client, start_date, cache=use_cache)
        quality_df.columns = quality_df.columns.str.lower()
        
        if len(quality_df) > 0:
            df = merge_issues_time_based(df, quality_df, issue_col='total_quality_issues', flag_col='complaint_flag')
        else:
            df['complaint_flag'] = False
            df['total_quality_issues'] = 0
            # Vul ook de detailkolommen met 0 (voor consistentie)
            for col in ['mix_up_qty', 'damaged_items_qty', 'dirty_items_qty', 'freshness_issue_items_qty', 
                        'underripe_items_qty', 'overripe_items_qty', 'spoiled_items_qty']:
                df[col] = 0

        # C. Shopper Issues
        logger.info("\n" + "="*50)
        logger.info("Step 2: Shopper Issues (Internal)")
        logger.info("="*50)
        
        shopper_df = get_shopper_issues(dwh_client, start_date, cache=use_cache)
        shopper_df.columns = shopper_df.columns.str.lower()
        
        if len(shopper_df) > 0:
            df = merge_issues_time_based(df, shopper_df, issue_col='number_of_issues', flag_col='shopper_flag')
        else:
            df['shopper_flag'] = False
            df['number_of_issues'] = 0

    else:
        logger.warning("DWH disabled")
        # Zet alles op 0/False
        for col in ['complaint_flag', 'shopper_flag', 'total_quality_issues', 'number_of_issues']:
            df[col] = 0 if 'flag' not in col else False
    
    # Final Cleanup
    drop_cols = ['tote_barcode', 'article_id', 'row_id', 'photo_datetime']
    df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)
    
    return df

# --- HELPER FUNCTIE OM CODE DUPLICATIE TE VOORKOMEN ---
def merge_issues_time_based(manifest_df, issues_df, issue_col, flag_col):
    """
    Helper function to merge issues (Quality or Shopper) based on Tote + SKU + Time.
    """
    # 1. Prepare Issue Data
    issues_df['stock_tote_barcode'] = issues_df['stock_tote_barcode'].astype(str)
    issues_df['article_id'] = issues_df['article_id'].astype(str)
    issues_df['last_decant_event_timestamp'] = pd.to_datetime(issues_df['last_decant_event_timestamp'])
    
    logger.info(f"Fetched {len(issues_df)} issue records")

    # 2. Merge
    merged = manifest_df.merge(
        issues_df,
        left_on=['tote_barcode', 'article_id'],
        right_on=['stock_tote_barcode', 'article_id'],
        how='left',
        suffixes=('', '_dwh')
    )
    
    # 3. Filter: Photo time >= Session End time
    valid_match_mask = (
        merged['last_decant_event_timestamp'].isna() | 
        (merged['photo_datetime'] >= merged['last_decant_event_timestamp'])
    )
    merged = merged[valid_match_mask]
    
    # 4. Pick nearest session
    merged = merged.sort_values('last_decant_event_timestamp', ascending=False)
    df_final = merged.drop_duplicates(subset=['row_id'], keep='first')
    
    # 5. Restore order & Set Flags
    df_result = df_final.sort_values('row_id').reset_index(drop=True)
    
    # Fill NaNs with 0 for the issue count column
    df_result[issue_col] = df_result[issue_col].fillna(0).astype(int)
    df_result[flag_col] = (df_result[issue_col] > 0)
    
    # Clean DWH columns specific to this merge
    cols_to_drop = ['stock_tote_barcode', 'last_decant_event_timestamp', 'article_id_dwh', 'issue_type']
    df_result.drop(columns=[c for c in cols_to_drop if c in df_result.columns], inplace=True)

    logger.info(f"✓ Matches found. Totes with {flag_col}: {df_result[flag_col].sum()}")
    
    return df_result


def create_quality_issues_sql(path: Path):
    """Create template SQL for quality issues query."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sql = """-- Quality Issues Query
-- Paste your quality issues query here
-- Must return: stock_tote_barcode, article_id, total_quality_issues
-- Use %(start_date)s and %(end_date)s parameters

SELECT 
    'TODO' as stock_tote_barcode,
    'TODO' as article_id,
    0 as total_quality_issues
WHERE 1=0
"""
    path.write_text(sql)
    logger.info(f"Created template SQL: {path}")


def create_shopper_issues_sql(path: Path):
    """Create template SQL for shopper issues query."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Updated template to match the new ROBUST query structure
    sql = """-- Shopper Issues Query (Photo Matching)
-- Returns: stock_tote_barcode, article_id, last_decant_event_timestamp, issue_type, number_of_issues
-- Parameter: %(start_date)s

SELECT 
    'TODO' as stock_tote_barcode,
    'TODO' as article_id,
    CURRENT_TIMESTAMP as last_decant_event_timestamp,
    'TODO' as issue_type,
    0 as number_of_issues
WHERE 1=0
"""
    path.write_text(sql)
    logger.info(f"Created template SQL: {path}")


def print_statistics(df: pd.DataFrame):
    """Print enrichment statistics."""
    logger.info("\n" + "="*60)
    logger.info("ENRICHMENT STATISTICS")
    logger.info("="*60)
    
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Product names filled: {df['product_name'].notna().sum()} ({df['product_name'].notna().sum()/len(df)*100:.1f}%)")
    logger.info(f"Totes with quality complaints: {df['complaint_flag'].sum()} ({df['complaint_flag'].sum()/len(df)*100:.1f}%)")
    logger.info(f"Totes with shopper issues: {df['shopper_flag'].sum()} ({df['shopper_flag'].sum()/len(df)*100:.1f}%)")
    logger.info(f"Totes with ANY issue: {(df['complaint_flag'] | df['shopper_flag']).sum()} ({(df['complaint_flag'] | df['shopper_flag']).sum()/len(df)*100:.1f}%)")
    
    logger.info("\nTop 10 SKUs by image count:")
    logger.info(df.groupby(['sku', 'product_name']).size().sort_values(ascending=False).head(10))
    
    logger.info("\nIssue breakdown:")
    logger.info(f"  Both flags True: {(df['complaint_flag'] & df['shopper_flag']).sum()}")
    logger.info(f"  Only complaint_flag: {(df['complaint_flag'] & ~df['shopper_flag']).sum()}")
    logger.info(f"  Only shopper_flag: {(~df['complaint_flag'] & df['shopper_flag']).sum()}")
    logger.info(f"  No issues: {(~df['complaint_flag'] & ~df['shopper_flag']).sum()}")
    
    logger.info("="*60 + "\n")


def main(use_dwh: bool = True, use_cache: bool = True, dry_run: bool = False):
    """Main execution."""
    logger.info("Starting manifest enrichment...")
    logger.info(f"DWH: {use_dwh}, Cache: {use_cache}, Dry-run: {dry_run}")
    
    # Load manifest
    df = load_manifest()
    
    # Get DWH client if needed
    dwh_client = None
    if use_dwh:
        try:
            dwh_client = get_dwh_client()
        except Exception as e:
            logger.error(f"Failed to initialize DWH client: {e}")
            logger.warning("Continuing without DWH access...")
            use_dwh = False
    
    # Enrich
    df_enriched = enrich_manifest(df, dwh_client, use_dwh=use_dwh, use_cache=use_cache)
    
    # Statistics
    print_statistics(df_enriched)
    
    # Save
    if not dry_run:
        save_manifest(df_enriched)
        logger.info("✓ Enrichment completed successfully")
    else:
        logger.info("DRY RUN - No changes saved")
        logger.info(f"Would save {len(df_enriched)} rows with {len(df_enriched.columns)} columns")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich manifest with DWH data")
    parser.add_argument("--no-dwh", action="store_true", help="Skip DWH queries (use cached data only)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh DWH queries")
    parser.add_argument("--dry-run", action="store_true", help="Don't save changes")
    
    args = parser.parse_args()
    
    try:
        main(
            use_dwh=not args.no_dwh,
            use_cache=not args.no_cache,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"✗ Enrichment failed: {e}", exc_info=True)
        raise
