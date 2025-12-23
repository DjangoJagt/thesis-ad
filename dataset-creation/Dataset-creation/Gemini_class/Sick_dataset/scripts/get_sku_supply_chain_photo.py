"""Get published view from PIM Cache."""

import asyncio
import json
import os
from pathlib import Path
import logging

import httpx
import pandas as pd
from picnic.client import ClientFactory
from picnic.tools.config_loader import load_config
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPTS_DIR.parent  # Sick_dataset/
CONFIG_DIR = DATASET_DIR.parent / "config" # Gemini_class/config/

# Define file locations
MANIFEST_FILE_LOCATION = DATASET_DIR / "manifest.csv"
SUPPLY_CHAIN_PHOTO_LOCATION = DATASET_DIR / "template_images"

env_file = DATASET_DIR.parent / ".env"
if env_file.exists():
    print(f"Loading environment from {env_file}")
    with open(env_file) as f:
        for line in f:
            # Simple parser for KEY=VALUE lines
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()
else:
    print(f"Warning: No .env file found at {env_file}")

# Load Config
CONFIG = load_config(config_dir=CONFIG_DIR)

logging.getLogger("httpx").setLevel(logging.WARNING)


async def fetch_supply_chain_photo(config: dict, article_image_counter: dict[int, int]):
    """
    Fetch supply chain photos for articles based on the PIM cache data.

    Args:
        config: Configuration dictionary.
        article_image_counter: Dictionary tracking image download status per article ID.
    """
    async_client = ClientFactory.from_config(config).get_async()
    
    # 1. Get PIM Data
    file_name = await get_pim_cache_data(async_client)
    with open(file_name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 2. Build Lookup Map
    pim_map = {
        int(item["article_id"]): item["supply_chain_photo_id"] 
        for item in data 
        if "supply_chain_photo_id" in item and item["supply_chain_photo_id"]
    }

    # 3. Filter work to be done
    articles_to_download = [aid for aid, status in article_image_counter.items() if status == 0]
    
    # --- FIX: Calculate skipped files correctly ---
    total_requested = len(article_image_counter)
    already_saved_counter = total_requested - len(articles_to_download)
    
    print(f"Total requested: {total_requested}")
    print(f"✅ Already have: {already_saved_counter}")
    print(f"⬇️ Downloading: {len(articles_to_download)}")

    # 4. Download Loop
    async with httpx.AsyncClient() as dl_client:
        for article_id in tqdm(articles_to_download, desc="Downloading images"):
            
            supply_chain_photo_id = pim_map.get(article_id)
            
            if not supply_chain_photo_id:
                # Optional: print(f"Warning: SKU {article_id} has no photo in PIM")
                continue

            url = f"https://picnic-nl-prod-images.s3.eu-west-1.amazonaws.com/{supply_chain_photo_id}/large.png"
            image_path = SUPPLY_CHAIN_PHOTO_LOCATION / f"{article_id}.png"

            try:
                response = await dl_client.get(url, timeout=10.0)
                if response.status_code == 200:
                    with open(image_path, "wb") as file:
                        file.write(response.content)
                    article_image_counter[article_id] = 1
                else:
                    pass # Silent fail
            except Exception as e:
                print(f"Error requesting {article_id}: {str(e)}")

    # Final Summary
    current_files = {int(f.stem) for f in SUPPLY_CHAIN_PHOTO_LOCATION.glob("*.png") if f.stem.isdigit()}
    # Check explicitly against the folder content to be sure
    actually_missing = [aid for aid in article_image_counter if aid not in current_files]
    
    if actually_missing:
        print(f"⚠️ Done, but {len(actually_missing)} photos are still missing (likely not in PIM).")
    else:
        print("🎉 All photos successfully downloaded!")


async def get_pim_cache_data(async_client, data_version: int = 52230):
    """
    Retrieve and save PIM cache data asynchronously.

    Args:
        async_client: An asynchronous HTTP client.
        data_version: The version of PIM cache data to download.

    Returns:
        str: Filename where the PIM cache data is saved.
    """
    # Save cache in the 'cache' folder to keep things clean
    cache_dir = DATASET_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    file_name = cache_dir / f"pim_cache_version_{data_version}.json"

    # If it exists, skip download
    if file_name.exists():
        print(f"Using cached PIM file: {file_name}")
        return file_name

    print(f"Downloading PIM cache v{data_version}...")
    response = await async_client.get(
        f"https://pim-cache-prod.nl.picnicinternational.com/api/1/views/etl_article?version={data_version}",
        follow_redirects=True,
    )

    if response.status_code == 200:
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(response.json(), json_file, ensure_ascii=False, indent=4)
    return file_name


def initialize_directory(path: Path):
    """Ensure directory exists for storing photos."""
    os.makedirs(path, exist_ok=True)

def load_article_ids_from_manifest(file_location: Path) -> set:
    """Load the unique SKUs from manifest.csv."""
    if not file_location.exists():
        print(f"Error: Manifest not found at {file_location}")
        return set()
        
    df = pd.read_csv(file_location)
    # Ensure SKUs are integers
    article_ids = set(df["sku"].astype(int).unique().tolist())
    print(f"🔢 Loaded {len(article_ids)} unique SKUs from {file_location.name}")
    return article_ids

def update_article_image_counter(path: Path, counter: dict[int, int]):
    """Update the article supply chain image counter with existing image files."""
    if not path.exists():
        return
    for file_name in os.listdir(path):
        if file_name.endswith(".png"):
            try:
                article_id = int(file_name.split(".")[0])
                if article_id in counter:
                    counter[article_id] = 1
            except ValueError:
                pass

def get_template_images():
    initialize_directory(SUPPLY_CHAIN_PHOTO_LOCATION)
    # article_ids = load_article_ids(PRODUCTS_FILE_LOCATION)
    article_ids = load_article_ids_from_manifest(MANIFEST_FILE_LOCATION)
    article_image_counter = {article_id: 0 for article_id in article_ids}
    update_article_image_counter(SUPPLY_CHAIN_PHOTO_LOCATION, article_image_counter)
    asyncio.run(fetch_supply_chain_photo(CONFIG, article_image_counter))


if __name__ == "__main__":
    get_template_images()