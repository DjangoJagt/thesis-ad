import asyncio
import base64
import csv
import logging
import os
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict

from picnic.ai.llm import AsyncLLMClient
from picnic.ai.messaging import user
from Sick_dataset.utils.crop import robust_industrial_crop

# --- PATH CONFIGURATION ---
# Assuming script is run from the root folder where .env is
ROOT_DIR = Path(__file__).resolve().parent
SICK_DIR = ROOT_DIR / "Sick_dataset" # Adjust if your folder structure differs
INPUT_CSV = SICK_DIR / "manifest.csv"
OUTPUT_CSV = SICK_DIR / "manifest_analyzed.csv"

# --- 1. LOAD .ENV (Crucial for Picnic Auth) ---
env_file = ROOT_DIR / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

# --- AI CONFIGURATION ---
MODEL_NAME = "openai/gpt-4.1" 
TEMPERATURE = 0.0
# Limit concurrent requests to avoid Rate Limits & Memory Crashes
MAX_CONCURRENT_REQUESTS = 5 

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# --- THE SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert Supply Chain Quality Control AI. Your task is to audit a stock tote image against a reference SKU image and classify the tote as either NORMAL or ANOMALOUS.

INPUT CONTEXT
- Image 1 is the REFERENCE SKU (The correct product: {product_name}).
- Image 2 is the TOTE to be audited.

CRITICAL OPERATING RULE
Operational Safety Priority:
- If you see a clear defect, classify as ANOMALOUS.

IMPORTANT PACKAGING RULE
If the product name specifies a quantity greater than 1 (e.g., "4 stuks", "6-pack", "2x"), 
then all individual units MUST be part of retail packaging in the tote.
ANY single loose unit outside retail packaging is ANOMALOUS.

--------------------
NORMAL CONDITIONS
--------------------
Classify as NORMAL ONLY if ALL visible items clearly match the reference SKU AND NO defect conditions are present.

The following are allowed and must be classified as NORMAL:
1. Quantity differences of the SAME product (from empty to full totes).
   - PROVIDED all units follow the Important Packaging Rule above.
   - An empty tote with no visible product is NORMAL.
2. Products that are rotated, flipped, or viewed from a different angle (e.g., Reference shows Side, Tote shows Top/Lid).
3. Product Naming Variations:
   - Different commercial, regional, or language-specific names for the SAME product are allowed,
     PROVIDED the product type, match visually.
   - If the product name contains "Merkloos" or "Generic", ANY visible brand name on the physical packaging is NORMAL, 
     PROVIDED the product type matches.
4. Harmless, packaging-related debris such as:
   - loose cardboard
   - cardboard multipack inserts or trays, even if placed vertically or sideways
   - separators
   - stickers
   - plastic air pillows
   - loose plastic caps  
   PROVIDED they do not contaminate the product.
5. Packaging Style vs Product Identity:
   - For single-unit SKUs sold by weight (e.g., "1 kilo"), the Reference Image may show the product loose.
     It is NORMAL for the Tote to contain the same product in simple retail packaging (e.g., nets or bags).
     This difference in packaging style is NOT a defect.
6. Packaging Label Updates: The label design or logo is slightly newer/different than the Reference, PROVIDED the Brand, Flavor, and Net Weight (if visible) match exactly.
7. Missing secondary outer lids (e.g., plastic overcaps), ONLY if the primary inner seal (e.g., aluminum foil) is fully intact.
10. Container Orientation & Base Visibility:
    - Many retail containers have opaque or semi-translucent plastic bases.
    - Seeing product color THROUGH a smooth, uniform plastic surface is NORMAL.
    - Classify as OPEN or BROKEN ONLY if the container interior is directly exposed
    (e.g., jagged edges, missing lid, uneven product surface, or leakage outside the container).

--------------------
ANOMALOUS CONDITIONS
--------------------
Classify as ANOMALOUS if ANY of the following are observed:

1. Wrong Product:
   - Packaging format does not match in a way that changes the product type (e.g., small cups vs large tub).
   - Unit size or Net Weight differs (e.g., 500g vs 1kg).
   - Different Flavor or Variant (e.g., Vegetarian "Beef" vs "Chicken").
   - Additional retail items present that do not belong to the SKU.

2. Open or Broken Packaging:
   - Ripped, torn, or open packaging.
   - Missing primary lid or broken inner foil seal.
     PROVIDED the container interior is visibly exposed or product is escaping the container.
   - Broken Multipacks: Items that are clearly intended to be shrink-wrapped or grouped (e.g., yogurt 4-pack) are found loose or scattered.
   - If all items in the tote are consistently contained in retail packaging as intact units (e.g., multipacks, nets), a single loose item is ANOMALOUS

4. Severe Physical Damage:
   - Heavily crushed, dented, or deformed items.

5. Freshness Issues:
   - Visible signs of spoilage such as mold, rot, discoloration, or browning.

6. Contamination:
   - Dirt, sludge, stains, or residues ON THE PRODUCT itself.

--------------------
OUTPUT FORMAT
--------------------
Return ONLY a valid JSON object with no markdown formatting:
{{
  "status": "NORMAL" or "ANOMALOUS",
  "reason": "N/A" (if Normal) or "Short specific description of the defect"
}}
"""

  
# def get_image_data(image_path: Path):
#     """Reads image and returns (base64_string, mime_type) by sniffing headers."""
    
#     with open(image_path, "rb") as f:
#         header = f.read(12)
#         f.seek(0)
#         file_bytes = f.read()

#     # Detect true image type via magic bytes
#     if header.startswith(b'\xff\xd8'):
#         mime_type = "image/jpeg"
#     elif header.startswith(b'\x89PNG\r\n\x1a\n'):
#         mime_type = "image/png"
#     elif header.startswith(b'RIFF') and b'WEBP' in header:
#         mime_type = "image/webp"
#     else:
#         # Conservative fallback
#         ext = image_path.suffix.lower()
#         if ext in [".jpg", ".jpeg", ".jfif"]:
#             mime_type = "image/jpeg"
#         elif ext == ".png":
#             mime_type = "image/png"
#         elif ext == ".webp":
#             mime_type = "image/webp"
#         else:
#             mime_type = "image/jpeg"  # safest default

#     b64_str = base64.b64encode(file_bytes).decode("utf-8")
#     return b64_str, mime_type

def get_image_data(
    image_path: Path,
    do_crop: bool = False,
    encode_format: str = ".png",  # IMPORTANT
):
    """
    Loads image with OpenCV, optionally crops it, encodes in-memory,
    and returns (base64_string, mime_type).
    """

    # --- 1. Load ---
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # --- 2. Optional Crop ---
    if do_crop:
        try:
            img, _ = robust_industrial_crop(img, debug_mode=False)
        except Exception as e:
            LOGGER.warning(
                f"Cropping failed for {image_path.name}, using full image. Error: {e}"
            )

    # --- 3. Encode (LOSSLESS) ---
    success, buffer = cv2.imencode(encode_format, img)
    if not success:
        raise ValueError("Could not encode image buffer")

    file_bytes = buffer.tobytes()
    b64_str = base64.b64encode(file_bytes).decode("utf-8")

    # --- 4. MIME ---
    if encode_format == ".png":
        mime_type = "image/png"
    elif encode_format in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    else:
        raise ValueError(f"Unsupported encode format: {encode_format}")

    return b64_str, mime_type


async def analyze_row(
    llm_client: AsyncLLMClient, 
    row: Dict[str, str], 
    row_idx: int, 
    semaphore: asyncio.Semaphore,
    crop: bool = False
) -> Dict[str, str]:
    """Process a single row with rate limiting."""
    
    async with semaphore: # Wait here if 5 requests are already running
        
        # 1. MAP CORRECT COLUMNS FROM MANIFEST
        # We prepend SICK_DIR because paths in manifest are relative (e.g. "template_images/123.png")
        sku_rel = row.get("sku_template_path", "")
        tote_rel = row.get("staging_filepath", "")
        product_name = row.get("product_name", "product")

        if not sku_rel or not tote_rel:
            row["ai_status"] = "SKIPPED"
            row["ai_reason"] = "Missing paths in CSV"
            return row

        sku_path = SICK_DIR / sku_rel
        tote_path = SICK_DIR / tote_rel

        # 2. VALIDATE FILES EXIST
        if not sku_path.exists():
            row["ai_status"] = "SKIPPED_NO_REF" # Specific error for missing PIM photo
            row["ai_reason"] = f"Reference image not found: {sku_rel}"
            # LOGGER.warning(f"Row {row_idx}: Missing reference {sku_rel}")
            return row
            
        if not tote_path.exists():
            row["ai_status"] = "ERROR"
            row["ai_reason"] = f"Tote image not found: {tote_rel}"
            return row

        try:
            # 3. PREPARE PAYLOAD (Updated for dynamic MIME types)
            sku_b64, sku_mime = get_image_data(sku_path, do_crop=False)    
            tote_b64, tote_mime = get_image_data(tote_path, do_crop=crop, encode_format=".png")

            # 1. PREPARE THE PROMPT WITH CONTEXT
            # We inject the specific product name into the general rules
            filled_prompt = SYSTEM_PROMPT.format(product_name=product_name)

            messages = [
                user([
                    {"type": "text", "text": filled_prompt},
                    {"type": "text", "text": "Image 1: Reference SKU"},
                    # Use the dynamic mime variable here (sku_mime) instead of hardcoded 'image/png'
                    {"type": "image_url", "image_url": {"url": f"data:{sku_mime};base64,{sku_b64}"}}, 
                    {"type": "text", "text": "Image 2: Tote to Audit"},
                    # Use the dynamic mime variable here (tote_mime)
                    {"type": "image_url", "image_url": {"url": f"data:{tote_mime};base64,{tote_b64}"}} 
                ])
            ]

            # 4. CALL AI
            # LOGGER.info(f"Row {row_idx}: Analyzing {product_name}...")
            response = await llm_client.complete(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE
            )

            # 5. PARSE RESULT
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_resp)
            
            row["ai_status"] = result.get("status", "UNKNOWN")
            row["ai_reason"] = result.get("reason", "Unknown")
            
            LOGGER.info(f"[{row_idx}] {product_name[:20]}: {row['ai_status']} ({row['ai_reason']})")

        except Exception as e:
            LOGGER.error(f"Row {row_idx} Failed: {e}")
            row["ai_status"] = "ERROR"
            row["ai_reason"] = str(e)

        return row

async def main():
    parser = argparse.ArgumentParser(description="Run Visual QA on Staging Data")
    parser.add_argument("--limit", type=int, help="Only run X images (good for testing)")
    parser.add_argument("--filter-sku", type=str, help="Run only for this specific SKU")
    parser.add_argument("--filter-product", type=str, help="Run only for products containing this text (e.g., 'Banana')")
    parser.add_argument("--crop", action="store_true", help="Enable robust cropping for tote images")
    args = parser.parse_args()

    LOGGER.info("Starting Defect Detection...")
    
    if not INPUT_CSV.exists():
        LOGGER.error(f"Manifest not found at {INPUT_CSV}")
        return

    llm_client = AsyncLLMClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Read CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # --- FILTERING LOGIC ---
    tasks = []
    filtered_rows = []
    
    for idx, row in enumerate(all_rows, start=1):
        # Filter by SKU
        if args.filter_sku and row.get("sku") != args.filter_sku:
            continue
        
        # Filter by Product Name
        if args.filter_product:
            p_name = row.get("product_name", "").lower()
            if args.filter_product.lower() not in p_name:
                continue

        filtered_rows.append((idx, row))

    # Apply Limit
    if args.limit:
        filtered_rows = filtered_rows[:args.limit]

    LOGGER.info(f"Processing {len(filtered_rows)} images (Total in manifest: {len(all_rows)})")

    # Create Tasks
    for idx, row in filtered_rows:
        tasks.append(analyze_row(llm_client, row, idx, semaphore, crop=args.crop))

    if not tasks:
        LOGGER.warning("No rows matched your filters.")
        return

    # Run Analysis
    results = await asyncio.gather(*tasks)

    # Write Output
    # We write a NEW file with just the processed results
    output_fields = fieldnames + ["ai_status", "ai_reason"]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)
    
    LOGGER.info(f"Finished! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())