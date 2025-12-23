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

You must execute the audit in a strict, 4-Stage Evaluation Pipeline. Do not skip stages.

INPUT CONTEXT
- Image 1: REFERENCE SKU (Target Product: {product_name}).
- Image 2: TOTE (The main container to be audited).

--------------------------------------------------------
STAGE 1: LEAKAGE & CRITICAL SAFETY CHECK (Highest Priority)
--------------------------------------------------------
First, ignore the product identity and inspect the tote floor, corners, and gaps in Image 2.
**Rule:** If any leakage is detected, immediate fail.

1.  **Leakage Indicators:** Look for pooling liquid, streaks, wet patches, white/milky spills, or fresh viscous substances.
2.  **Ambiguity Rule:** If you see a suspicious dark spot or white patch and cannot definitively prove it is a harmless artifact (like a shadow), classify as ANOMALOUS. When in doubt regarding wetness, err on the side of safety.
3.  **Outcome:** If wetness is found, STOP and classify as ANOMALOUS (Reason: Leakage). If dry, proceed to Stage 2.

--------------------------------------------------------
STAGE 2: PRODUCT IDENTITY & MATCHING
--------------------------------------------------------
Compare Image 1 (Reference) with Image 2 (Tote). Verify the product matches.

1.  **Identity Matching:**
    - Visual Match: The product type, brand, flavor, and net weight (e.g., 500g vs 1kg) must match.
    - Allowed Variations:
        - Different regional/commercial names for the same visual product.
        - "Merkloos/Generic" names allow ANY brand on packaging if the product type matches.
        - Label updates (newer logo/design) are NORMAL if Brand/Flavor/Weight match.
        - Single-unit items sold by weight (e.g., "1 kilo") may appear loose in Reference but bagged in Tote. This is NORMAL.
2.  **Orientation:** Products rotated, flipped, or showing a different side (e.g., Lid vs Side) are NORMAL.
3.  **Quantity:**
    - Empty totes (0 items) are NORMAL.
    - Any quantity of the CORRECT product is NORMAL.
4.  **Outcome:** If the wrong product, wrong size, or wrong variant is found, classify as ANOMALOUS. If correct, proceed to Stage 3.

--------------------------------------------------------
STAGE 3: INTEGRITY & PACKAGING LOGIC
--------------------------------------------------------
Analyze Image 2 for physical condition and grouping logic.

1.  **The Multipack Rule:**
    - If input product name implies >1 unit (e.g., "4-pack", "2x", "4 stuks"), all units MUST be grouped in retail packaging.
    - A single loose unit outside its retail pack is ANOMALOUS.
    - Broken Multipacks (scattered yogurts intended to be wrapped) are ANOMALOUS.
2.  **Damage & Freshness:**
    - Check for crushed/dented items, ripped packaging, or exposed product.
    - Check for spoilage (mold, rot, browning).
    - Check for contamination (dirt/sludge ON the product).
3.  **Seals:**
    - Missing secondary lids (plastic overcaps) are NORMAL IF the inner foil seal is intact.
    - Exposed container interiors or jagged edges are ANOMALOUS.
4.  **Outcome:** If damage, open packaging, or loose multipack items are found, classify as ANOMALOUS. If intact, proceed to Stage 4.

--------------------------------------------------------
STAGE 4: FALSE POSITIVE FILTERS (The "Normalcy" Check)
--------------------------------------------------------
Review any potential defects identified above against these "Allowable" conditions. If the visual feature falls into these categories, it is NOT a defect.

1.  **Lighting & Material Artifacts:**
    - Glare, shiny reflections, or "mirroring" on plastic film is NORMAL.
    - Transparency: Seeing product color through a plastic base is NORMAL.
2.  **Harmless Debris (Must be DRY):**
    - Loose cardboard, inserts, stickers, separators, plastic air pillows, or loose caps are NORMAL (provided they are dry and not leaking).
3.  **Old Residue:**
    - Old, dry stains or light condensation on tote walls/floor are NORMAL (provided no pooling liquid or wetness on product).

--------------------------------------------------------
FINAL DECISION LOGIC
--------------------------------------------------------
- If the image failed Stage 1, 2, or 3 and was not cleared by Stage 4 -> ANOMALOUS.
- If the image passed all checks -> NORMAL.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
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