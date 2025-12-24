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
from Sick_dataset.utils.crop import (
    robust_industrial_crop,
    extract_floor_tiles,
    extract_floor_tiles_4x,
)

# --- PATH CONFIGURATION ---
# Assuming script is run from the root folder where .env is
ROOT_DIR = Path(__file__).resolve().parent
SICK_DIR = ROOT_DIR / "Sick_dataset" # Adjust if your folder structure differs
INPUT_CSV = SICK_DIR / "manifest.csv"
OUTPUT_CSV = SICK_DIR / "manifest_analyzed_leakage_gpt5-mini.csv"

# --- 1. LOAD .ENV (Crucial for Picnic Auth) ---
env_file = ROOT_DIR / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

# --- AI CONFIGURATION ---
MODEL_NAME = "openai/gpt-5-mini" 
TEMPERATURE = 0.0
# Limit concurrent requests to avoid Rate Limits & Memory Crashes
MAX_CONCURRENT_REQUESTS = 5 

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)



# --- THE SYSTEM PROMPT ---
SYSTEM_PROMPT = """
# Role and Objective
You are a Forensic Surface Analyst. Your task is to detect PRODUCT LEAKAGE (liquids, smears, residues) in a tote while ignoring benign artifacts. Your primary goal is to distinguish between actual wet residues and normal specular reflections, packaging color casts, or dry material scuffs.

--------------------------------------------------
# Reasoning Strategy (Internal Checklist)
--------------------------------------------------
Follow the steps below internally as a checklist to reach your decision.
Do NOT enumerate or restate these steps in the output.

1. **Context Check:**  
   Identify only the *visible* products and note their packaging materials (plastic, cardboard) and primary colors.

2. **Visual Inventory:**  
   Scan the tote floor, walls, and cardboard for any marks, highlights, discolorations, or dark areas.

3. **Evidence Testing:**  
   Apply the <discrimination_guide> to each observed mark.

4. **Physics Refutation:**  
   Actively test benign explanations:
   - Could the shape be caused by light reflecting off curved or textured plastic?
   - Could the color or sheen be a reflection or cast from nearby product packaging or lighting?

5. **Synthesis:**  
   Decide whether any mark remains that cannot be reasonably explained as a normal artifact.

--------------------------------------------------
# Instructions
--------------------------------------------------

<discrimination_guide>

## 1. THE "WET vs. REFLECTION" TEST

- **Normal Reflection (Specular Highlight):**
  * SHAPE: Sharp, linear, or irregular due to curved or ribbed plastic.
  * TEXTURE: Matches the surrounding clean plastic gloss.
  * CONTEXT: Follows lighting direction and tote geometry.

- **Leakage (Wet Sheen / Film / Residue):**
  * TEXTURE: Alters surface appearance (diffused gloss, greasy, smeared).
  * BEHAVIOR: Interrupts or breaks the normal reflection pattern.
  * VISCOSITY: May show pooling, droplets, ridges, or residue boundaries.

## 2. THE "WHITE MARK" TEST

- **Dry Scuff (Normal):**  
  Matte, chalky, scratchy; reflects no light.

- **Product Cap (Normal):**  
  Perfect geometric circle visible through cardboard.

- **Leakage (Anomalous):**  
  Glossy, creamy, translucent, or viscous white residue.

## 3. THE "CARDBOARD" TEST

- **Shadow (Normal):**  
  Diffuse grey area with soft edges and no saturation.

- **Wicking (Anomalous):**  
  High-contrast dark area with feathered or sharp edges that appears saturated or bleeding into fibers.

</discrimination_guide>

--------------------------------------------------
# Decision Logic
--------------------------------------------------

Classify **ANOMALOUS** if one or more of the following are present **and cannot be reasonably explained by benign artifacts**:

1. Any substance with clear physical volume (blobs, pools, ridges).
2. An irregular glossy patch or smear that clearly alters the surface texture of the plastic (not just a light reflection);  
   prefer supporting liquid-behavior cues such as broken reflection patterns, smeared edges, droplets, pooling, or residue boundaries.
3. Distinct, high-contrast wicking or saturation on cardboard.

Classify **NORMAL** if:

1. White marks are strictly matte or chalky (dry scuffs).
2. Shiny areas are consistent with specular highlights or follow the geometric curve of the tote walls.
3. Dark marks on cardboard are diffuse shadows without saturation or fiber wicking.

--------------------------------------------------
# Final Arbitration (Highest Priority)
--------------------------------------------------
If no pooling, spreading, saturation, deformation, or gravity-consistent liquid behavior is visible, and the appearance is fully explained by lighting, reflections, packaging color casts, or normal material scuffs, the correct classification is **NORMAL**.

--------------------------------------------------
# Output Format (JSON ONLY)
--------------------------------------------------
Keep all text fields concise (2–4 sentences total across all text fields).

{
  "justification": "Brief summary explaining why observed marks are classified as leakage or as benign artifacts (e.g., reflection, shadow, color cast).",
  "analysis": "Concise description of key marks, focusing on TEXTURE (Wet vs Dry) and SHAPE (Organic vs Geometric).",
  "confidence_score": 0-100,
  "status": "NORMAL" or "ANOMALOUS",
  "reason": "Cite the specific leakage indicator or clearly identify the benign artifact."
}

--------------------------------------------------
# FINAL RULES (Instruction Anchoring)
--------------------------------------------------
- Do NOT classify a spot as ANOMALOUS based solely on an irregular shape; curved plastic naturally creates irregular reflections.
- Do NOT ignore packaging color; if a spot matches the color of a nearby product, treat it as a likely reflection or color cast unless additional liquid behavior (e.g., pooling, smearing, saturation) is present.
- **FINAL RULE:** If the appearance is fully explained by light, reflections, or material scuffs, you must classify the tote as **NORMAL**.
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
    b64_str, mime_type = encode_np_image(img, encode_format=encode_format)

    return b64_str, mime_type, img


def encode_np_image(
    img: np.ndarray,
    encode_format: str = ".png",
):
    """
    Encodes a numpy BGR image to base64 along with mime type.
    Returns (base64_string, mime_type).
    """
    if img is None:
        raise ValueError("encode_np_image received None image")

    success, buffer = cv2.imencode(encode_format, img)
    if not success:
        raise ValueError("Could not encode image buffer")

    file_bytes = buffer.tobytes()
    b64_str = base64.b64encode(file_bytes).decode("utf-8")

    if encode_format == ".png":
        mime_type = "image/png"
    elif encode_format in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    else:
        raise ValueError(f"Unsupported encode format: {encode_format}")

    return b64_str, mime_type


def get_image_data_floor(
    cropped_img: np.ndarray,
    encode_format: str = ".png",
):
    """
    Extracts 4 tote floor tiles from the processed tote image and returns
    encoded payloads for leakage-only inspection.
    
    Returns:
        (tile_top_left_b64, tile_top_left_mime, 
         tile_top_right_b64, tile_top_right_mime,
         tile_bottom_left_b64, tile_bottom_left_mime,
         tile_bottom_right_b64, tile_bottom_right_mime)
    """
    if cropped_img is None or cropped_img.size == 0:
        return None, None, None, None, None, None, None, None

    try:
        tile_tl, tile_tr, tile_bl, tile_br = extract_floor_tiles_4x(cropped_img)
    except Exception as e:
        LOGGER.warning(
            f"Floor tile extraction failed; continuing without tiles. Error: {e}"
        )
        return None, None, None, None, None, None, None, None

    def encode_tile(tile):
        if tile is None or tile.size == 0:
            return None, None
        try:
            return encode_np_image(tile, encode_format=encode_format)
        except Exception as tile_err:
            LOGGER.warning(
                f"Encoding floor tile failed; skipping tile. Error: {tile_err}"
            )
            return None, None

    tile_tl_b64, tile_tl_mime = encode_tile(tile_tl)
    tile_tr_b64, tile_tr_mime = encode_tile(tile_tr)
    tile_bl_b64, tile_bl_mime = encode_tile(tile_bl)
    tile_br_b64, tile_br_mime = encode_tile(tile_br)

    return (tile_tl_b64, tile_tl_mime, 
            tile_tr_b64, tile_tr_mime, 
            tile_bl_b64, tile_bl_mime, 
            tile_br_b64, tile_br_mime)


async def analyze_row(
    llm_client: AsyncLLMClient, 
    row: Dict[str, str], 
    row_idx: int, 
    semaphore: asyncio.Semaphore,
    crop: bool = False,
    floor_tiles: list[str] | None = None,
) -> Dict[str, str]:
    """Process a single row with rate limiting."""
    
    async with semaphore: # Wait here if 5 requests are already running
        
        # 1. MAP CORRECT COLUMNS FROM MANIFEST (Leakage-only: only need tote image)
        tote_rel = row.get("staging_filepath", "")
        if not tote_rel:
            row["ai_status"] = "SKIPPED"
            row["ai_reason"] = "Missing tote path in CSV"
            return row

        tote_path = SICK_DIR / tote_rel

        # 2. VALIDATE FILE EXISTS
        if not tote_path.exists():
            row["ai_status"] = "ERROR"
            row["ai_reason"] = f"Tote image not found: {tote_rel}"
            return row

        try:
            # 3. PREPARE PAYLOAD (Updated for dynamic MIME types)
            # Tote image (optionally cropped) plus encoded payload
            tote_b64, tote_mime, tote_proc = get_image_data(
                tote_path,
                do_crop=crop,
                encode_format=".png",
            )

            # Compute requested floor tiles from the processed tote image
            floor_tiles = floor_tiles or []

            # Pre-extract tiles if requested
            tiles_2 = None
            tiles_4 = None
            if any(t in ["top", "bottom"] for t in floor_tiles):
                try:
                    tiles_2 = extract_floor_tiles(tote_proc)
                except Exception as e:
                    LOGGER.warning(f"2-tile extraction failed; continuing without 2-tiles. Error: {e}")
            if any(t in ["tl", "tr", "bl", "br"] for t in floor_tiles):
                try:
                    tiles_4 = extract_floor_tiles_4x(tote_proc)
                except Exception as e:
                    LOGGER.warning(f"4-tile extraction failed; continuing without 4-tiles. Error: {e}")

            # 1. PREPARE THE PROMPT WITH CONTEXT (Leakage-only)
            filled_prompt = SYSTEM_PROMPT

            content_items = [
                {"type": "text", "text": filled_prompt},
                {"type": "text", "text": "Main: Cropped Tote Image — Leakage-only"},
                {"type": "image_url", "image_url": {"url": f"data:{tote_mime};base64,{tote_b64}"}},
            ]

            # Helper to encode tiles
            def encode_tile(tile_arr):
                if tile_arr is None or tile_arr.size == 0:
                    return None, None
                try:
                    return encode_np_image(tile_arr, encode_format=".png")
                except Exception as e:
                    LOGGER.warning(f"Encoding tile failed; skipping. Error: {e}")
                    return None, None

            # Append selected tiles only
            for t in floor_tiles:
                if t in ["top", "bottom"] and tiles_2 is not None:
                    top_tile, bottom_tile = tiles_2
                    if t == "top":
                        b64, mt = encode_tile(top_tile)
                        if b64 and mt:
                            content_items.extend([
                                {"type": "text", "text": "Tile: Top Half — Leakage-only"},
                                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
                            ])
                    else:
                        b64, mt = encode_tile(bottom_tile)
                        if b64 and mt:
                            content_items.extend([
                                {"type": "text", "text": "Tile: Bottom Half — Leakage-only"},
                                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
                            ])
                elif t in ["tl", "tr", "bl", "br"] and tiles_4 is not None:
                    tl, tr, bl, br = tiles_4
                    tile_map = {
                        "tl": (tl, "Top-Left"),
                        "tr": (tr, "Top-Right"),
                        "bl": (bl, "Bottom-Left"),
                        "br": (br, "Bottom-Right"),
                    }
                    tile_arr, label = tile_map[t]
                    b64, mt = encode_tile(tile_arr)
                    if b64 and mt:
                        content_items.extend([
                            {"type": "text", "text": f"Tile: {label} — Leakage-only"},
                            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
                        ])

            messages = [user(content_items)]

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
            row["ai_analysis"] = result.get("analysis", "")
            row["ai_confidence_score"] = result.get("confidence_score", "")
            
            # Log by tote filename to avoid undefined product_name
            try:
                tote_name = Path(tote_rel).name
            except Exception:
                tote_name = str(tote_rel)
            LOGGER.info(f"[{row_idx}] {tote_name}: {row['ai_status']} ({row['ai_reason']})")

        except Exception as e:
            LOGGER.error(f"Row {row_idx} Failed: {e}")
            row["ai_status"] = "ERROR"
            row["ai_reason"] = str(e)

        return row

async def main():
    parser = argparse.ArgumentParser(description="Run Visual QA on Staging Data")
    parser.add_argument("--limit", type=int, help="Only run X images (good for testing)")
    parser.add_argument("--filter-sku", type=str, nargs='*', help="Run only for these specific SKU(s) (e.g., --filter-sku 11400153 11400464)")
    parser.add_argument("--filter-product", type=str, help="Run only for products containing this text (e.g., 'Banana')")
    parser.add_argument("--crop", action="store_true", help="Enable robust cropping for tote images")
    parser.add_argument(
        "--floor-tiles",
        type=str,
        nargs='*',
        choices=["top", "bottom", "tl", "tr", "bl", "br"],
        help=(
            "Optional: which floor tiles to include with the cropped image. "
            "Choices: top, bottom (2-tile); tl, tr, bl, br (4-tile). "
            "Example: --floor-tiles top tl br"
        ),
    )
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
        if args.filter_sku and row.get("sku") not in args.filter_sku:
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
        tasks.append(
            analyze_row(
                llm_client,
                row,
                idx,
                semaphore,
                crop=args.crop,
                floor_tiles=args.floor_tiles,
            )
        )

    if not tasks:
        LOGGER.warning("No rows matched your filters.")
        return

    # Run Analysis
    results = await asyncio.gather(*tasks)

    # Write Output
    # We write a NEW file with just the processed results
    output_fields = fieldnames + ["ai_status", "ai_reason", "ai_analysis", "ai_confidence_score"]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)
    
    LOGGER.info(f"Finished! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())