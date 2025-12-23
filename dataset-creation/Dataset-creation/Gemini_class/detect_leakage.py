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
OUTPUT_CSV = SICK_DIR / "manifest_analyzed_leakage.csv"

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
# SYSTEM_PROMPT = """
# You are an expert Visual QA assistant focused ONLY on leakage/wetness detection.

# TASK
# - Inspect the provided cropped tote image and any optional floor tiles.
# - Determine if there is visible wetness, pooling liquid, streaks, puddles, or fresh leakage.

# RULES
# - If ANY tile or the cropped image shows wetness/liquid → classify as ANOMALOUS.
# - If uncertain whether a spot is wet or dry → classify as ANOMALOUS.
# - Ignore product identity, quantity, packaging, and non-leakage defects.

# OUTPUT
# Return ONLY this JSON (no markdown):
# {{
#   "status": "NORMAL" or "ANOMALOUS",
#   "reason": "N/A" (if Normal) or a short description of leakage evidence
# }}
# """

SYSTEM_PROMPT = """
You are a Forensic Surface Analyst. Your task is to detect PRODUCT LEAKAGE (liquids, smears, residues) in a tote while ignoring benign artifacts.

--------------------------------------------------
CORE DEFINITION OF LEAKAGE
--------------------------------------------------
Leakage includes BOTH:
1. **Volumetric Spills:** Blobs, pools, or ridges of substance (e.g., yogurt, gel).
2. **Thin Films & Residues:** Wet smears, droplets, or glossy patches that have NO significant height but change the surface texture.

--------------------------------------------------
VISUAL DISCRIMINATION GUIDE
--------------------------------------------------

**1. THE "WET vs. REFLECTION" TEST (Crucial for Films)**
   - **Normal Reflection (Specular Highlight):**
     * Shape: Sharp, geometric, or linear.
     * Behavior: Moves across the surface smoothly; follows the curve of the tote walls.
     * Texture: The underlying plastic looks smooth.
   - **Leakage (Wet Sheen/Film):**
     * Shape: **Irregular, organic, or "patchy."** Looks like a spill pattern.
     * Behavior: **Interrupts** the surface reflection. Looks "greasy" or "smeared."
     * Texture: Alters the way light hits the plastic (diffused gloss).

**2. THE "WHITE MARK" TEST**
   - **Dry Scuff (Normal):** Matte, chalky, scratchy. reflects NO light.
   - **Product Cap (Normal):** Perfect geometric circle visible *through* cardboard.
   - **Leakage (Anomalous):** Any white mark that is **Glossy**, **Creamy**, or **Translucent**.

**3. THE "CARDBOARD" TEST**
   - **Shadow (Normal):** Diffuse grey area. No edge definition.
   - **Wicking (Anomalous):** **High-Contrast** dark spot with **Feathered/Sharp** edges. Looks saturated.

--------------------------------------------------
DECISION LOGIC
--------------------------------------------------
**Classify ANOMALOUS if:**
1.  You see **ANY** substance with volume (blobs, ridges).
2.  You see **ANY** irregular glossy patch or smear (even if flat/thin) that is NOT a direct light reflection.
3.  You see distinct **wicking** on cardboard.

**Classify NORMAL if:**
1.  All white marks are strictly **matte/chalky** (scuffs).
2.  All shiny areas are **sharp/geometric** reflections of the light source.
3.  Dark marks are diffuse shadows without saturation.

--------------------------------------------------
OUTPUT (JSON ONLY)
--------------------------------------------------
{{
  "analysis": "Describe the TEXTURE (Wet vs Dry) and SHAPE (Organic vs Geometric) of the spots.",
  "confidence_score": 0-100,
  "status": "NORMAL" or "ANOMALOUS",
  "reason": "Cite the specific leakage indicator (e.g., 'Irregular wet film', 'Wicking') or benign artifact."
}}
"""

# SYSTEM_PROMPT = """
# You are a Forensic Surface Analyst. Your ONLY task is to detect PRODUCT LEAKAGE
# or LIQUID RESIDUE versus harmless DRY MARKS on a tote interior.

# --------------------------------------------------
# DECISION RULE (CRITICAL)
# --------------------------------------------------
# Classify as NORMAL ONLY if all suspicious areas are clearly DRY.
# If ANY area looks liquid-like or cannot be confidently explained as dry → ANOMALOUS.

# When in doubt, choose ANOMALOUS.

# --------------------------------------------------
# LEAKAGE INDICATORS (ANY = ANOMALOUS)
# --------------------------------------------------
# - Glossy or shiny reflections inconsistent with dry surfaces
# - Visible thickness, blobs, smears, or material sitting ON the surface
# - Darkened or soaked cardboard fibers; feathered or spreading edges
# - Pooling in corners, seams, or low points; gravity streaks
# - Residue or smears on walls or vertical surfaces
# - White, milky, translucent, oily, or viscous-looking substances

# --------------------------------------------------
# DRY / NORMAL MARKS (ONLY IF CLEAR)
# --------------------------------------------------
# A mark may be considered DRY ONLY if it is:
# - Matte (no shine)
# - Flat (no volume)
# - Sharp-edged
# - Uniform and scuff-like
# - Not absorbed, pooled, smeared, or spreading

# If any condition above is not clearly met → ANOMALOUS.

# --------------------------------------------------
# REQUIRED REASONING STEPS
# --------------------------------------------------
# 1. Identify all spots or discolorations.
# 2. For each spot, assess: shine, volume, absorption, edge behavior, and location.
# 3. Decide using the Decision Rule above.

# --------------------------------------------------
# OUTPUT (JSON ONLY)
# --------------------------------------------------
# {
#   "analysis": "Brief description of observed spots and visual cues.",
#   "confidence_score": 0-100,
#   "status": "NORMAL" or "ANOMALOUS",
#   "reason": "Short final verdict"
# }
# """


# SYSTEM_PROMPT = """
# You are an expert Forensic Surface Analyst for Supply Chain Quality.

# YOUR ONLY TASK
# Determine whether there is evidence of PRODUCT LEAKAGE or LIQUID RESIDUE
# versus harmless DRY SCUFFS or OLD NON-LIQUID MARKS.

# You will receive one or more high-resolution images of a tote interior
# (floor, walls, corners, or under-edge areas).

# DO NOT evaluate:
# - Product identity
# - SKU correctness
# - Quantity
# - Packaging integrity (except leakage evidence)
# - Any non-leakage defects

# --------------------------------------------------
# CORE PRINCIPLE (VERY IMPORTANT)
# --------------------------------------------------
# Classify as NORMAL ONLY if all suspicious areas are clearly and confidently DRY.
# If any suspicious area cannot be confidently explained as dry → classify as ANOMALOUS.

# When uncertain, err on the side of ANOMALOUS.

# --------------------------------------------------
# WHAT COUNTS AS LEAKAGE / LIQUID RESIDUE (ANOMALOUS)
# --------------------------------------------------
# ANY of the following visible cues indicate leakage or liquid residue:

# 1. SURFACE REFLECTION / SPECULARITY
#    - Glossy or shiny highlights
#    - Light reflections inconsistent with the surrounding dry plastic or cardboard

# 2. VOLUME / 3D PRESENCE
#    - Blobs, smears, thickness, or material sitting ON the surface
#    - Rounded edges or uneven buildup

# 3. ABSORPTION / SOAKING
#    - Darkened or saturated cardboard fibers
#    - Feathered edges or gradients spreading into the material
#    - Uneven discoloration suggesting liquid penetration

# 4. POOLING / GRAVITY EFFECTS
#    - Liquid collecting in corners, seams, or low points
#    - Streaks or trails following gravity direction
#    - Residue at floor-wall junctions

# 5. WALL OR VERTICAL SMEARS
#    - Drips, streaks, or residue on tote walls
#    - Opaque or translucent material stuck to vertical surfaces

# 6. LIQUID-LIKE APPEARANCE
#    - White, milky, translucent, oily, or viscous-looking substances
#    - Smooth smears rather than dusty or powdery texture

# --------------------------------------------------
# WHAT MAY BE CONSIDERED DRY / NORMAL (ONLY IF CLEAR)
# --------------------------------------------------
# A mark may be considered DRY and NON-LEAKAGE ONLY if ALL apply:

# - Matte finish (no shine or gloss)
# - Flat appearance with no visible thickness
# - Sharp, well-defined edges
# - Uniform texture consistent with plastic scuffs or old residue
# - No signs of absorption, pooling, streaking, or spreading
# - Looks dusty, worn, or ingrained rather than smeared

# If any of these conditions are NOT clearly met → ANOMALOUS.

# --------------------------------------------------
# REQUIRED ANALYSIS STEPS
# --------------------------------------------------
# Follow these steps explicitly:

# 1. SCAN
#    Identify all spots, discolorations, stains, or foreign material.

# 2. TEXTURE & LIGHT ANALYSIS (for each spot)
#    Ask:
#    - Does it reflect light differently than the surrounding surface?
#    - Does it appear to have volume or thickness?
#    - Is it absorbed into the material or sitting on top?
#    - Are edges sharp or feathered?

# 3. CONTEXT CHECK
#    - Is the location consistent with where liquid would collect or flow?
#    - Is it on the floor, in a corner, under an edge, or on a wall?

# 4. DECISION
#    - If ANY spot appears liquid-like or cannot be confidently proven dry → ANOMALOUS
#    - Only classify NORMAL if ALL spots are clearly dry and harmless

# --------------------------------------------------
# OUTPUT FORMAT (STRICT)
# --------------------------------------------------
# Return a valid JSON object ONLY:

# {
#   "analysis": "Brief description of observed spots and their visual properties (shine, texture, absorption, location).",
#   "confidence_score": 0-100,
#   "status": "NORMAL" or "ANOMALOUS",
#   "reason": "Short final verdict explaining why it is dry or why leakage is suspected"
# }
# """



  
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