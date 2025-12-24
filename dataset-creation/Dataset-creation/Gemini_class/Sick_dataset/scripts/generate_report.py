import sys
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[2]

# 3. Add root to sys.path so Python can find 'Sick_dataset' module
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- NOW IMPORTS WILL WORK ---
import pandas as pd
import html
import argparse
import cv2
import base64

# This specific import caused the crash. Now it will work.
from Sick_dataset.utils.crop import robust_industrial_crop

# --- CONFIGURATION ---
SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPTS_DIR.parent
CSV_PATH = DATASET_DIR / "manifest_analyzed_leakage.csv"
HTML_OUTPUT = DATASET_DIR / "reports" / "verification_report_leakage.html"

def get_image_src(rel_path, do_crop=False):
    """
    Returns the string to put in <img src="...">.
    - If do_crop=False: Returns the file path (browser loads file).
    - If do_crop=True:  Loads, crops, and returns Base64 data URI (browser loads embedded image).
    """
    if pd.isna(rel_path) or not rel_path:
        return None

    # Clean path string
    rel_path = str(rel_path).replace("\\", "/")
    full_path = DATASET_DIR / rel_path

    if not do_crop:
        # Standard mode: just return path for browser to load locally
        return rel_path

    # --- CROPPING MODE ---
    if not full_path.exists():
        return None

    try:
        # 1. Load
        img = cv2.imread(str(full_path))
        if img is None: return None

        # 2. Crop
        img, _ = robust_industrial_crop(img, debug_mode=False)

        # 3. Encode to JPEG in memory
        _, buffer = cv2.imencode('.jpg', img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        # 4. Return Data URI
        return f"data:image/jpeg;base64,{b64_str}"
        
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return rel_path # Fallback to original if crop fails

def generate_html(crop_mode=False):
    if not CSV_PATH.exists():
        print(f"Error: Could not find {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    if "ai_status" not in df.columns:
        print("Error: 'ai_status' column not found.")
        return

    df = df[df["ai_status"].notna()]
    
    # Stats
    total = len(df)
    df["ai_status"] = df["ai_status"].astype(str).str.strip().str.upper()
    normal_count = (df["ai_status"] == "NORMAL").sum()
    anomalous_count = (df["ai_status"] == "ANOMALOUS").sum()

    mode_text = "CROPPED VIEW" if crop_mode else "ORIGINAL VIEW"
    print(f"Generating report ({mode_text}): {total} images...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Defect Verification ({mode_text})</title>
        <style>
            body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #e0e0e0; padding: 10px; margin: 0; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            
            .controls {{ 
                background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; 
                display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                position: sticky; top: 10px; z-index: 100;
            }}
            .stats span {{ margin-right: 15px; font-weight: bold; }}
            .mode-badge {{ background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 10px; }}
            
            .btn {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin-left: 5px; }}
            .btn:hover {{ opacity: 0.9; }}
            .btn-all {{ background: #6c757d; color: white; }}
            .btn-norm {{ background: #28a745; color: white; }}
            .btn-anom {{ background: #dc3545; color: white; }}

            .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
            
            .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); overflow: hidden; }}
            
            .header {{ 
                padding: 10px 15px; border-bottom: 1px solid #eee; 
                display: flex; justify-content: space-between; align-items: center; background: #fdfdfd;
            }}
            .product-name {{ font-size: 1.1em; font-weight: bold; color: #333; }}
            .sku-badge {{ background: #eee; padding: 2px 8px; border-radius: 4px; color: #555; font-size: 0.9em; }}

            .images {{ display: flex; height: 500px; background: #f9f9f9; }} 
            .img-container {{ flex: 1; border-right: 1px solid #fff; position: relative; }}
            .img-container img {{ width: 100%; height: 100%; object-fit: contain; padding: 5px; box-sizing: border-box; }}
            .img-label {{ position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.6); color: white; padding: 2px 6px; font-size: 12px; border-radius: 4px; }}
            
            .footer {{ padding: 10px 15px; background: #fff; border-top: 1px solid #eee; display: flex; align-items: center; justify-content: space-between; }}
            .status-group {{ display: flex; align-items: center; }}
            .status-badge {{ padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; font-size: 0.9em; margin-right: 15px; min-width: 80px; text-align: center; }}
            .reason-text {{ color: #444; font-size: 0.95em; line-height: 1.3; font-weight: 500; }}
            .obs-text {{ font-size: 0.85em; color: #666; font-style: italic; margin-left: auto; max-width: 300px; text-align: right; }}
            
            .NORMAL {{ border-left: 8px solid #28a745; }}
            .bg-NORMAL {{ background-color: #28a745; }}
            
            .ANOMALOUS {{ border-left: 8px solid #dc3545; }}
            .bg-ANOMALOUS {{ background-color: #dc3545; }}
            
            .ERROR {{ border-left: 8px solid #ffc107; }}
            .bg-ERROR {{ background-color: #ffc107; color: black; }}

            .SKIPPED {{ border-left: 8px solid #6c757d; }}
            .bg-SKIPPED {{ background-color: #6c757d; }}
        </style>
        
        <script>
            function filterCards(status) {{
                const cards = document.querySelectorAll('.card');
                cards.forEach(card => {{
                    if (status === 'ALL' || card.classList.contains(status)) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
        
        <div class="controls">
            <div class="stats">
                <span style="font-size: 1.2em;">🔍 Results Analysis <span class="mode-badge">{mode_text}</span></span>
                <span>Total: {total}</span>
                <span style="color:#28a745;">Normal: {normal_count}</span>
                <span style="color:#dc3545;">Anomalous: {anomalous_count}</span>
            </div>
            <div>
                <button class="btn btn-all" onclick="filterCards('ALL')">Show All</button>
                <button class="btn btn-norm" onclick="filterCards('NORMAL')">Only Normal</button>
                <button class="btn btn-anom" onclick="filterCards('ANOMALOUS')">Only Anomalous</button>
            </div>
        </div>

        <div class="grid">
    """

    for idx, row in df.iterrows():
        status = str(row.get("ai_status", "UNKNOWN")).strip().upper()
        reason = html.escape(str(row.get("ai_reason", "N/A")))
        obs = html.escape(str(row.get("ai_obs", "")))  # Show observations if available
        product = html.escape(str(row.get("product_name", "Unknown Product")))
        sku = html.escape(str(row.get("sku", "Unknown SKU")))
        
        # 1. Get Image Sources
        # We NEVER crop reference images (consistent with detection script)
        ref_src = get_image_src(row.get("sku_template_path"), do_crop=False)
        
        # We CONDITIONALLY crop tote images based on the flag
        tote_src = get_image_src(row.get("staging_filepath"), do_crop=crop_mode)

        ref_display = f'<img src="{ref_src}" loading="lazy">' if ref_src else '<div style="padding:20px; color:#999;">No Reference Image</div>'
        tote_display = f'<img src="{tote_src}" loading="lazy">' if tote_src else '<div style="padding:20px; color:#999;">No Tote Image</div>'

        html_content += f"""
        <div class="card {status}">
            <div class="header">
                <span class="product-name">{product}</span>
                <span class="sku-badge">SKU: {sku}</span>
            </div>
            <div class="images">
                <div class="img-container">
                    <span class="img-label">Reference</span>
                    {ref_display}
                </div>
                <div class="img-container">
                    <span class="img-label">Tote Audit ({'Cropped' if crop_mode else 'Original'})</span>
                    {tote_display}
                </div>
            </div>
            <div class="footer">
                <div class="status-group">
                    <span class="status-badge bg-{status}">{status}</span>
                    <span class="reason-text">{reason}</span>
                </div>
                <div class="obs-text">{obs}</div>
            </div>
        </div>
        """

    html_content += """
        </div>
        </div>
    </body>
    </html>
    """

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Report generated: {HTML_OUTPUT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", action="store_true", help="Display CROPPED images in report (what the AI saw)")
    args = parser.parse_args()
    
    generate_html(crop_mode=args.crop)