import os
import shutil
from pathlib import Path

# ---- CONFIGURATION ----
# Where your iPhone images currently are
SOURCE_DIR = Path("raw_iphone_images") 

# Where you want the organized folders to go
TARGET_DIR = Path("ordered_dataset")

# The Product Data (ID, Name, and Image Ranges)
# Note: Ranges are inclusive (start, end)
PRODUCT_MAP = [
    {
        "id": "11478313", 
        "name": "Picnic Bio halfvolle melk 1 liter", 
        "ranges": [(8302, 8359)]
    },
    {
        "id": "12077488", 
        "name": "Picnic Yoghurt griekse stijl 2pct vet 1 kg", 
        "ranges": [(8360, 8416)]
    },
    {
        "id": "11738660", 
        "name": "Picnic Zure room 125 gram", 
        "ranges": [(8417, 8482)]
    },
    {
        "id": "10512371", 
        "name": "Van Wijngaardens Zaanse mayonaise 200 ml", 
        "ranges": [(8483, 8529)]
    },
    {
        "id": "11780914", 
        "name": "Lodik Lodik dikke bleek toilet 750 ml", 
        "ranges": [(8530, 8587)]
    },
    {
        "id": "11810623", 
        "name": "De Klok pils", 
        "ranges": [(8588, 8642)]
    },
    {
        "id": "11801970", 
        "name": "Picnic Cola zero sugar 6 x 500 ml", 
        "ranges": [(8643, 8655), (8832, 8889)] # Handling the split range
    },
    {
        "id": "12461215", 
        "name": "Picnic mangostukjes 240 gr", 
        "ranges": [(8656, 8714)]
    },
    {
        "id": "11463862", 
        "name": "Zoete aardappel in blokjes", 
        "ranges": [(8715, 8765)]
    },
    {
        "id": "11299303", 
        "name": "Merkloos Blauwe bessen 500 gram", 
        "ranges": [(8766, 8831)]
    },
]

def sanitize_name(name):
    """Replaces spaces with underscores and removes special chars for folder safety."""
    clean = name.replace(" ", "_").replace("%", "pct").replace("&", "en")
    return "".join(c for c in clean if c.isalnum() or c in ('_', '-'))

def process_images():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    # Create target directory if it doesn't exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    total_moved = 0

    for product in PRODUCT_MAP:
        # 1. Create Folder Name: ID
        # safe_name = sanitize_name(product['name'])
        folder_name = f"{product['id']}" #_{safe_name}"
        product_dir = TARGET_DIR / folder_name
        product_dir.mkdir(exist_ok=True)
        
        print(f"Processing: {folder_name}...")
        
        # 2. Iterate through ranges
        counter = 0
        for start, end in product['ranges']:
            # range is inclusive, so end + 1
            for i in range(start, end + 1):
                # Construct possible source filenames (iPhone usually uses IMG_xxxx.JPEG)
                src_filename = f"IMG_{i}.JPEG" 
                src_path = SOURCE_DIR / src_filename
                
                # Check if file exists (sometimes extension might be .jpg or .jpeg)
                if not src_path.exists():
                    # Try alternate extension just in case
                    src_path = SOURCE_DIR / f"IMG_{i}.jpg"
                
                if src_path.exists():
                    # 3. Create New Name: ID_000.jpg
                    new_filename = f"{product['id']}_{counter:03d}.jpg"
                    dest_path = product_dir / new_filename
                    
                    # Copy the file
                    shutil.copy2(src_path, dest_path)
                    counter += 1
                    total_moved += 1
                else:
                    print(f"  Warning: Missing file IMG_{i}")

        print(f"  -> Saved {counter} images.")

    print(f"\nDone! Organized {total_moved} images into '{TARGET_DIR}'.")

if __name__ == "__main__":
    process_images()