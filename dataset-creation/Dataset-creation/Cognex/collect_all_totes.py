import pandas as pd
import shutil
from pathlib import Path

# --- 1. CONFIGURATIE ---
# De map waar ALLES in komt te staan
OUTPUT_DIR = Path("data_all")

# Je input bestanden
CSV_FILE = Path("products.csv")
# LET OP: Check of deze mapnaam klopt! (train_validation of validation?)
INPUT_DIR = Path("train_validation") 

# --- 2. HOOFDLOGICA ---

def collect_and_rename_images():
    print(f"Starting process...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 30)

    # Maak de output map (als die nog niet bestaat)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CSV inlezen
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"CSV loaded: {len(df)} rows found.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read CSV file: {e}")
        return

    # Kolomnamen opschonen (voor de zekerheid)
    df.columns = df.columns.str.strip()

    # 2. Tellers initialiseren
    # We gebruiken een dictionary om bij te houden hoeveel foto's we al
    # hebben gehad voor elke unieke combinatie van (naam, sku).
    # Bijv: product_counters[('Coca_Cola', '12345')] = 2
    product_counters = {}

    success_count = 0
    fail_count = 0

    # 3. Loop door alle rijen
    for index, row in df.iterrows():
        try:
            # Haal de originele filenaam op
            original_filename = str(row['filename']).strip()
            source_path = INPUT_DIR / original_filename

            # Check of het bronbestand bestaat
            if not source_path.exists():
                # print(f"Warning: Source file not found: {original_filename}")
                fail_count += 1
                continue

            # Haal naam en SKU op en maak ze schoon voor gebruik in bestandsnaam
            raw_name = str(row['name']).strip()
            safe_name = raw_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
            sku = str(row['sku']).strip()

            # Maak een unieke sleutel voor dit product
            product_key = (safe_name, sku)

            # Hoog de teller op voor dit specifieke product
            if product_key not in product_counters:
                product_counters[product_key] = 1
            else:
                product_counters[product_key] += 1

            # Huidige tellerstand ophalen (bv. 1, 2, 3...)
            count = product_counters[product_key]

            # 4. Maak de NIEUWE bestandsnaam
            # Krijg de originele extensie (bv. .jpg of .png)
            extension = source_path.suffix
            # F-string met :02d zorgt voor voorloopnullen (01, 02, etc.)
            new_filename = f"{safe_name}_{sku}_{count:02d}{extension}"
            
            dest_path = OUTPUT_DIR / new_filename

            # 5. Kopiëren
            shutil.copy2(source_path, dest_path)
            # Optioneel: print elke kopieeractie (kan veel tekst zijn)
            # print(f"Copied: {original_filename} -> {new_filename}")
            success_count += 1

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            fail_count += 1

    print("-" * 30)
    print(f"Done! Successfully collected {success_count} images in '{OUTPUT_DIR}'.")
    if fail_count > 0:
        print(f"Warning: {fail_count} files could not be found or processed.")

if __name__ == "__main__":
    collect_and_rename_images()