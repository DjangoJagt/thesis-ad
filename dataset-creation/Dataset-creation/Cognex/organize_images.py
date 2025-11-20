import pandas as pd
import shutil
from pathlib import Path

# --- 1. DEFINE YOUR FOLDERS AND FILES ---
# We schrijven nu DIRECT naar de MVTec-structuur
OUTPUT_DIR = Path("data2/cognex_dataset")
CSV_FILE = Path("products.csv")
INPUT_DIR = Path("train_validation") # Zorg dat dit overeenkomt met je echte mapnaam
SUPPLY_CHAIN_DIR = INPUT_DIR / "supply_chain_photos"
TOTE_PHOTO_DIR = INPUT_DIR

# --- 2. MAIN SCRIPT LOGIC ---

def process_images():
    """
    Leest het CSV-bestand en organiseert afbeeldingen direct in een
    MVTec-compatibele structuur voor AnomalyDINO.
    Maakt productnamen uniek door de SKU toe te voegen.
    """
    
    print(f"Starting image organization...")
    print(f"Output directory (MVTec style): {OUTPUT_DIR}")
    print(f"CSV file: {CSV_FILE}")
    print("-" * 30)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # We hoeven alleen nog maar SKUs bij te houden om dubbel werk te voorkomen,
    # omdat de productnaam nu uniek is per SKU.
    copied_skus = set()

    # --- Read and Clean CSV ---
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df.columns = df.columns.str.strip()
    for col in ['filename', 'name', 'sku']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            print(f"Error: Missing expected column '{col}' in CSV.")
            return

    print(f"Found {len(df)} records in {CSV_FILE}.")

    # --- Iterate over each row in the CSV ---
    for index, row in df.iterrows():
        try:
            tote_filename = row['filename']
            raw_name = row['name'].replace(" ", "_").replace("+", "plus").replace("/", "_")
            sku = row['sku']

            if not all([tote_filename, raw_name, sku]):
                print(f"Warning: Skipping row {index} due to missing data.")
                continue

            # --- NIEUW: Maak productnaam uniek door SKU toe te voegen ---
            # Bijv: "merkloos_kastanjechampignons_12345"
            product_name = f"{raw_name}_{sku}"

            # --- 3. Definieer MVTec Paden ---
            product_folder = OUTPUT_DIR / product_name
            train_good_folder = product_folder / "train" / "good"
            test_good_folder = product_folder / "test" / "good"
            test_anomaly_folder = product_folder / "test" / "tote_photos"

            source_tote_path = TOTE_PHOTO_DIR / tote_filename

            # --- 4. CHECK IF TOTE IMAGE EXISTS FIRST ---
            if source_tote_path.exists():
                
                # --- 4a. Create Destination Folders ---
                train_good_folder.mkdir(parents=True, exist_ok=True)
                test_good_folder.mkdir(parents=True, exist_ok=True)
                test_anomaly_folder.mkdir(parents=True, exist_ok=True)

                # --- 4b. Process and Copy "Test" (Tote) Image ---
                dest_tote_path = test_anomaly_folder / tote_filename
                shutil.copy2(source_tote_path, dest_tote_path)

                # --- 5. Process and Copy "Good" (Supply Chain) Image ---
                # We kopiëren deze maar één keer per unieke product_name (wat nu per SKU is)
                if sku not in copied_skus:
                    all_files = list(SUPPLY_CHAIN_DIR.glob(f"{sku}.*"))
                    source_supply_files = [
                        f for f in all_files if ":Zone.Identifier" not in f.name
                    ]

                    if not source_supply_files:
                        print(f"  Warning: Supply chain file for SKU '{sku}' not found in {SUPPLY_CHAIN_DIR}")
                    else:
                        source_supply_path = source_supply_files[0]
                        
                        dest_train_path = train_good_folder / source_supply_path.name
                        dest_test_path = test_good_folder / source_supply_path.name
                        
                        shutil.copy2(source_supply_path, dest_train_path)
                        shutil.copy2(source_supply_path, dest_test_path)
                        
                        print(f"  [GOOD] Copied {source_supply_path.name} for {product_name}")
                        copied_skus.add(sku)
            
            else:
                # Optioneel: uncomment als je veel warnings krijgt
                # print(f"  Warning: Tote file not found: {source_tote_path}")
                pass
        
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    print("-" * 30)
    print("Image organization complete! (Unique folders by SKU)")


# --- Run the script ---
if __name__ == "__main__":
    process_images()