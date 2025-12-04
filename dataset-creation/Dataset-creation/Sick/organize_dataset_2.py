import os
import shutil
import csv
from pathlib import Path
from collections import defaultdict

# --- INSTELLINGEN ---
SOURCE_DIR = Path("sick_data_staging")       # Waar je foto's nu staan
TARGET_DIR = Path("sick_data_issues_3")  # De nieuwe output map
ISSUE_LIST_FILE = "flagged_quality_issues_2.txt"
MANIFEST_FILE = "manifest.csv"
SUMMARY_FILE = "dataset_summary.csv"

# Split ratio voor Totes (bijv. 0.7 betekent 70% van de bakken naar train)
TRAIN_RATIO = 0.55 

# MINIMAAL AANTAL TOTES
# We hebben minimaal 2 unieke bakken met goede data nodig om betrouwbaar te splitten.
MIN_GOOD_TOTES = 2

def get_tote_id(filename):
    """
    Haalt het Tote ID uit de bestandsnaam.
    Verwacht formaat: DATUM-TOTE-SKU-TIJD.png
    """
    try:
        parts = filename.split('-')
        if len(parts) >= 2:
            return parts[1]
    except Exception:
        pass
    return "unknown_tote"

def load_product_names(manifest_path):
    """Leest SKU -> Product Name mapping uit manifest.csv"""
    sku_map = {}
    if not os.path.exists(manifest_path):
        print(f"WAARSCHUWING: {manifest_path} niet gevonden. Productnamen blijven leeg.")
        return sku_map
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sku = row.get('sku') or row.get('article_id')
                name = row.get('product_name') or row.get('article_name')
                if sku and name:
                    sku_map[sku.strip()] = name.strip()
    except Exception as e:
        print(f"FOUT bij lezen manifest: {e}")
    return sku_map

def main():
    print(f"--- START DATASET ORGANISATIE (SORTED BY GOOD COUNT) ---")
    
    # 0. Laad productnamen
    product_names = load_product_names(MANIFEST_FILE)
    
    # 1. Lees het issue bestand in
    print(f"Lezen van {ISSUE_LIST_FILE}...")
    issues_map = defaultdict(set)
    summary_data = [] 
    
    skipped_low_data = 0
    skipped_no_issues = 0
    processed_count = 0
    
    if not Path(ISSUE_LIST_FILE).exists():
        print(f"FOUT: Kan {ISSUE_LIST_FILE} niet vinden.")
        return

    with open(ISSUE_LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[source") or "Folder name" in line: 
                continue 
            parts = line.split('\t') 
            if len(parts) < 2: parts = line.split()
            
            if len(parts) >= 2:
                sku = parts[0].strip()
                filename = parts[1].strip()
                issues_map[sku].add(filename)

    print(f"{len(issues_map)} SKU's gevonden in de issue-lijst.")
    print("-" * 60)

    # 2. Ga door elke SKU heen
    if not SOURCE_DIR.exists():
        print(f"FOUT: Bronmap {SOURCE_DIR} bestaat niet.")
        return

    for sku_path in SOURCE_DIR.iterdir():
        if not sku_path.is_dir():
            continue
            
        sku = sku_path.name
        bad_filenames = issues_map.get(sku, set())

        # 3. Inventariseer bestanden per Tote
        tote_data = defaultdict(lambda: {'good': [], 'issue': []})
        all_files = [f for f in sku_path.iterdir() if f.is_file()]
        
        if not all_files:
            continue

        total_issues_found = 0

        for file_path in all_files:
            tote_id = get_tote_id(file_path.name)
            is_issue = (file_path.stem in bad_filenames or file_path.name in bad_filenames)
            
            if is_issue:
                tote_data[tote_id]['issue'].append(file_path)
                total_issues_found += 1
            else:
                tote_data[tote_id]['good'].append(file_path)

        # 4. Analyseer Totes
        good_totes_ids = [tid for tid, data in tote_data.items() if len(data['good']) > 0]
        
        # --- CHECK 1: Zijn er überhaupt issues? ---
        if total_issues_found == 0:
            skipped_no_issues += 1
            continue

        # --- CHECK 2: Is er genoeg goede data? ---
        if len(good_totes_ids) < MIN_GOOD_TOTES:
            print(f"[OVERSLAAN] SKU {sku}: Te weinig unieke totes met goede data.")
            skipped_low_data += 1
            continue

        # Als we hier zijn, gaan we verwerken
        processed_count += 1
        print(f"[VERWERKEN] SKU {sku}")

        # Maak mappen
        target_base = TARGET_DIR / sku
        dirs = {
            'train_good': target_base / "train" / "good",
            'test_good': target_base / "test" / "good",
            'test_issue': target_base / "test" / "issue",
            'ground_truth': target_base / "ground_truth"
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # 5. Verdeel de Totes (SORTEREN OP AANTAL GOEDE FOTO'S)
        # We sorteren ALLE totes. De tote met de MEESTE 'good' foto's komt op index 0.
        # Dit zorgt ervoor dat Train altijd de beste data krijgt.
        sorted_totes = sorted(good_totes_ids, key=lambda tid: len(tote_data[tid]['good']), reverse=True)
        
        # Bepaal het split punt (minimaal 1 voor test, minimaal 1 voor train)
        split_idx = int(len(good_totes_ids) * TRAIN_RATIO + 0.5)
        
        # Correctie voor kleine aantallen (bijv. 2 totes)
        if split_idx >= len(good_totes_ids):
            split_idx = len(good_totes_ids) - 1
        if split_idx == 0:
            split_idx = 1
            
        train_tote_ids = sorted_totes[:split_idx]
        test_tote_ids = sorted_totes[split_idx:]

        # Log de keuze voor zekerheid
        print(f"    -> Verdeling (Totaal {len(good_totes_ids)} totes):")
        print(f"       TRAIN: {len(train_tote_ids)} totes (Top count: {len(tote_data[train_tote_ids[0]]['good'])} good imgs)")
        if test_tote_ids:
            print(f"       TEST:  {len(test_tote_ids)} totes (Top count: {len(tote_data[test_tote_ids[0]]['good'])} good imgs)")

        # Zet om naar sets voor snelle lookup
        train_tote_set = set(train_tote_ids)

        # 6. Kopiëren & Statistieken
        count_good_train = 0
        count_good_test = 0
        count_issue_total = 0 
        
        # A. Goede foto's
        for tid in train_tote_ids:
            imgs = tote_data[tid]['good']
            count_good_train += len(imgs)
            for f in imgs:
                shutil.copy2(f, dirs['train_good'] / f.name)
                
        for tid in test_tote_ids:
            imgs = tote_data[tid]['good']
            count_good_test += len(imgs)
            for f in imgs:
                shutil.copy2(f, dirs['test_good'] / f.name)

        # B. Issue foto's
        issue_totes_count = 0
        for tid, data in tote_data.items():
            issues = data['issue']
            if not issues:
                continue
            
            issue_totes_count += 1
            count_issue_total += len(issues)
            
            if tid in train_tote_set:
                print(f"    [INFO] Tote {tid} zit in TRAIN (meeste good data) maar heeft ook {len(issues)} ISSUES (naar test).")
            
            for f in issues:
                shutil.copy2(f, dirs['test_issue'] / f.name)

        # 7. Data toevoegen aan statistieken
        summary_data.append({
            'sku': sku,
            'article_name': product_names.get(sku, "Unknown"),
            '#good training img': count_good_train,
            '#good test img': count_good_test,
            '#issue img': count_issue_total,
            '#good training totes': len(train_tote_ids),
            '#good test totes': len(test_tote_ids),
            '#issue totes': issue_totes_count
        })

    # 8. Schrijf CSV bestand
    print("-" * 60)
    print(f"Schrijven van statistieken naar {SUMMARY_FILE}...")
    
    csv_columns = [
        'sku', 'article_name', 
        '#good training img', '#good test img', '#issue img', 
        '#good training totes', '#good test totes', '#issue totes'
    ]
    
    try:
        with open(SUMMARY_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in summary_data:
                writer.writerow(data)
        print("CSV bestand succesvol aangemaakt.")
    except Exception as e:
        print(f"FOUT bij schrijven CSV: {e}")

    print("-" * 60)
    print(f"KLAAR!")
    print(f"Succesvol verwerkt: {processed_count} SKU's")
    print(f"Overgeslagen (Geen issues): {skipped_no_issues} SKU's")
    print(f"Overgeslagen (Te weinig good data): {skipped_low_data} SKU's")
    print(f"Locatie: {TARGET_DIR.absolute()}")

if __name__ == "__main__":
    main()