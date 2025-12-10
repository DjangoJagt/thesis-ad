import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# --- INSTELLINGEN ---
SOURCE_DIR = Path("sick_data_staging")       # Waar je foto's nu staan
TARGET_DIR = Path("sick_data_issues_2")        # Waar de nieuwe structuur moet komen
ISSUE_LIST_FILE = "flagged_quality_issues.txt"

# MINIMUM AANTAL GOEDE FOTO'S
# Als een SKU minder dan dit aantal 'goede' foto's heeft, wordt hij overgeslagen.
# Voor anomaly detection wil je er minimaal 6-10 hebben (zodat je train/test kunt splitten).
MIN_GOOD_IMAGES = 3

def main():
    # 1. Lees het issue bestand in
    print(f"Lezen van {ISSUE_LIST_FILE}...")
    issues_map = defaultdict(set)
    
    skipped_low_data = 0
    processed_count = 0
    
    try:
        with open(ISSUE_LIST_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Sla headers en lege regels over
                if not line or line.startswith("[source") or "Folder name" in line: 
                    continue 
                
                parts = line.split('\t') 
                if len(parts) < 2:
                    parts = line.split()
                
                if len(parts) >= 2:
                    sku = parts[0].strip()
                    filename = parts[1].strip()
                    issues_map[sku].add(filename)
    except FileNotFoundError:
        print(f"FOUT: Kan {ISSUE_LIST_FILE} niet vinden.")
        return

    print(f"{len(issues_map)} SKU's gevonden in de issue-lijst.")
    print("-" * 60)

    # 2. Ga door elke SKU heen
    for sku, bad_filenames in issues_map.items():
        src_sku_path = SOURCE_DIR / sku
        
        if not src_sku_path.exists():
            # Dit gebeurt soms als XnView data heeft van een map die je al verplaatst/hernoemd hebt
            # print(f"Waarschuwing: Bronmap {sku} niet gevonden.")
            continue

        # 3. Inventariseer bestanden
        all_files = [f for f in src_sku_path.iterdir() if f.is_file()]
        
        issue_files = []
        good_files = []

        for file_path in all_files:
            # Check op bestandsnaam (zonder extensie is veiliger)
            if file_path.stem in bad_filenames or file_path.name in bad_filenames:
                issue_files.append(file_path)
            else:
                good_files.append(file_path)

        # --- DE FILTER CHECK ---
        # Als er te weinig goede foto's zijn, sla deze SKU over
        if len(good_files) < MIN_GOOD_IMAGES:
            print(f"[OVERSLAAN] SKU {sku}: Te weinig goede data.")
            print(f"    -> Issues: {len(issue_files)}, Goed: {len(good_files)} (Minimaal nodig: {MIN_GOOD_IMAGES})")
            skipped_low_data += 1
            continue

        # Als we hier zijn, is de SKU goed genoeg om te verwerken
        print(f"[VERWERKEN] SKU {sku}...")
        processed_count += 1

        # Maak mappenstructuur
        target_base = TARGET_DIR / sku
        dirs_to_create = [
            target_base / "ground_truth",
            target_base / "test" / "issue",
            target_base / "test" / "good",
            target_base / "train" / "good"
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

        # Kopiëren issues
        for f in issue_files:
            shutil.copy2(f, target_base / "test" / "issue" / f.name)

        # Verdelen good files (Train/Test split)
        # We zetten iets meer in Train (bijv 70%) of 50/50. Hieronder staat 50/50.
        random.shuffle(good_files)
        
        # Optie: Als je liever 70% training data hebt, verander // 2 naar int(len(good_files) * 0.7)
        split_idx = len(good_files) // 2 
        
        # Zorg dat er altijd minimaal 1 in test en 1 in train zit als het krap is
        if split_idx == 0 and len(good_files) > 1:
            split_idx = 1

        train_good = good_files[:split_idx]
        test_good = good_files[split_idx:]

        for f in train_good:
            shutil.copy2(f, target_base / "train" / "good" / f.name)
            
        for f in test_good:
            shutil.copy2(f, target_base / "test" / "good" / f.name)

    print("-" * 60)
    print(f"KLAAR!")
    print(f"Succesvol verwerkt: {processed_count} SKU's")
    print(f"Overgeslagen (te weinig data): {skipped_low_data} SKU's")
    print(f"Locatie: {TARGET_DIR.absolute()}")

if __name__ == "__main__":
    main()