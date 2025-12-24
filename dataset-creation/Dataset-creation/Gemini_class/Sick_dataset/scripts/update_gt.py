import pandas as pd
from pathlib import Path

# --- CONFIG ---
ROOT_DIR = Path(__file__).parent.parent  # Goes up to Sick_dataset/
csv_input = ROOT_DIR / "manifest_analyzed_leakage_gpt5.1_nozoom.csv"
issues_file = ROOT_DIR / "issues.txt"
csv_output = ROOT_DIR / "manifest_analyzed_gt_leakage_gpt5.1_nozoom.csv"

def update_gt():
    # 1. Laden van de data
    if not csv_input.exists():
        print(f"Error: {csv_input} niet gevonden.")
        return
    
    df = pd.read_csv(csv_input)

    # 2. Flagged images in een set zetten voor snelle lookup
    flagged_names = set()
    if issues_file.exists():
        with open(issues_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[1]
                    # Zorg dat de extensie matcht met de CSV (waarschijnlijk .png)
                    if not name.lower().endswith('.png'):
                        name += '.png'
                    flagged_names.add(name)
    
    # 3. GT kolom invullen
    df['GT'] = df['filename'].apply(
        lambda x: 'ANOMALOUS' if x in flagged_names else 'NORMAL'
    )

    # 4. Vergelijkingskolom maken (is_correct)
    # We zorgen dat we eventuele spaties of hoofdletterverschillen negeren
    df['ai_status'] = df['ai_status'].fillna('UNKNOWN').str.upper().str.strip()
    df['is_correct'] = df['ai_status'] == df['GT']

    # 5. Opslaan
    df.to_csv(csv_output, index=False)

    # --- STATISTIEKEN ---
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("-" * 30)
    print(f"Analyse voltooid voor {total} foto's.")
    print(f"AI Nauwkeurigheid: {accuracy:.2f}%")
    print("-" * 30)
    
    # Confusion Matrix weergave
    print("\nVerdeling:")
    print(pd.crosstab(df['GT'], df['ai_status'], margins=True))
    print(f"\nBestand opgeslagen als: {csv_output}")

if __name__ == "__main__":
    update_gt()