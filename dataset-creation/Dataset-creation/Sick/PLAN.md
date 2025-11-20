# Sick Data - Computer Vision Dataset Plan van Aanpak

## Doel
Een gestructureerde dataset bouwen voor anomalie detectie in stock totes met meerdere producten in verschillende oriëntaties.

## Huidige Status
- ✅ **2644 foto's** verzameld van **714 SKUs** uit **1514 totes**
- ✅ Datum range: 2025-11-14 tot 2025-11-17
- ✅ Staging folder structuur per SKU
- ✅ Manifest.csv met metadata

## Folder Structuur (Doelstelling)

```
sick_data/
├── {sku}/
│   ├── train/
│   │   └── good/           # Normale, goede afbeeldingen
│   └── test/
│       ├── good/           # Test set normale afbeeldingen
│       ├── issue/          # Test set met quality issues
│       └── ground_truth/   # Handmatig geverifieerde labels
└── manifest.csv            # Centrale metadata file
```

## Stappen Plan

### ✅ Stap 1: Staging & Basis Manifest (DONE)
**Script:** `01_build_staging.py`
- [x] Foto's sorteren per SKU
- [x] Filename parsing (datum, tote_id, sku, tijd)
- [x] Basis manifest.csv aanmaken
- [x] Validatie van data kwaliteit

**Output:** 
- `sick_data_staging/{sku}/` folders
- `manifest.csv` met basis metadata

---

### 🔄 Stap 2: DWH Data Verrijking (NEXT)
**Script:** `02_enrich_from_dwh.py`

**Doelen:**
1. **Expected Quantity** ophalen per tote
   - Query: `stock_events` table met `tote_id` + `sku` + `timestamp`
   - Toevoegen aan manifest: `expected_qty` kolom
   
2. **Complaint Flags** linken
   - Query: `customer_complaints` table met `tote_id` of `sku`
   - Boolean: heeft deze tote/sku een klacht gehad?
   - Toevoegen aan manifest: `complaint_flag` kolom
   
3. **Shopper Flags** linken
   - Query: `shopper_reports` table met `tote_id`
   - Boolean: heeft een shopper dit gemarkeerd?
   - Toevoegen aan manifest: `shopper_flag` kolom

4. **Product Namen** (optioneel)
   - Query: `products` table met `sku`
   - Makkelijker voor interpretatie
   - Toevoegen aan manifest: `product_name` kolom

**Overwegingen:**
- **Privacy:** Zorg dat alleen aggregated/anonymized data gebruikt wordt
- **Performance:** Batch queries voor efficiency
- **Caching:** Cache DWH results om herhaalde queries te vermijden

**Output:** 
- Verrijkte `manifest.csv` met DWH data

---

### Stap 3: Labeling Strategie Bepalen
**Script:** `03_analyze_and_suggest_labels.py`

**Doelen:**
1. **Automatische Suggesties** genereren o.b.v.:
   - `complaint_flag == True` → waarschijnlijk `issue`
   - `shopper_flag == True` → waarschijnlijk `issue`
   - Beide `False` → waarschijnlijk `good`
   
2. **Statistieken** per SKU:
   - Hoeveel foto's per SKU?
   - Verdeling good/issue/unknown
   - Welke SKUs hebben genoeg data voor training?

3. **Sampling Strategie**:
   - Welke foto's moeten handmatig gereviewd worden?
   - Prioriteit aan edge cases (uncertain labels)

**Output:**
- Analyse rapport
- Suggested labels in manifest: `label_suggested` kolom
- Review lijst voor manual labeling

---

### Stap 4: Manual Review & Labeling
**Script:** `04_label_interface.py` (optioneel GUI) of manual CSV edit

**Doelen:**
1. Review van uncertain cases
2. Ground truth voor test set
3. Validatie van suggested labels

**Methoden:**
- **Eenvoudig:** Excel/CSV editing van manifest
- **Geavanceerd:** Simple Python GUI met image viewer
- **Best:** Labeling tool zoals LabelStudio/CVAT

**Output:**
- Manifest met `label` kolom (manual verified)
- Confidence scores voor labels

---

### Stap 5: Train/Test Split
**Script:** `05_create_splits.py`

**Strategie:**
- **Stratified per SKU:** Zorg voor goede verdeling per product
- **Temporal Split:** Oudere data → train, nieuwere → test?
- **Tote-level Split:** Foto's van zelfde tote in dezelfde set
- **Ratio:** 80/20 of 70/30 afhankelijk van data volume

**Criteria:**
- Minimaal X foto's per SKU voor training
- Balans good/issue in beide sets
- Representativiteit qua tijd/omstandigheden

**Output:**
- Manifest met `split` kolom (`train`/`test`)

---

### Stap 6: Final Dataset Assembly
**Script:** `06_build_final_dataset.py`

**Acties:**
1. Kopieer/symlink foto's naar finale structuur
2. Splits per `split` en `label`
3. Genereer dataset statistics
4. Creëer README per SKU
5. Validatie checks (geen lege folders, etc.)

**Output:**
- Finale `sick_data/` folder structuur
- Dataset README.md
- Statistics rapport

---

### Stap 7: Ground Truth Test Set
**Script:** `07_create_ground_truth.py`

**Alleen voor test set:**
- Extra zorgvuldige manual review
- Mogelijk meerdere reviewers
- Consensus labels
- Documentatie van edge cases

**Output:**
- `test/ground_truth/` folder met verified labels
- Annotatie notes

---

## Belangrijke Overwegingen

### Data Balans
- **Per SKU analyse:** Sommige SKUs hebben maar 1 foto
- **Minimum threshold:** Bijv. min 10 foto's per SKU voor training
- **Issue rate:** Waarschijnlijk weinig echte issues (imbalanced)
- **Augmentation:** Overweeg data augmentation voor minority class

### Labeling Criteria
**Wat is een "issue"?**
- Verkeerd aantal producten
- Beschadigde verpakking
- Verkeerde oriëntatie (afhankelijk van context)
- Vervuiling/contaminatie
- Product niet herkenbaar

**Wat is "good"?**
- Correct aantal (expected_qty)
- Alle producten zichtbaar
- Verpakking intact
- Normale variatie in oriëntatie is OK

### False Positives Vermijden
- **Unseen normal variations** zijn het gevaar
- Daarom: **breed spectrum** aan oriëntaties in training set
- Test op edge cases (bijv. producten half in beeld)

### Productie Overwegingen
- Welke precision/recall is acceptabel?
- Wat zijn de kosten van false positives vs false negatives?
- Real-time inference vereisten?

---

## Volgende Acties

1. **Review dit plan** - is deze aanpak logisch?
2. **DWH toegang** - heb je de queries klaar?
3. **Labeling criteria** - wat zijn de exacte definities?
4. **Resources** - hoeveel tijd voor manual labeling?

---

## Scripts Overzicht

| Script | Status | Beschrijving |
|--------|--------|--------------|
| `01_build_staging.py` | ✅ DONE | Staging folder + basis manifest |
| `02_enrich_from_dwh.py` | 📝 TODO | DWH data toevoegen |
| `03_analyze_and_suggest_labels.py` | 📝 TODO | Label suggesties + analyse |
| `04_label_interface.py` | 📝 OPTIONAL | Manual labeling tool |
| `05_create_splits.py` | 📝 TODO | Train/test split |
| `06_build_final_dataset.py` | 📝 TODO | Finale dataset structuur |
| `07_create_ground_truth.py` | 📝 TODO | Ground truth voor test |

---

## Notities
- Start klein: begin met subset van SKUs met veel data
- Itereer: eerste model trainen op subset, dan uitbreiden
- Documenteer: houd bij welke beslissingen je maakt en waarom
