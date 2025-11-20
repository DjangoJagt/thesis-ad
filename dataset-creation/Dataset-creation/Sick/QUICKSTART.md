# Sick Dataset - Quick Start Guide

## Scripts Overzicht

### ✅ 01_build_staging.py - COMPLETED
**Doel:** Organiseer foto's per SKU en maak basis manifest

**Gebruik:**
```bash
# Dry-run (test zonder wijzigingen)
python scripts/01_build_staging.py --dry-run

# Echte run
python scripts/01_build_staging.py
```

**Output:**
- `sick_data_staging/{sku}/` - Foto's georganiseerd per SKU  
- `manifest.csv` - Basis metadata (2644 rows, 714 SKUs)

---

### 🔄 02_enrich_from_dwh.py - READY TO USE
**Doel:** Verrijk manifest met DWH data (product namen, complaints, shopper issues)

**Gebruik:**
```bash
# Met DWH (reële data)
python scripts/02_enrich_from_dwh.py

# Dry-run met DWH
python scripts/02_enrich_from_dwh.py --dry-run

# Zonder DWH (mock mode, voor testen)
python scripts/02_enrich_from_dwh.py --no-dwh --dry-run

# Force fresh queries (negeer cache)
python scripts/02_enrich_from_dwh.py --no-cache
```

**Vereisten:**
- ✅ SQL queries klaar in `sql/` folder
- ⚠️ Picnic modules moeten beschikbaar zijn
- ⚠️ DWH toegang nodig
- ⚠️ Controleer of `tote_id` in manifest overeenkomt met `stock_tote_barcode` in DWH

**Belangrijke Opmerking over Tote IDs:**
Je foto filenames gebruiken numerieke tote_ids (bijv. `4600051131`, `2600048683`).
De DWH queries gebruiken `stock_tote_barcode`.

**CHECK FIRST**: Zijn deze hetzelfde format? Test met:
```sql
SELECT DISTINCT stock_tote_barcode 
FROM edge.fulfilment_pick_events 
WHERE source_barcode IS NOT NULL 
LIMIT 10;
```

Als het format verschilt, moet je mogelijk een mapping maken.

**Output (nieuwe kolommen in manifest):**
- `product_name` - Naam van het product
- `complaint_flag` - Boolean: heeft quality complaint?
- `shopper_flag` - Boolean: heeft shopper issue gedetecteerd?
- `total_quality_issues` - Aantal quality issues
- `number_of_issues` - Aantal shopper issues

**Cache:**
- Queries worden gecached in `cache/` folder
- Gebruik `--no-cache` om fresh data te halen

---

## Folder Structuur

```
Sick/
├── scripts/
│   ├── 01_build_staging.py       ✅ DONE
│   └── 02_enrich_from_dwh.py     🔄 READY
├── sql/
│   ├── quality_issues.sql        ✅ Created
│   └── shopper_issues.sql        ✅ Created
├── cache/                        (auto-created)
│   ├── quality_issues.parquet
│   ├── shopper_issues.parquet
│   └── product_names.json
├── sick_data_staging/            ✅ Created
│   ├── {sku}/
│   │   └── *.png
├── manifest.csv                  ✅ Created
├── manifest.csv.backup          (auto-created by script 02)
├── PLAN.md
└── QUICKSTART.md
```

---

## Workflow

### Stap 1: Staging ✅ DONE
```bash
cd /home/picnic/Dataset-creation/Sick
python scripts/01_build_staging.py
```

### Stap 2: DWH Enrichment 🔄 NEXT

**2a. Check DWH toegang**
```bash
# Test of je toegang hebt tot DWH
# Vanuit je normale werk environment
```

**2b. Verify tote_id format**
Controleer of de tote_ids in je foto's matchen met stock_tote_barcode in DWH:
- Foto format: `20251114-4600051131-10575563-16-30-22.png`
- tote_id: `4600051131` (numeriek, 10 digits)
- DWH barcode: `???` (check dit!)

**2c. Run enrichment**
```bash
# Dry-run eerst
python scripts/02_enrich_from_dwh.py --dry-run

# Echte run
python scripts/02_enrich_from_dwh.py
```

**2d. Verify results**
```python
import pandas as pd
df = pd.read_csv('manifest.csv')

print(f"Product names filled: {df['product_name'].notna().sum()}")
print(f"Complaints: {df['complaint_flag'].sum()}")
print(f"Shopper issues: {df['shopper_flag'].sum()}")
```

### Stap 3: Label Analysis (TODO)
Na enrichment, analyseer de data en suggereer labels.

---

## Troubleshooting

### Issue: Picnic modules not found
**Symptoom:** `ModuleNotFoundError: No module named 'picnic'`

**Oplossing:**
- Run script vanuit je normale werk environment waar picnic modules beschikbaar zijn
- OF gebruik `--no-dwh` flag om mock mode te testen

### Issue: No DWH access
**Symptoom:** Connection errors

**Oplossing:**
- Check VPN verbinding
- Verify config file: `config/config-nl-local.yml`
- Test met een simpele DWH query eerst

### Issue: Tote IDs don't match
**Symptoom:** Geen matches na enrichment, 0% coverage

**Oplossing:**
1. Check sample van beide formaten:
   ```sql
   -- DWH format
   SELECT DISTINCT source_barcode FROM edge.fulfilment_pick_events LIMIT 10;
   ```
   ```python
   # Manifest format
   import pandas as pd
   df = pd.read_csv('manifest.csv')
   print(df['tote_id'].head(10))
   ```

2. Als formaten verschillen, pas mapping aan in script of SQL

### Issue: Queries te langzaam
**Oplossing:**
- Gebruik cache (`--no-cache` alleen als je fresh data wil)
- Beperk date range indien mogelijk
- Optimize SQL queries (add indexes, filters)

---

## Next Steps

1. ✅ Validate staging folder - DONE
2. 🔄 Run DWH enrichment - IN PROGRESS
3. ⏳ Analyze enriched data
4. ⏳ Create label suggestions
5. ⏳ Manual review & labeling
6. ⏳ Train/test split
7. ⏳ Build final dataset

---

## Tips

- **Backup je data:** Scripts maken automatisch backups, maar check ze!
- **Start klein:** Test eerst met een subset van SKUs
- **Cache is je vriend:** DWH queries kunnen langzaam zijn, cache bespaart tijd
- **Dry-run altijd eerst:** Test wijzigingen voordat je ze echt uitvoert
- **Check statistics:** Kijk altijd naar de output statistics om unexpected results te spotten
