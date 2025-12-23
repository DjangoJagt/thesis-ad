import shutil
import random
from pathlib import Path
from collections import defaultdict

# ---- CONFIGURATION ----
SOURCE_DIR = Path("dataset_cropped")
ISSUES_FILE = Path("quality_issues.txt")

# These IDs will NOT have their issues split. They get the full set in both folders.
KEEP_WHOLE_IDS = ["11463862", "11478313"]

# Minimum number of good images to force into training (safety buffer)
MIN_TRAIN_GOOD = 5

def load_issues(filepath):
    issues_map = defaultdict(set)
    if not filepath.exists():
        return issues_map
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                issues_map[parts[0].strip()].add(parts[1].strip())
    return issues_map

def write_dataset(dataset_name, data_map):
    """
    Creates the folders and copies files for one dataset (Set 1 or Set 2).
    data_map structure: { 'article_id': {'issues': [paths], 'good': [paths]} }
    """
    target_dir = Path(dataset_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    print(f"\nGeneratin {dataset_name}...")
    print(f"{'Product ID':<15} | {'Issues':<8} | {'Test Good':<10} | {'Train Good':<10}")
    print("-" * 65)

    for article_id, files in data_map.items():
        issue_files = files['issues']
        good_files = files['good']

        # 1. Randomize Good Files (Fresh shuffle for this dataset)
        # We work on a copy so we don't affect the master list
        current_good = list(good_files)
        random.shuffle(current_good)

        # 2. Determine Split Sizes
        num_issues = len(issue_files)
        total_good = len(current_good)

        # Goal: Test Good == Test Issues
        target_test_good = num_issues
        
        # Constraint: Must leave at least MIN_TRAIN_GOOD for training
        max_allowed_test = total_good - MIN_TRAIN_GOOD
        
        # If we have very few good images, we might be forced to reduce test set
        if max_allowed_test < 0:
            max_allowed_test = 0 # Should not happen with your counts, but safe to add
            
        final_test_count = min(target_test_good, max_allowed_test)
        
        # Perform Split
        test_good_files = current_good[:final_test_count]
        train_good_files = current_good[final_test_count:]

        # 3. Create Folder Structure
        # Find original folder name (e.g., "10512371_Mayonaise")
        folder_name = f"{article_id}_Unknown"
        for d in SOURCE_DIR.iterdir():
            if d.name.startswith(article_id):
                folder_name = d.name
                break
        
        base_path = target_dir / folder_name
        (base_path / "ground_truth").mkdir(parents=True)
        (base_path / "test" / "issue").mkdir(parents=True)
        (base_path / "test" / "good").mkdir(parents=True)
        (base_path / "train" / "good").mkdir(parents=True)

        # 4. Copy Files
        for f in issue_files:
            shutil.copy2(f, base_path / "test" / "issue" / f.name)
        for f in test_good_files:
            shutil.copy2(f, base_path / "test" / "good" / f.name)
        for f in train_good_files:
            shutil.copy2(f, base_path / "train" / "good" / f.name)

        print(f"{article_id:<15} | {len(issue_files):<8} | {len(test_good_files):<10} | {len(train_good_files):<10}")

def main():
    issues_map = load_issues(ISSUES_FILE)
    
    # We will build two maps: one for Set 1, one for Set 2
    data_set_1 = {}
    data_set_2 = {}

    for product_folder in sorted(SOURCE_DIR.iterdir()):
        if not product_folder.is_dir(): continue
        
        article_id = product_folder.name.split('_')[0]
        all_images = list(product_folder.glob("*.jpg"))
        
        # Identify Issue vs Good
        bad_stems = issues_map.get(article_id, set())
        
        all_issues = []
        all_good = []
        
        for img in all_images:
            if img.stem in bad_stems:
                all_issues.append(img)
            else:
                all_good.append(img)

        # ---- SPLITTING STRATEGY ----
        
        if article_id in KEEP_WHOLE_IDS:
            # STRATEGY A: Keep Whole
            # Both sets get ALL issues
            issues_1 = all_issues
            issues_2 = all_issues
        else:
            # STRATEGY B: Split 50/50
            # Shuffle issues to ensure random distribution of defects
            random.shuffle(all_issues)
            mid_point = len(all_issues) // 2
            
            # Ensure at least 1 issue in both sets if possible
            if mid_point == 0 and len(all_issues) > 1:
                mid_point = 1
            
            issues_1 = all_issues[:mid_point]
            issues_2 = all_issues[mid_point:]

        # Assign to the data maps
        # Note: We pass 'all_good' to both. The write_dataset function 
        # will shuffle them independently later.
        data_set_1[article_id] = {'issues': issues_1, 'good': all_good}
        data_set_2[article_id] = {'issues': issues_2, 'good': all_good}

    # Generate the actual folders
    write_dataset("custom_dataset_1", data_set_1)
    write_dataset("custom_dataset_2", data_set_2)

if __name__ == "__main__":
    main()