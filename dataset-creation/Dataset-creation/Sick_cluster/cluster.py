import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import transforms

# 1. SETUP
# ======================
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = SCRIPT_DIR / '10468098/train/good'
REP_FOLDER = IMAGE_FOLDER.with_name(f"{IMAGE_FOLDER.name}_cluster")
x_clusters = 10  # Change this to your desired number of representative images
# ======================

torch.manual_seed(42)
np.random.seed(42)

# ... (Imports remain the same) ...

# 1. LOAD DINOv2 MODEL
# ======================
# 'dinov2_vits14' is the smallest/fastest version (ViT-Small). 
# You can use 'dinov2_vitb14' (Base) or 'dinov2_vitl14' (Large) for even better features.
print("Loading DINOv2 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# DINOv2 expects slightly different preprocessing than standard ResNet
# It handles its own resizing internally usually, but standardizing helps.
preprocess = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448), # Crops center 448x 448
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(path):
    try:
        img = Image.open(path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.inference_mode():
            # DINOv2 output is a dictionary or direct tensor depending on version
            # We want the output of the [CLS] token or average pooling
            # The hub model usually returns the feature vector directly
            embedding = model(batch_t.to(device))

        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

# ... (The rest of the clustering code remains exactly the same) ...

# 2. EXTRACT FEATURES
# ======================
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

image_files = [
    IMAGE_FOLDER / f
    for f in os.listdir(IMAGE_FOLDER)
    if Path(f).suffix.lower() in valid_extensions
]

print(f"Found {len(image_files)} images. Extracting features...")

features = []
valid_files = []

for img_path in image_files:
    emb = get_embedding(img_path)
    if emb is not None:
        features.append(emb)
        valid_files.append(img_path)

if not features:
    raise ValueError("No valid images found to extract features from.")

features = np.array(features)

if len(features) < x_clusters:
    raise ValueError(
        f"Requested {x_clusters} clusters but only {len(features)} valid images were loaded."
    )

# 3. CLUSTERING & SELECTION
# ======================
print(f"Clustering into {x_clusters} groups...")
kmeans = KMeans(n_clusters=x_clusters, random_state=42, n_init=10)
kmeans.fit(features)

# Find the image closest to the center of each cluster
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)

# 4. RESULTS
# ======================
print("\n--- Representative Images ---")
selected_paths = [valid_files[i] for i in closest_indices]

for i, path in enumerate(selected_paths):
    print(f"Cluster {i+1} Representative: {path}")

# Save representatives alongside original folder
if REP_FOLDER.exists():
    print(f"Cleaning existing representative folder: {REP_FOLDER}")
    for child in REP_FOLDER.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
else:
    REP_FOLDER.mkdir(parents=True)

print(f"Copying representatives to {REP_FOLDER} ...")
for path in selected_paths:
    shutil.copy(path, REP_FOLDER / Path(path).name)

print("Done copying representatives.")