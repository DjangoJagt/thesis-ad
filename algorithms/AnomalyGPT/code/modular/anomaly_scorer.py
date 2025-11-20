import torch
import numpy as np
from typing import Dict, Any

class FeatureBank:
    def __init__(self):
        self.global_feats = []
        self.mean = None
        self.cov_inv = None

    def add(self, feats: torch.Tensor):
        # feats: (N, D)
        self.global_feats.append(feats.cpu())

    def finalize(self):
        if not self.global_feats:
            return
        all_feats = torch.cat(self.global_feats, dim=0)
        self.mean = all_feats.mean(dim=0, keepdim=True)
        # Ledoit-Wolf style shrinkage covariance (simplified)
        X = all_feats - self.mean
        cov = (X.T @ X) / (X.shape[0] - 1)
        # Add small diagonal for stability
        cov = cov + torch.eye(cov.shape[0]) * 1e-5
        self.cov_inv = torch.linalg.inv(cov)

    def score(self, feat: torch.Tensor) -> Dict[str, Any]:
        # Cosine distance to mean
        feat_norm = feat / feat.norm(dim=-1, keepdim=True)
        mean_norm = self.mean / self.mean.norm(dim=-1, keepdim=True)
        cos_sim = (feat_norm @ mean_norm.T).squeeze()  # scalar
        cos_dist = 1 - cos_sim.item()
        # Mahalanobis distance
        diff = (feat - self.mean).squeeze().unsqueeze(0)  # (1,D)
        maha = torch.sqrt((diff @ self.cov_inv @ diff.T)).item()
        return {
            'cosine_distance': float(cos_dist),
            'mahalanobis': float(maha)
        }

class AdaptiveThreshold:
    def __init__(self, quantile: float = 0.90):
        self.quantile = quantile
        self.threshold = None

    def fit(self, distances):
        arr = np.array(distances)
        self.threshold = float(np.quantile(arr, self.quantile))

    def predict(self, distance: float) -> int:
        if self.threshold is None:
            return 0
        return 1 if distance > self.threshold else 0
