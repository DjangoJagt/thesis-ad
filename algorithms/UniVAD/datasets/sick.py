import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json


class SickDataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        target_transform,
        aug_rate=-1,
        mode="train",
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        # load dataset
        self.meta_path = f"{root}/meta.json"
        self.cls_names = []
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.types = []

        # load meta info
        with open(self.meta_path, "r") as f:
            meta_info = json.load(f)

        meta_info = meta_info[self.mode]
        for k, v in meta_info.items():
            self.cls_names += [k] * len(v)
            for info in v:
                self.image_paths.append(os.path.join(self.root, info["img_path"]))
                self.mask_paths.append(os.path.join(self.root, info["mask_path"]))
                self.labels.append(info["anomaly"])
                self.types.append(info["specie_name"])

        self.length = len(self.image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        cls_name = self.cls_names[index]
        label = self.labels[index]
        mask_path = self.mask_paths[index]

        # Load image
        img = Image.open(image_path).convert("RGB")
        img_pil = img.copy()
        
        if self.transform is not None:
            img = self.transform(img)

        # For SICK data, we don't have pixel-level ground truth masks
        # Create a dummy mask (all zeros for normal, all ones for anomaly)
        if self.mode == "test":
            # Dummy mask - same size as transformed image
            import torch
            h, w = img.shape[1], img.shape[2]
            if label == 1:  # anomaly
                mask = torch.ones((1, h, w))
            else:  # normal
                mask = torch.zeros((1, h, w))
        else:
            mask = Image.new('L', img_pil.size, 0)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

        return {
            "img": img,
            "img_mask": mask,
            "img_path": image_path,
            "cls_name": cls_name,
            "anomaly": label,
            "img_pil": np.array(img_pil.resize((448, 448))),  # Convert to numpy array and resize
        }

    def get_cls_names(self):
        return list(set(self.cls_names))
