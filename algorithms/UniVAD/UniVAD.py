import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import cv2
from modules import DinoFeaturizer, LinearLayer

from models.clip_prompt import encode_text_with_prompt_ensemble
from utils.filter_algorithm import filter_bg_noise
from utils.sampler import GreedyCoresetSampler
from utils.crf import dense_crf
import models.clip as open_clip
import os

from models.component_feature_extractor import ComponentFeatureExtractor
from models.component_segmentaion import (
    split_masks_from_one_mask,
    split_masks_from_one_mask_torch,
    split_masks_from_one_mask_with_bg
)

from matplotlib import pyplot as plt
from PIL import Image
from enum import Enum

device = "cuda" if torch.cuda.is_available() else "cpu"


class object_type(Enum):
    TEXTURE = 0
    SINGLE = 1
    MULTI = 2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]


def get_heatmaps(img, query_feature, net, color_tensor):
    with torch.no_grad():
        # feats1, f1_lowdim = net(img.cuda())
        feats1, f1_lowdim = net(img.to(device))

    sfeats1 = query_feature
    attn_intra = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1)
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)
    heatmap_intra = (
        F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True)
        .squeeze(0)
        .detach()
        .cpu()
    )
    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf, heatmap_intra)
    heatmap_intra = torch.from_numpy(crf_result)
    heatmap = heatmap_intra.argmax(dim=0)
    return heatmap, heatmap_intra


def see_image(data, heatmap, savepath, heatmap_intra):
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = np.clip((data * i_std + i_m) * 255, 0, 255).astype(np.uint8)
    data = data.transpose(1, 2, 0)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{savepath}/img.jpg", data)

    for i in range(heatmap_intra.shape[0]):
        heat = heatmap_intra[i, :, :].cpu().numpy()
        heat = np.round(heat * 128).astype(np.uint8)
        cv2.imwrite(f"{savepath}/heatresult{i}.jpg", heat)

class UniVAD(nn.Module):

    def __init__(self, image_size=224, lightweight: bool = False, enable_cfa: bool = False, force_texture: bool = False, masks_path: str = "./masks", data_path: str = "./data") -> None:
        super().__init__()
        # lightweight mode reduces memory by using a smaller CLIP and skipping DINO/component-heavy pieces
        self.lightweight = lightweight
        self.enable_cfa = enable_cfa  # allow toggling CFA at runtime for testing
        self.force_texture = force_texture  # force TEXTURE mode even when masks available
        self.masks_path = masks_path  # configurable masks directory path
        self.data_path = data_path  # configurable data directory path

        clip_name = "ViT-L-14-336"
        if self.lightweight:
            # smaller CLIP to reduce memory footprint on CPU devices
            clip_name = "ViT-B-16" # "ViT-B-32" 
        self.image_size = image_size
        pretrained = "openai"
        # device = torch.device("cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Adjust output layers based on model depth
        # ViT-L/14: 24 layers, ViT-B: 12 layers
        if self.lightweight:
            # ViT-B/32 has 12 layers, use every 3rd layer
            self.out_layers = [3, 6, 9, 12]
        else:
            # ViT-L/14 has 24 layers, use every 6th layer
            self.out_layers = [6, 12, 18, 24]

        # create CLIP model (smaller variant in lightweight mode)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, self.image_size, pretrained=pretrained
        )  # CLIP

        # load DINO models: full (heavy) vs lightweight (smaller backbone)
        if not self.lightweight:
            # full mode: keep original heavy components
            self.dino_net = DinoFeaturizer()
            self.dinov2_net = torch.hub.load(
                "./models/dinov2", "dinov2_vitg14", pretrained=True, source="local"
            ).to(device)
            # CFA can be toggled via enable_cfa flag
            self.cfa = CFA() if self.enable_cfa else None
        else:
            # lightweight mode: attempt to load the smaller DINOv2 (ViT-S/14) + optional dino_net
            # This increases RAM compared to pure-skip, but is much smaller than ViT-G/14
            # and enables MULTI gate support for multi-component objects.
            try:
                self.dino_net = DinoFeaturizer()
                self.dinov2_net = torch.hub.load(
                    "./models/dinov2", "dinov2_vits14", pretrained=True, source="local"
                ).to(device)
            except Exception as e:
                # if loading fails, fall back to no DINO; keep going
                print(f"[Warning] Failed to load DINO models in lightweight mode: {e}")
                self.dinov2_net = None
                self.dino_net = None
            # component fusion/feature aggregator can be toggled in lightweight mode too
            self.cfa = CFA() if self.enable_cfa else None

        self.clip_model.to(device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.device = device

        # Store CLIP patch grid size for mask interpolation
        # ViT-L/14: grid_size=32 (448/14), ViT-B/32: grid_size=14 (448/32)
        if hasattr(self.clip_model.visual, 'grid_size'):
            grid = self.clip_model.visual.grid_size
            # grid_size can be an int or tuple (h, w) - convert to int if square
            self.clip_patch_grid_size = grid[0] if isinstance(grid, (tuple, list)) else grid
        else:
            # fallback: infer from image_size and model name
            if 'L-14' in clip_name:
                patch_size = 14
            elif '16' in clip_name:  # <--- ADD THIS CHECK
                patch_size = 16
            else:
                patch_size = 32      # Default for ViT-B/32
            
            self.clip_patch_grid_size = self.image_size // patch_size

        # Store CLIP embedding dimension
        # ViT-L/14: 1024-dim, ViT-B/32: 768-dim, ViT-B/16: 768-dim
        if hasattr(self.clip_model.visual, 'width'):
            # Width is the vision transformer width (actual embedding dim before projection)
            self.clip_embed_dim = self.clip_model.visual.width
        elif hasattr(self.clip_model.visual, 'output_dim'):
            self.clip_embed_dim = self.clip_model.visual.output_dim
        elif hasattr(self.clip_model, 'embed_dim'):
            self.clip_embed_dim = self.clip_model.embed_dim
        else:
            # fallback: infer from model name
            self.clip_embed_dim = 1024 if 'L-14' in clip_name else 768

        # record DINO embedding dimension when dinov2 model is present
        if getattr(self, "dinov2_net", None) is not None:
            # dinov2 models expose `embed_dim` (vision transformer embed dim)
            self.dino_embed_dim = getattr(self.dinov2_net, "embed_dim", None)
            if self.dino_embed_dim is None:
                # fallback to num_features if present
                self.dino_embed_dim = getattr(self.dinov2_net, "num_features", None)
        else:
            self.dino_embed_dim = None

        self.transform_clip = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

        self.transform_dino = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
        )

        self.just_resize = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
            ],
        )

        transform_ce_clip = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        transform_ce_dino = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD
                ),
            ]
        )


        self.config = {}
        self.config["com_config"] = {}
        self.config["com_config"]["transform_clip"] = transform_ce_clip
        self.config["com_config"]["transform_dino"] = transform_ce_dino


        # component extractor: always create (needed for SINGLE/MULTI gates even when CFA disabled)
        # CFA flag only controls the aggregation module, not component extraction itself
        self.component_feature_extractor = ComponentFeatureExtractor(
            self.config["com_config"], clip_model=self.clip_model, dino_model=self.dinov2_net
        )

        self.decoder = LinearLayer()

        with torch.no_grad():
            self.text_prompts = encode_text_with_prompt_ensemble(
                self.clip_model, ["object"], self.tokenizer, self.device
            )

    def forward(
        self, batch: torch.Tensor, image_path, image_pil=None
    ) -> dict[str, torch.Tensor]:
        clip_transformed_image = self.transform_clip(batch)
        dino_transformed_image = self.transform_dino(batch)

        # lightweight forward fast-path: only when gate is TEXTURE (no component aggregation)
        if self.lightweight and (not hasattr(self, "gate") or self.gate == object_type.TEXTURE):
            with torch.no_grad():
                image_features, patch_tokens = self.clip_model.encode_image(
                    clip_transformed_image, self.out_layers
                )

                image_features = image_features[:, 0, :]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_tokens = self.decoder(patch_tokens)
                
                # Auto-detect actual CLIP embedding dimension from decoded tokens
                if not hasattr(self, '_clip_dim_detected'):
                    actual_dim = patch_tokens[0].shape[-1]
                    if actual_dim != self.clip_embed_dim:
                        print(f"[WARNING] Detected CLIP embedding mismatch: expected {self.clip_embed_dim}, got {actual_dim}. Using {actual_dim}.")
                        self.clip_embed_dim = actual_dim
                    self._clip_dim_detected = True
                
                text_features = self.text_prompts["object"]

            # simple global score using any precomputed normal image features when available
            if hasattr(self, "normal_image_features") and self.normal_image_features is not None:
                global_score = (
                    1
                    - (image_features @ self.normal_image_features.transpose(-2, -1))
                    .max()
                    .item()
                )
            else:
                global_score = 0.0

            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue
                # infer spatial grid and channel dim from tokens to support different CLIP variants
                B, L, C = patch_tokens[i].shape
                patch_tokens_reshaped = patch_tokens[i].reshape(B * L, 1, C)
                normal_tokens_reshaped = self.normal_patch_tokens[i].reshape(1, -1, C)
                cosine_similarity_matrix = F.cosine_similarity(
                    patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                )
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                sims.append(sim_max)
            # average sims across layers and reshape back to spatial grid inferred from L
            sim = torch.mean(torch.stack(sims, dim=0), dim=0)
            grid = int(np.sqrt(L))
            sim = sim.reshape(1, 1, grid, grid)
            sim = F.interpolate(
                sim, size=self.image_size, mode="bilinear", align_corners=True
            )
            anomaly_map_ret = 1 - sim

            # attempt to use a small dinov2 (ViT-S/14) in lightweight mode if it was loaded
            if getattr(self, "dinov2_net", None) is not None and getattr(self, "normal_dino_patches", None) is not None:
                dino_patch_tokens = self.dinov2_net.forward_features(
                    dino_transformed_image
                )["x_norm_patchtokens"]
                dino_dim = self.dino_embed_dim if getattr(self, "dino_embed_dim", None) is not None else 1536
                dino_patch_tokens_reshaped = dino_patch_tokens.reshape(-1, 1, dino_dim)
                dino_normal_tokens_reshaped = self.normal_dino_patches.reshape(1, -1, dino_dim)
                cosine_similarity_matrix = F.cosine_similarity(
                    dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                )
                sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
                sim_max_dino = sim_max_dino.reshape(
                    1, 1, int(self.image_size / 14), int(self.image_size / 14)
                )
                sim_max_dino = F.interpolate(
                    sim_max_dino, size=self.image_size, mode="bilinear", align_corners=True
                )
                anomaly_map_ret_dino = 1 - sim_max_dino
            else:
                anomaly_map_ret_dino = torch.zeros_like(anomaly_map_ret)

            anomaly_map_vls = []
            for layer in range(len(patch_tokens)):
                if layer != 6 and layer != (len(patch_tokens) // 2):
                    # try to pick a layer similar to the original behaviour but be robust
                    continue

                # keep CLIP text-based map computation (cheap)
                patch_tokens[layer] = patch_tokens[layer] @ self.clip_model.visual.proj
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                    dim=-1, keepdim=True
                )
                anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
                B, L, C = anomaly_map_vl.shape
                H = int(np.sqrt(L))
                anomaly_map_vl = F.interpolate(
                    anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
                anomaly_map_vl = (
                    anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                ) / 2
                anomaly_map_vls.append(anomaly_map_vl)
            if len(anomaly_map_vls) > 0:
                anomaly_map_vls = torch.mean(torch.stack(anomaly_map_vls, dim=0), dim=0).unsqueeze(1)
            else:
                anomaly_map_vls = torch.zeros_like(anomaly_map_ret)

            # in lightweight mode prefer the cheap texture path
            anomaly_map_ret_all = (anomaly_map_ret + anomaly_map_ret_dino + anomaly_map_vls) / 3
            if "HIS" in image_path:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.mean().item()),
                    "pred_mask": anomaly_map_ret_all,
                }
            else:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.max().item()),
                    "pred_mask": anomaly_map_ret_all,
                }

        # --- full mode (original behaviour) ---
        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                clip_transformed_image, self.out_layers
            )

            image_features = image_features[:, 0, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            patch_tokens = self.decoder(patch_tokens)
            dino_patch_tokens = self.dinov2_net.forward_features(
                dino_transformed_image
            )["x_norm_patchtokens"]
            text_features = self.text_prompts["object"]

        global_score = (
            1
            - (image_features @ self.normal_image_features.transpose(-2, -1))
            .max()
            .item()
        )

        sims = []
        for i in range(len(patch_tokens)):
            if i % 2 == 0:
                continue
            patch_tokens_reshaped = patch_tokens[i].view(
                self.clip_patch_grid_size ** 2, 1, self.clip_embed_dim
            )
            normal_tokens_reshaped = self.normal_patch_tokens[i].reshape(1, -1, self.clip_embed_dim)
            cosine_similarity_matrix = F.cosine_similarity(
                patch_tokens_reshaped, normal_tokens_reshaped, dim=2
            )
            sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
            sims.append(sim_max)
        sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(
            1, 1, self.clip_patch_grid_size, self.clip_patch_grid_size
        )
        sim = F.interpolate(
            sim, size=self.image_size, mode="bilinear", align_corners=True
        )
        anomaly_map_ret = 1 - sim

        # use dynamic DINO embedding dimension (supports ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14)
        dino_dim = self.dino_embed_dim if getattr(self, "dino_embed_dim", None) is not None else 1536
        dino_patch_tokens_reshaped = dino_patch_tokens.reshape(-1, 1, dino_dim)
        dino_normal_tokens_reshaped = self.normal_dino_patches.reshape(1, -1, dino_dim)
        cosine_similarity_matrix = F.cosine_similarity(
            dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
        )
        sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
        sim_max_dino = sim_max_dino.reshape(
            1, 1, int(self.image_size / 14), int(self.image_size / 14)
        )
        sim_max_dino = F.interpolate(
            sim_max_dino, size=self.image_size, mode="bilinear", align_corners=True
        )
        anomaly_map_ret_dino = 1 - sim_max_dino

        anomaly_map_vls = []
        for layer in range(len(patch_tokens)):

            # Use middle layer for text-based anomaly map (robust across different model sizes)
            if layer != (len(patch_tokens) // 2):
                continue

            patch_tokens[layer] = patch_tokens[layer] @ self.clip_model.visual.proj
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                dim=-1, keepdim=True
            )
            anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
            B, L, C = anomaly_map_vl.shape
            H = int(np.sqrt(L))
            anomaly_map_vl = F.interpolate(
                anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
            anomaly_map_vl = (
                anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
            ) / 2
            anomaly_map_vls.append(anomaly_map_vl)
        anomaly_map_vls = torch.mean(
            torch.stack(anomaly_map_vls, dim=0), dim=0
        ).unsqueeze(1)

        if self.gate == object_type.TEXTURE:

            anomaly_map_ret_all = (
                anomaly_map_ret + anomaly_map_ret_dino + anomaly_map_vls
            ) / 3

            if "HIS" in image_path:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.mean().item()),
                    "pred_mask": anomaly_map_ret_all,
                }
            else:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.max().item()),
                    "pred_mask": anomaly_map_ret_all,
                }

        # Construct mask path using configurable paths
        # Extract relative path from data_path
        # 1. Get dataset name (e.g. "sick") explicitly
        ds_name = os.path.basename(self.data_path.rstrip('/'))

        if self.data_path in image_path:
            relative_path = image_path.replace(self.data_path, "").lstrip("/")
        else:
            # Fallback: try to split by '/data/'
            parts = image_path.split('/data/')
            relative_path = parts[-1] if len(parts) > 1 else image_path
        
        query_sam_mask_path = (
            os.path.join(self.masks_path, ds_name, relative_path)
            .replace(".png", "/grounding_mask.png")
            .replace(".PNG", "/grounding_mask.png")
            .replace(".jpg", "/grounding_mask.png")
            .replace(".JPG", "/grounding_mask.png")
            .replace(".jpeg", "/grounding_mask.png")
            .replace(".JPEG", "/grounding_mask.png")
        )
        # print(f"[DEBUG] Image path: {image_path}")
        # print(f"[DEBUG] Data path: {self.data_path}")
        # print(f"[DEBUG] Relative path: {relative_path}")
        # print(f"[DEBUG] Mask path: {query_sam_mask_path}")
        query_tmp_mask = np.array(
            Image.open(query_sam_mask_path).resize((self.image_size, self.image_size))
        )
        query_sam_masks = split_masks_from_one_mask_torch(torch.tensor(query_tmp_mask))
        if len(query_sam_masks) == 0:
            query_sam_masks = [torch.ones((self.image_size, self.image_size))]

        if self.gate == object_type.SINGLE:

            anomaly_map_ret_part = torch.zeros(
                (1, 1, self.clip_patch_grid_size, self.clip_patch_grid_size)
            ).to(self.device)

            for query_sam_mask in query_sam_masks:
                H, W = query_sam_mask.shape
                kernel = np.ones((5, 5), np.uint8)
                query_sam_mask = cv2.dilate(
                    np.array(query_sam_mask), kernel, iterations=1
                )
                
                # CLIP mask: interpolate to CLIP patch grid size
                thresh_clip = torch.tensor(query_sam_mask).reshape(1, 1, H, W)
                thresh_clip = F.interpolate(
                    thresh_clip,
                    size=self.clip_patch_grid_size,
                    mode="bilinear",
                    align_corners=True,
                ).reshape(self.clip_patch_grid_size ** 2)
                thresh_clip[thresh_clip > 0] = 1
                
                # DINO mask: interpolate to DINO patch grid size
                thresh_dino = torch.tensor(query_sam_mask).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size // 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1

                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:
                        continue
                    patch_tokens_reshaped = patch_tokens[i].view(
                        self.clip_patch_grid_size ** 2, 1, self.clip_embed_dim
                    )[thresh_clip > 0]
                    normal_tokens_reshaped = self.normal_clip_part_patch_features[i][
                        0
                    ].reshape(1, -1, self.clip_embed_dim)
                    cosine_similarity_matrix = F.cosine_similarity(
                        patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                    )
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    # print(sim_max.max())
                    sims.append(sim_max)
                sim = torch.mean(torch.stack(sims, dim=0), dim=0)

                # DINO anomaly map uses DINO patch grid size (32x32 for 448px images)
                anomaly_map_ret_dino_part = torch.zeros(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                ).to(self.device)
                dino_dim = self.dino_embed_dim if getattr(self, "dino_embed_dim", None) is not None else 1536
                dino_patch_tokens_reshaped = dino_patch_tokens.reshape(-1, 1, dino_dim)[
                    thresh_dino > 0
                ]
                dino_normal_tokens_reshaped = self.normal_dino_part_patch_features[
                    0
                ].reshape(1, -1, dino_dim)
                cosine_similarity_matrix = F.cosine_similarity(
                    dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                )
                sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)

                thresh_clip_2d = thresh_clip.reshape(
                    (1, 1, self.clip_patch_grid_size, self.clip_patch_grid_size)
                )
                thresh_dino_2d = thresh_dino.reshape(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                )

                anomaly_map_ret_part[thresh_clip_2d > 0] += 1 - sim
                anomaly_map_ret_dino_part[thresh_dino_2d > 0] += 1 - sim_max_dino

            anomaly_map_ret_part = F.interpolate(
                anomaly_map_ret_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_ret_dino_part = F.interpolate(
                anomaly_map_ret_dino_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

            anomaly_map_ret_all = (
                (anomaly_map_ret + anomaly_map_ret_dino) / 2
                + (anomaly_map_ret_part + anomaly_map_ret_dino_part) / 2
                + anomaly_map_vls
            ) / 3

        if self.gate == object_type.MULTI:

            heatmap, heatmap_intra = get_heatmaps(
                dino_transformed_image,
                self.train_features_sampled,
                self.dino_net,
                self.color_tensor,
            )

            # query_sam_masks = torch.tensor(query_sam_masks)
            gs_masks = torch.stack(query_sam_masks)
            cluster_masks = torch.stack(split_masks_from_one_mask_torch(heatmap + 1))

            heatmap_refined = assign_fine_to_coarse_torch(cluster_masks, gs_masks)
            heatmap_refined = heatmap_refined.max(dim=0).values

            savepath = f"./heat_masks/{self.class_name}_heat/test/0"
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            cv2.imwrite(f"{savepath}/heatresult_refined.png", heatmap_refined.numpy())

            anomaly_map_dist = torch.zeros((1, 1, self.image_size, self.image_size)).to(
                self.device
            )

            query_mask_path = (
                f"./heat_masks/{self.class_name}_heat/test/0/heatresult_refined.png"
            )
            query_tmp_mask = cv2.imread(query_mask_path, cv2.IMREAD_GRAYSCALE)

            # query_tmp_mask = torch.tensor(query_tmp_mask)

            query_masks_capm, query_mask_idxs = split_masks_from_one_mask_with_bg(
                query_tmp_mask
            )

            # query_masks_capm = query_masks_capm

            query_masks, _ = split_masks_from_one_mask(query_tmp_mask)

            # query_masks = query_masks

            kernel = np.ones((5, 5), np.uint8)
            query_masks_capm = [
                cv2.dilate(mask, kernel, iterations=1) for mask in query_masks_capm
            ]
            query_masks = [
                cv2.dilate(mask, kernel, iterations=1) for mask in query_masks
            ]

            anomaly_map_ret_part = torch.zeros(
                (1, 1, self.clip_patch_grid_size, self.clip_patch_grid_size)
            ).to(self.device)
            anomaly_map_ret_part = 100 + anomaly_map_ret_part

            anomaly_map_ret_dino_part = torch.zeros(
                (1, 1, int(self.image_size / 14), int(self.image_size / 14))
            ).to(self.device)
            anomaly_map_ret_dino_part = 100 + anomaly_map_ret_dino_part

            for j in range(len(query_masks_capm)):
                query_sam_mask = query_masks_capm[j]
                H, W = query_sam_mask.shape
                
                # CLIP mask: interpolate to CLIP patch grid size
                thresh_clip = torch.tensor(query_sam_mask).reshape(1, 1, H, W)
                thresh_clip = F.interpolate(
                    thresh_clip,
                    size=self.clip_patch_grid_size,
                    mode="bilinear",
                    align_corners=True,
                ).reshape(self.clip_patch_grid_size ** 2)

                if thresh_clip.sum() < 1:
                    continue

                thresh_clip[thresh_clip > 0] = 1
                
                # DINO mask: interpolate to DINO patch grid size
                thresh_dino = torch.tensor(query_sam_mask).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size // 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1

                if self.normal_dino_part_patch_features[query_mask_idxs[j]] == []:
                    continue

                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:  # (layer+1)//2!=0:
                        continue
                    patch_tokens_reshaped = patch_tokens[i].view(
                        self.clip_patch_grid_size ** 2, 1, self.clip_embed_dim
                    )[thresh_clip > 0]
                    normal_tokens_reshaped = self.normal_clip_part_patch_features[i][
                        query_mask_idxs[j]
                    ].reshape(1, -1, self.clip_embed_dim)
                    cosine_similarity_matrix = F.cosine_similarity(
                        patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                    )
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)
                sim = torch.mean(torch.stack(sims, dim=0), dim=0)

                dino_dim = self.dino_embed_dim if getattr(self, "dino_embed_dim", None) is not None else 1536
                dino_patch_tokens_reshaped = dino_patch_tokens.reshape(-1, 1, dino_dim)[
                    thresh_dino > 0
                ]
                dino_normal_tokens_reshaped = self.normal_dino_part_patch_features[
                    query_mask_idxs[j]
                ].reshape(1, -1, dino_dim)
                cosine_similarity_matrix = F.cosine_similarity(
                    dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                )
                sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
                
                thresh_clip_2d = thresh_clip.reshape(
                    (1, 1, self.clip_patch_grid_size, self.clip_patch_grid_size)
                )
                thresh_dino_2d = thresh_dino.reshape(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                )
                
                anomaly_map_ret_part[thresh_clip_2d > 0] = torch.min(
                    1 - sim, anomaly_map_ret_part[thresh_clip_2d > 0]
                )
                anomaly_map_ret_dino_part[thresh_dino_2d > 0] = torch.min(
                    1 - sim_max_dino, anomaly_map_ret_dino_part[thresh_dino_2d > 0]
                )

            anomaly_map_ret_part[anomaly_map_ret_part == 100] = 0
            anomaly_map_ret_dino_part[anomaly_map_ret_dino_part == 100] = 0

            anomaly_map_ret_part = F.interpolate(
                anomaly_map_ret_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_ret_dino_part = F.interpolate(
                anomaly_map_ret_dino_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

            if image_pil is not None:
                image = np.array(image_pil[0])
            else:
                image = np.array(
                    Image.open(image_path)
                    .convert("RGB")
                    .resize((self.image_size, self.image_size))
                )

            features = self.component_feature_extractor.extract(image, query_masks)

            query_component_feats = {
                "area": [],
                "color": [],
                "position": [],
                "clip_image": [],
                "dino_image": [],
                "geo": [],
            }

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                query_component_feats[feature_name].append(features[feature_name])

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                query_component_feats[feature_name] = torch.cat(
                    query_component_feats[feature_name], axis=0
                )

            query_component_feats["geo"] = torch.cat(
                [
                    query_component_feats["area"],
                    query_component_feats["color"],
                    query_component_feats["position"],
                ],
                dim=1,
            )
            query_component_feats["clip_image"] = query_component_feats[
                "clip_image"
            ].transpose(0, 1)

            # query_component_feats["dino_image"] = query_component_feats[
            #     "dino_image"
            # ].transpose(0, 1)

            # Apply CFA only if enabled
            if self.cfa is not None:
                for layer in range(query_component_feats["clip_image"].shape[0]):
                    query_component_feats["clip_image"][layer] = self.cfa(
                        query_component_feats["clip_image"][layer]
                    )

                query_component_feats["dino_image"] = self.cfa(
                    query_component_feats["dino_image"]
                )

            for mask_idx in range(len(query_masks)):
                query_mask = query_masks[mask_idx]
                thresh_ori = torch.tensor(query_mask).reshape(
                    1, 1, self.image_size, self.image_size
                )
                sim_clip_component = F.cosine_similarity(
                    query_component_feats["clip_image"][:, mask_idx, :]
                    .unsqueeze(1)
                    .unsqueeze(1),
                    self.normal_component_feats["clip_image"].unsqueeze(1),
                    dim=-1,
                )

                # print( query_component_feats["dino_image"][mask_idx].shape, self.normal_component_feats["dino_image"].shape)

                sim_dino_component = F.cosine_similarity(
                    query_component_feats["dino_image"][mask_idx].unsqueeze(0),
                    self.normal_component_feats["dino_image"],
                    dim=1,
                )

                # print(sim_dino_component.shape)

                sim_geo = F.cosine_similarity(
                    query_component_feats["geo"][mask_idx],
                    self.normal_component_feats["geo"].unsqueeze(0),
                    dim=2,
                )
                dist = torch.mean(
                    1 - sim_clip_component.max(dim=-1).values, dim=0
                ).item()
                dist += 1 - sim_dino_component.max().item()
                dist += 1 - sim_geo.max().item()
                anomaly_map_dist[thresh_ori > 0] += dist

            anomaly_map_ret_all = (
                (anomaly_map_ret + anomaly_map_ret_dino) / 2
                + (anomaly_map_ret_part + anomaly_map_ret_dino_part) / 2
                + anomaly_map_vls
            ) / 3 + anomaly_map_dist / 2

        return {
            "pred_score": torch.tensor(anomaly_map_ret_all.max().item() + global_score),
            "pred_mask": anomaly_map_ret_all,
        }

    def setup(self, data: dict, re_seg=True) -> None:

        few_shot_samples = data.get("few_shot_samples")
        self.class_name = data.get("dataset_category")
        image_paths = data.get("image_path")

        self.kernel = np.ones((20, 20), np.uint8)
        self.shot = len(few_shot_samples)

        # lightweight setup: compute CLIP-based normal features and optional DINO patches
        if self.lightweight:
            clip_transformed_normal_image = self.transform_clip(few_shot_samples).to(
                self.device
            )
            # if a small dinov2 model was loaded in lightweight mode, also compute its normal patches
            if getattr(self, "dinov2_net", None) is not None:
                dino_transformed_normal_image = self.transform_dino(few_shot_samples).to(
                    self.device
                )
            with torch.no_grad():
                self.normal_image_features, self.normal_patch_tokens = (
                    self.clip_model.encode_image(
                        clip_transformed_normal_image, self.out_layers
                    )
                )
                self.normal_image_features = self.normal_image_features[:, 0, :]
                self.normal_image_features = (
                    self.normal_image_features / self.normal_image_features.norm()
                )
                self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)
                
                # Auto-detect actual CLIP embedding dimension from decoded tokens
                # (output_dim gives projected dimension, but we need pre-projection width)
                actual_dim = self.normal_patch_tokens[0].shape[-1]
                if actual_dim != self.clip_embed_dim:
                    self.clip_embed_dim = actual_dim
                
                if getattr(self, "dinov2_net", None) is not None:
                    # compute and store normal dino patches for similarity comparisons
                    self.normal_dino_patches = self.dinov2_net.forward_features(
                        dino_transformed_normal_image
                    )["x_norm_patchtokens"]

            # Dynamic gate detection in lightweight mode (same as full mode)
            # Load grounded SAM masks to determine object type
            grounded_sam_mask_paths = []
            ds_name = os.path.basename(self.data_path.rstrip('/'))
            for image_path in image_paths:
                # Extract relative path from data_path
                if self.data_path in image_path:
                    relative_path = image_path.replace(self.data_path, "").lstrip("/")
                else:
                    # Fallback: try to split by '/data/'
                    parts = image_path.split('/data/')
                    relative_path = parts[-1] if len(parts) > 1 else image_path
                
                mask_path = (
                    os.path.join(self.masks_path, ds_name, relative_path)
                    .replace(".png", "/grounding_mask.png")
                    .replace(".PNG", "/grounding_mask.png")
                    .replace(".jpg", "/grounding_mask.png")
                    .replace(".JPG", "/grounding_mask.png")
                    .replace(".jpeg", "/grounding_mask.png")
                    .replace(".JPEG", "/grounding_mask.png")
                )
                grounded_sam_mask_paths.append(mask_path)
            
            # Try to load masks, fallback to TEXTURE mode if masks don't exist
            grounded_sam_masks = []
            masks_exist = all(os.path.exists(p) for p in grounded_sam_mask_paths)
            
            if masks_exist:
                try:
                    grounded_sam_masks = [
                        split_masks_from_one_mask_torch(
                            torch.tensor(
                                np.array(Image.open(x).resize((self.image_size, self.image_size)))
                            )
                        )
                        for x in grounded_sam_mask_paths
                    ]
                except Exception as e:
                    print(f"[Warning] Failed to load masks: {e}. Falling back to TEXTURE mode.")
                    grounded_sam_masks = []
            else:
                print(f"[Info] Grounding masks not found for {self.class_name}. Using TEXTURE mode (no component segmentation).")
                grounded_sam_masks = []

            # Determine gate type based on mask analysis
            if len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) > 0:
                H, W = grounded_sam_masks[0][0].shape
                object_ratio = (torch.sum(sorted(grounded_sam_masks[0], key=lambda x:torch.sum(x), reverse=True)[0]) / 255) / (H * W)
            else:
                object_ratio = 1

            # Force TEXTURE mode if requested (bypasses component-level detection)
            if self.force_texture:
                self.gate = object_type.TEXTURE
                print(f"[Info] Force TEXTURE mode enabled, skipping component-level features.")
            elif object_ratio > 0.65 and len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) <= 2:
                self.gate = object_type.TEXTURE
            elif len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) == 1:
                self.gate = object_type.SINGLE
            elif len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) > 1:
                self.gate = object_type.MULTI
            else:
                # No masks found or failed to load - default to TEXTURE mode
                self.gate = object_type.TEXTURE
                print(f"[Info] No valid masks for {self.class_name}, defaulting to TEXTURE gate")
            
            # Early return ONLY for TEXTURE gate (no component aggregation needed)
            if self.gate == object_type.TEXTURE:
                if getattr(self, "dinov2_net", None) is None:
                    self.normal_dino_patches = None
                # minimal placeholders so forward() won't break
                self.normal_dino_part_patch_features = []
                self.normal_clip_part_patch_features = []
                return
            # For SINGLE / MULTI gates in lightweight mode: we need DINO models
            # If they're not available, force TEXTURE mode
            if getattr(self, "dinov2_net", None) is None or getattr(self, "dino_net", None) is None:
                print(f"[Warning] SINGLE/MULTI gate detected but DINO models not available in lightweight mode. Forcing TEXTURE mode.")
                self.gate = object_type.TEXTURE
                self.normal_dino_part_patch_features = []
                self.normal_clip_part_patch_features = []
                return
            # Continue with full component preparation for SINGLE/MULTI gates

        clip_transformed_normal_image = self.transform_clip(few_shot_samples).to(
            self.device
        )
        dino_transformed_normal_image = self.transform_dino(few_shot_samples).to(
            self.device
        )

        self.part_num = {
            "breakfast_box": [4],
            "screw_bag": [3],
            "splicing_connectors": [2],
            "pushpins": [3],
            "juice_bottle": [4],
        }

        num_cluster = {
            "breakfast_box": 5,
            "screw_bag": 5,
            "splicing_connectors": 5,
            "pushpins": 5,
            "juice_bottle": 5,
        }

        color_list = [
            [0, 0, 0],
            [127, 123, 229],
            [195, 240, 251],
            [146, 223, 255],
            [243, 241, 230],
            [224, 190, 144],
            [178, 116, 75],
        ]
        color_tensor = torch.tensor(color_list)
        color_tensor = color_tensor[:, :, None, None]
        self.color_tensor = color_tensor.repeat(1, 1, self.image_size, self.image_size)

        grounded_sam_mask_paths = []
        ds_name = os.path.basename(self.data_path.rstrip('/'))
        for image_path in image_paths:
            # Extract relative path from data_path
            if self.data_path in image_path:
                relative_path = image_path.replace(self.data_path, "").lstrip("/")
            else:
                # Fallback: try to split by '/data/'
                parts = image_path.split('/data/')
                relative_path = parts[-1] if len(parts) > 1 else image_path
            
            mask_path = (
                os.path.join(self.masks_path, ds_name, relative_path)
                .replace(".png", "/grounding_mask.png")
                .replace(".PNG", "/grounding_mask.png")
                .replace(".jpg", "/grounding_mask.png")
                .replace(".JPG", "/grounding_mask.png")
                .replace(".jpeg", "/grounding_mask.png")
                .replace(".JPEG", "/grounding_mask.png")
            )
            grounded_sam_mask_paths.append(mask_path)
        
        # Try to load masks, fallback to TEXTURE mode if masks don't exist
        grounded_sam_masks = []
        masks_exist = all(os.path.exists(p) for p in grounded_sam_mask_paths)
        
        if masks_exist:
            try:
                grounded_sam_masks = [
                    split_masks_from_one_mask_torch(
                        torch.tensor(
                            np.array(Image.open(x).resize((self.image_size, self.image_size)))
                        )
                    )
                    for x in grounded_sam_mask_paths
                ]
            except Exception as e:
                print(f"[Warning] Failed to load masks: {e}. Falling back to TEXTURE mode.")
                grounded_sam_masks = []
        else:
            print(f"[Info] Grounding masks not found for {self.class_name}. Using TEXTURE mode (no component segmentation).")
            grounded_sam_masks = []

        if len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) > 0:
            H, W = grounded_sam_masks[0][0].shape
            object_ratio = (torch.sum(sorted(grounded_sam_masks[0], key=lambda x:torch.sum(x), reverse=True)[0]) / 255) / (H * W)
        else:
            object_ratio = 1

        # Force TEXTURE mode if requested (bypasses component-level detection)
        if self.force_texture:
            self.gate = object_type.TEXTURE
            print(f"[Info] Force TEXTURE mode enabled, skipping component-level features.")
        elif object_ratio > 0.65 and len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) <= 2:
            self.gate = object_type.TEXTURE
            print(f"[Info] Gate set to TEXTURE (object_ratio={object_ratio:.3f}, num_masks={len(grounded_sam_masks[0]) if len(grounded_sam_masks) > 0 else 0})")
        elif len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) == 1:
            self.gate = object_type.SINGLE
            print(f"[Info] Gate set to SINGLE (num_masks={len(grounded_sam_masks[0])})")
        elif len(grounded_sam_masks) > 0 and len(grounded_sam_masks[0]) > 1:
            self.gate = object_type.MULTI
            print(f"[Info] Gate set to MULTI (num_masks={len(grounded_sam_masks[0])})")
        else:
            # No masks found or failed to load - default to TEXTURE mode
            self.gate = object_type.TEXTURE
            print(f"[Info] No valid masks for {self.class_name}, defaulting to TEXTURE gate")



        with torch.no_grad():
            self.normal_image_features, self.normal_patch_tokens = (
                self.clip_model.encode_image(
                    clip_transformed_normal_image, self.out_layers
                )
            )
            self.normal_image_features = self.normal_image_features[:, 0, :]
            self.normal_image_features = (
                self.normal_image_features / self.normal_image_features.norm()
            )

            self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)
            
            # Auto-detect actual CLIP embedding dimension from decoded tokens
            # (output_dim gives projected dimension, but we need pre-projection width)
            actual_dim = self.normal_patch_tokens[0].shape[-1]
            if actual_dim != self.clip_embed_dim:
                self.clip_embed_dim = actual_dim
            
            self.normal_dino_patches = self.dinov2_net.forward_features(
                dino_transformed_normal_image
            )["x_norm_patchtokens"]

        self.normal_dino_part_patch_features = [[], [], [], [], [], [], [], [], [], []]

        self.normal_clip_part_patch_features = [
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
        ]

        if self.gate == object_type.SINGLE:

            for i in range(self.shot):
                normal_sam_mask = cv2.dilate(
                    np.array(grounded_sam_masks[i][0]), self.kernel, iterations=1
                )
                
                # CLIP mask: interpolate to CLIP patch grid size
                thresh_clip = torch.tensor(normal_sam_mask).reshape(1, 1, H, W)
                thresh_clip = F.interpolate(
                    thresh_clip,
                    size=self.clip_patch_grid_size,
                    mode="bilinear",
                    align_corners=True,
                ).reshape(self.clip_patch_grid_size ** 2)
                thresh_clip[thresh_clip > 0] = 1
                
                # DINO mask: interpolate to DINO patch grid size (always 32x32 for 448px images with /14 patches)
                thresh_dino = torch.tensor(normal_sam_mask).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size // 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1

                selected_patch_features = self.normal_dino_patches[i][thresh_dino]
                self.normal_dino_part_patch_features[0].append(selected_patch_features)

                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    selected_patch_features = self.normal_patch_tokens[layer][i][thresh_clip]
                    self.normal_clip_part_patch_features[layer][0].append(
                        selected_patch_features
                    )

            self.normal_dino_part_patch_features[0] = torch.cat(
                self.normal_dino_part_patch_features[0], dim=0
            )
            for layer in range(len(self.normal_patch_tokens)):
                if layer % 2 == 0:
                    continue
                self.normal_clip_part_patch_features[layer][0] = torch.cat(
                    self.normal_clip_part_patch_features[layer][0], dim=0
                )

        if self.gate == object_type.MULTI:

            if re_seg:
                train_feature_list = []
                # greedsampler_perimg = GreedyCoresetSampler(
                #     percentage=0.01, device="cuda"
                # )
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.01, device=device
                )


                for Img in dino_transformed_normal_image:
                    Img = Img.unsqueeze(0)
                    feats0, f_lowdim = self.dino_net(Img)
                    feats = feats0.squeeze()
                    feats = feats.reshape(feats0.shape[1], -1).permute(1, 0)
                    feats_sample = greedsampler_perimg.run(feats)
                    train_feature_list.append(feats_sample)

                train_features = torch.cat(train_feature_list, dim=0)
                train_features = F.normalize(train_features, dim=1)
                train_features = train_features.cpu().numpy()

                part_num = -1
                if self.class_name in self.part_num.keys():
                    part_num_right = self.part_num[self.class_name]
                    n_cluster = num_cluster[self.class_name]
                else:
                    part_num_right = [1]
                    n_cluster = 2

                while part_num not in part_num_right:
                    kmeans = KMeans(init="k-means++", n_clusters=n_cluster)
                    c = kmeans.fit(train_features)
                    cluster_centers = torch.from_numpy(c.cluster_centers_)
                    train_features_sampled = cluster_centers.to(self.device)
                    train_features_sampled = train_features_sampled.unsqueeze(
                        0
                    ).unsqueeze(0)
                    self.train_features_sampled = train_features_sampled.permute(
                        0, 3, 1, 2
                    )

                    for i, Img in enumerate(dino_transformed_normal_image):
                        Img = Img.unsqueeze(0)
                        heatmap, heatmap_intra = get_heatmaps(
                            Img,
                            self.train_features_sampled,
                            self.dino_net,
                            self.color_tensor,
                        )

                        savepath = f"./heat_masks/{self.class_name}_heat/train/{i}"
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)

                        gs_masks = torch.stack(grounded_sam_masks[i])
                        cluster_masks = torch.stack(
                            split_masks_from_one_mask_torch(heatmap + 1)
                        )
                        heatmap_refined = assign_fine_to_coarse_torch(
                            cluster_masks, gs_masks
                        )
                        heatmap_refined = heatmap_refined.max(dim=0).values

                        cv2.imwrite(
                            f"{savepath}/heatresult_refined.png",
                            heatmap_refined.detach().cpu().numpy(),
                        )
                        see_image(Img, heatmap, f"{savepath}", heatmap_intra)
                        part_num = len(filter_bg_noise("./heat_masks", self.class_name))

                        plt.clf()
                        plt.imshow(heatmap_refined.detach().cpu().numpy())
                        plt.savefig(savepath + "/" + "masks_color.png")

                    torch.save(
                        self.train_features_sampled,
                        f"./heat_masks/{self.class_name}_heat/train_features_sampled.pth",
                    )
            else:
                self.train_features_sampled = torch.load(
                    f"./heat_masks/{self.class_name}_heat/train_features_sampled.pth"
                )

            self.normal_component_feats = {
                "area": [],
                "color": [],
                "position": [],
                "clip_image": [],
                "dino_image": [],
                "geo": [],
            }

            for i in range(self.shot):
                image = np.array(
                    Image.open(image_paths[i])
                    .convert("RGB")
                    .resize((self.image_size, self.image_size))
                )
                normal_mask_path = f"./heat_masks/{self.class_name}_heat/train/{i}/heatresult_refined.png"
                normal_masks = cv2.imread(normal_mask_path, cv2.IMREAD_GRAYSCALE)

                normal_masks_capm, normal_mask_idxs = split_masks_from_one_mask_with_bg(
                    normal_masks
                )

                normal_masks, _ = split_masks_from_one_mask(normal_masks)

                normal_masks_capm = [
                    cv2.dilate(mask, self.kernel, iterations=1)
                    for mask in normal_masks_capm
                ]

                kernel = np.ones((5, 5), np.uint8)
                normal_masks = [
                    cv2.dilate(mask, kernel, iterations=1) for mask in normal_masks
                ]

                for j in range(len(normal_mask_idxs)):
                    # CLIP mask: interpolate to CLIP patch grid size
                    thresh_clip = torch.tensor(normal_masks_capm[j]).reshape(1, 1, H, W)
                    thresh_clip = F.interpolate(
                        thresh_clip,
                        size=self.clip_patch_grid_size,
                        mode="bilinear",
                        align_corners=True,
                    ).reshape(self.clip_patch_grid_size ** 2)

                    if thresh_clip.sum() < 1:
                        continue
                    thresh_clip[thresh_clip > 0] = 1
                    
                    # DINO mask: interpolate to DINO patch grid size (always 32x32 for 448px images)
                    thresh_dino = torch.tensor(normal_masks_capm[j]).reshape(1, 1, H, W)
                    thresh_dino = F.interpolate(
                        thresh_dino,
                        size=int(self.image_size / 14),
                        mode="bilinear",
                        align_corners=True,
                    ).reshape(int((self.image_size // 14) ** 2))
                    thresh_dino[thresh_dino > 0] = 1

                    selected_patch_features = self.normal_dino_patches[i][thresh_dino]
                    self.normal_dino_part_patch_features[normal_mask_idxs[j]].append(
                        selected_patch_features
                    )

                    for layer in range(len(self.normal_patch_tokens)):
                        if layer % 2 == 0:
                            continue
                        selected_patch_features = self.normal_patch_tokens[layer][i][
                            thresh_clip
                        ]
                        self.normal_clip_part_patch_features[layer][
                            normal_mask_idxs[j]
                        ].append(selected_patch_features)

                features = self.component_feature_extractor.extract(
                    image,
                    normal_masks,
                )
                for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                    self.normal_component_feats[feature_name].append(
                        features[feature_name]
                    )

            for j in range(len(normal_mask_idxs)):
                if self.normal_dino_part_patch_features[normal_mask_idxs[j]] == []:
                    continue
                self.normal_dino_part_patch_features[normal_mask_idxs[j]] = torch.cat(
                    self.normal_dino_part_patch_features[normal_mask_idxs[j]], dim=0
                )

                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    self.normal_clip_part_patch_features[layer][normal_mask_idxs[j]] = (
                        torch.cat(
                            self.normal_clip_part_patch_features[layer][
                                normal_mask_idxs[j]
                            ],
                            dim=0,
                        )
                    )

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                self.normal_component_feats[feature_name] = torch.cat(
                    self.normal_component_feats[feature_name], axis=0
                )
            self.normal_component_feats["clip_image"] = (
                self.normal_component_feats["clip_image"].transpose(0, 1)
            )

            # Apply CFA only if enabled
            if self.cfa is not None:
                for layer in range(self.normal_component_feats["clip_image"].shape[0]):
                    self.normal_component_feats["clip_image"][layer] = self.cfa(
                        self.normal_component_feats["clip_image"][layer]
                    )

                self.normal_component_feats["dino_image"] = self.cfa(
                    self.normal_component_feats["dino_image"]
                )


            self.normal_component_feats["geo"] = torch.cat(
                [
                    self.normal_component_feats["area"],
                    self.normal_component_feats["color"],
                    self.normal_component_feats["position"],
                ],
                dim=1,
            )

        if self.gate in (object_type.SINGLE, object_type.MULTI):
            self._finalize_part_feature_lists()


    def _finalize_part_feature_lists(self) -> None:
        """Convert cached part features from Python lists to tensors."""
        if getattr(self, "normal_dino_patches", None) is None:
            return

        dino_dim = (
            self.dino_embed_dim
            if getattr(self, "dino_embed_dim", None) is not None
            else self.normal_dino_patches.shape[-1]
        )
        dino_dtype = self.normal_dino_patches.dtype

        for idx in range(len(self.normal_dino_part_patch_features)):
            entry = self.normal_dino_part_patch_features[idx]
            if isinstance(entry, list):
                if len(entry) == 0:
                    self.normal_dino_part_patch_features[idx] = torch.empty(
                        (0, dino_dim), dtype=dino_dtype, device=self.device
                    )
                else:
                    self.normal_dino_part_patch_features[idx] = torch.cat(
                        entry, dim=0
                    )

        for layer in range(len(self.normal_clip_part_patch_features)):
            if layer % 2 == 0:
                continue
            clip_dtype = self.normal_patch_tokens[layer].dtype
            for idx in range(len(self.normal_clip_part_patch_features[layer])):
                entry = self.normal_clip_part_patch_features[layer][idx]
                if isinstance(entry, list):
                    if len(entry) == 0:
                        self.normal_clip_part_patch_features[layer][idx] = torch.empty(
                            (0, self.clip_embed_dim),
                            dtype=clip_dtype,
                            device=self.device,
                        )
                    else:
                        self.normal_clip_part_patch_features[layer][idx] = torch.cat(
                            entry, dim=0
                        )


def calculate_iou_torch(mask1, mask2):
    intersection = torch.sum((mask1 & mask2).float())
    return intersection

def assign_fine_to_coarse_torch(coarse_masks, fine_masks):
    M, H, W = coarse_masks.shape
    N = fine_masks.shape[0]

    coarse_to_fine_masks = {i: [] for i in range(M)}
    for fine_idx in range(N):
        if N > 1:
            if fine_masks[fine_idx][0, 0] and fine_masks[fine_idx][H - 1, W - 1]:
                continue
            if fine_masks[fine_idx][10, 10] and fine_masks[fine_idx][H - 10, W - 10]:
                continue
        best_iou = 0
        best_coarse_idx = -1
        for coarse_idx in range(M):
            iou = calculate_iou_torch(fine_masks[fine_idx], coarse_masks[coarse_idx])
            if iou > best_iou:
                best_iou = iou
                best_coarse_idx = coarse_idx
        if best_coarse_idx != -1:
            coarse_to_fine_masks[best_coarse_idx].append(fine_masks[fine_idx])

    new_coarse_masks = torch.zeros_like(coarse_masks)
    for coarse_idx in coarse_to_fine_masks.keys():
        assigned_fine_masks = coarse_to_fine_masks[coarse_idx]
        if len(assigned_fine_masks) > 0:
            for fine_mask in assigned_fine_masks:
                new_coarse_masks[coarse_idx][fine_mask > 0] = coarse_idx + 1

    return new_coarse_masks


class CFA(nn.Module):
    def __init__(self):
        super(CFA, self).__init__()

    def _compute_similarity_matrix(self, tensors):
        similarity_matrix = F.cosine_similarity(
            tensors.unsqueeze(1), tensors.unsqueeze(0), dim=-1
        )
        return similarity_matrix

    def _normalize_adjacency(self, adj_matrix):
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        normalized_adj = adj_matrix / row_sum
        return normalized_adj

    def forward(self, tensor):
        # print(tensor.shape)
        similarity_matrix = self._compute_similarity_matrix(tensor)
        normalized_adj_matrix = self._normalize_adjacency(similarity_matrix)
        aggregated_tensors = normalized_adj_matrix @ tensor
        return aggregated_tensors
