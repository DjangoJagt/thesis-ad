import torch
import numpy as np
from PIL import Image
import open_clip
from typing import Tuple, Optional

class VisionEncoderCLIP:
    """Lightweight CLIP encoder wrapper returning global embedding.
    Patch-level features are approximated via registering forward hook on the last convolutional block (if available).
    """
    def __init__(self, model_name: str = 'ViT-B-32', pretrained: str = 'openai', device: str = 'cpu'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.model.eval()
        self._patch_features = None
        # Try to hook into visual transformer blocks to capture token embeddings before pooling
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
            def _hook(module, inp, out):
                # out shape: (B, seq_len, dim) for ViT
                self._patch_features = out.detach()
            # Register on transformer forward
            self.model.visual.transformer.register_forward_hook(_hook)
        self.embed_dim = self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512

    def encode_image(self, image: Image.Image) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            global_feat = self.model.encode_image(image_input)
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
            patch_feats = None
            if self._patch_features is not None:
                # Remove CLS token (index 0)
                patch_feats = self._patch_features[:, 1:, :]
                patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)
        return global_feat.cpu(), patch_feats.cpu() if patch_feats is not None else None

    def batch_encode_paths(self, paths):
        globals = []
        patches = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            g, pf = self.encode_image(img)
            globals.append(g)
            patches.append(pf)
        global_tensor = torch.cat(globals, dim=0)
        return global_tensor, patches
