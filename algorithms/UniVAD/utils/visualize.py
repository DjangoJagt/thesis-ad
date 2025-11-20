import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def save_anomaly_map(anomaly_map, output_path, colormap=cv2.COLORMAP_JET):
    """Save anomaly map as a heatmap PNG image.

    This mirrors the behavior previously inline in `test_univad.py`.
    """
    anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]
    # Normalize to [0, 1]
    if anomaly_map.max() > 0:
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, colormap)
    cv2.imwrite(output_path, heatmap)


def save_visualization_plot(original_image_pil, anomaly_map_tensor, image_score, output_path, mask_image_path=None):
    """
    Save a 4-panel AnomalyDINO-style visualization using UniVAD's native data.

    Panels:
    1. Original image
    2. Mask overlay (from grounding SAM if available)
    3. UniVAD anomaly heatmap (raw scale)
    4. Histogram of anomaly scores

    Uses raw anomaly map values (no per-image min-max) so colorbar and histogram
    reflect the true UniVAD distances/scores (AnomalyDINO-style).
    """
    orig = np.array(original_image_pil)

    if isinstance(anomaly_map_tensor, torch.Tensor):
        anomaly_map = anomaly_map_tensor.detach().cpu().numpy()
    else:
        anomaly_map = anomaly_map_tensor
    anomaly_map = np.squeeze(anomaly_map)
    anomaly_map_raw = anomaly_map.astype(np.float32)

    # Create 4-panel figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original Image
    ax1.imshow(orig)
    ax1.axis('off')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')

    # Panel 2: Mask Overlay (from grounding SAM) with distinct component coloring
    if mask_image_path and os.path.exists(mask_image_path):
        try:
            from PIL import Image as PILImage
            mask_img = PILImage.open(mask_image_path)
            
            # Check if it's already a color mask or grayscale
            if mask_img.mode == 'RGB' or mask_img.mode == 'RGBA':
                # Already colored mask - just overlay it
                mask_rgb = np.array(mask_img.convert('RGB'))
                if mask_rgb.shape[:2] != orig.shape[:2]:
                    mask_rgb = np.array(PILImage.fromarray(mask_rgb).resize((orig.shape[1], orig.shape[0])))
                
                # Create overlay with the colored mask
                overlay = orig.copy().astype(np.float32)
                # Detect non-black pixels as mask regions
                mask_regions = np.any(mask_rgb > 10, axis=2)
                alpha = 0.5
                overlay[mask_regions] = overlay[mask_regions] * (1 - alpha) + mask_rgb[mask_regions] * alpha
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                
                ax2.imshow(overlay)
                ax2.axis('off')
                ax2.set_title('Mask Overlay (color)', fontsize=12, fontweight='bold')
            else:
                # Grayscale mask - process as before
                mask_gray = np.array(mask_img.convert('L'))
                if mask_gray.shape != orig.shape[:2]:
                    mask_gray = np.array(PILImage.fromarray(mask_gray).resize((orig.shape[1], orig.shape[0])))

                # Normalize label space: common formats are {0,255} or {0,1,2,...}
                unique_vals = sorted([v for v in np.unique(mask_gray) if v != 0])
                # If only values are {0,255} treat 255 as label 1
                remap = {}
                if unique_vals == [255]:
                    remap[255] = 1
                    unique_vals = [1]
                elif max(unique_vals) > 50 and len(unique_vals) < 15:  # large grayscale values, remap for clarity
                    # Scale large sparse labels down to consecutive ints
                    for i,v in enumerate(unique_vals, start=1):
                        remap[v] = i
                    unique_vals = list(range(1, len(remap)+1))
                if remap:
                    mask_remapped = np.zeros_like(mask_gray)
                    for old,new in remap.items():
                        mask_remapped[mask_gray == old] = new
                    mask_gray = mask_remapped

                overlay = orig.copy().astype(np.float32)
                cmap = plt.get_cmap('tab10')
                alpha_fill = 0.55
                alpha_edge = 0.9

                # Draw each component with a distinct color
                for idx,label in enumerate(unique_vals):
                    comp = mask_gray == label
                    if not np.any(comp):
                        continue
                    color = np.array(cmap(idx % 10)[:3]) * 255.0
                    overlay[comp] = overlay[comp] * (1 - alpha_fill) + color * alpha_fill
                    # Edge highlight
                    comp_uint8 = comp.astype(np.uint8) * 255
                    edges = cv2.Canny(comp_uint8, 100, 200) > 0
                    overlay[edges] = np.array([255, 255, 255]) * alpha_edge + overlay[edges] * (1 - alpha_edge)

                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                ax2.imshow(overlay)
                ax2.axis('off')
                num_components = len(unique_vals)
                ax2.set_title(
                    f'Mask Overlay ({num_components} component{"s" if num_components != 1 else ""})',
                    fontsize=12,
                    fontweight='bold'
                )
        except Exception as e:
            print(f"[Warning] Failed to load/parse mask at {mask_image_path}: {str(e)}")
            ax2.imshow(orig)
            ax2.axis('off')
            ax2.set_title('Mask (Failed to load)', fontsize=12, fontweight='bold')
    else:
        ax2.imshow(orig)
        ax2.axis('off')
        if mask_image_path is None:
            status = 'CFA disabled'
        elif not os.path.exists(mask_image_path):
            status = 'File not found'
        else:
            status = 'Path not provided'
        ax2.set_title(f'Mask Overlay ({status})', fontsize=12, fontweight='bold')
        if mask_image_path and not os.path.exists(mask_image_path):
            print(f"[Warning] Mask file not found: {mask_image_path}")

    # Panel 3: UniVAD Anomaly Heatmap with colorbar (raw scale)
    vmin = float(np.nanmin(anomaly_map_raw)) if anomaly_map_raw.size > 0 else 0.0
    vmax = float(np.nanmax(anomaly_map_raw)) if anomaly_map_raw.size > 0 else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    im3 = ax3.imshow(anomaly_map_raw, cmap='jet', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.set_label('Anomaly Score')
    ax3.axis('off')
    ax3.set_title('UniVAD Anomaly Map', fontsize=12, fontweight='bold')

    # Panel 4: Histogram of anomaly scores with image-level score line
    anomaly_scores_flat = anomaly_map_raw.flatten()
    ax4.hist(anomaly_scores_flat, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(image_score, color='r', linestyle='--', linewidth=2, label=f'score={image_score:.3f}')
    ax4.legend()
    ax4.set_title('Histogram of Anomaly Scores', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Anomaly Score (raw)')
    ax4.set_ylabel('Frequency')
    ax4.set_xlim(vmin, vmax)

    plt.suptitle(f'UniVAD Analysis: {os.path.basename(output_path)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
