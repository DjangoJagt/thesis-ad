# Modular CPU Anomaly Pipeline

This directory contains a lightweight anomaly detection pipeline designed to run on CPU with ~16GB RAM using:
- CLIP (ViT-B/32) for vision features
- Optional quantized GGUF LLM (via llama.cpp wrapper) for textual Yes/No + explanation (can be skipped)
- Statistical few-shot feature bank + adaptive threshold for anomaly scoring

## Files
- `vision_encoder_clip.py`: Wraps CLIP model; returns global embedding and (if available) patch tokens.
- `anomaly_scorer.py`: FeatureBank (mean + covariance) and AdaptiveThreshold logic.
- `inference_cpu_modular.py`: Main entry script.

## Installation
Ensure `open_clip_torch` is installed:
```bash
pip install open_clip_torch==2.26.1
```
Optional LLM (already in repo): ensure `llama-cpp-python` and GGUF model path.

## Usage Examples
Basic anomaly inference without LLM explanation:
```bash
python code/modular/inference_cpu_modular.py \
  --data_root data/mvtec_anomaly_detection \
  --class_name bottle \
  --few_shot_k 8
```
With LLM explanation (if wrapper available):
```bash
python code/modular/inference_cpu_modular.py \
  --data_root data/mvtec_anomaly_detection \
  --class_name bottle \
  --few_shot_k 8 \
  --use_llm \
  --llm_path pretrained_ckpt/llm/mistral-7b-instruct-v0.2.Q4_K_S.gguf
```
Adjust threshold quantile (default 0.9):
```bash
python code/modular/inference_cpu_modular.py --quantile 0.95 ...
```

## Output Metrics
Printed summary includes:
- Accuracy, Precision, Recall over test images
- Learned cosine distance threshold
- (Optional) sample LLM outputs

## Custom Data
Expected layout:
```
my_data/
  class_x/
    train/good/*.png
    test/good/*.png
    test/defect_type_a/*.png
    test/defect_type_b/*.png
```
Run:
```bash
python code/modular/inference_cpu_modular.py --data_root my_data --class_name class_x --few_shot_k 5
```

## Extending
- Swap encoder: modify `VisionEncoderCLIP` to load different backbone (e.g., DINOv2) if available.
- Add patch-level heatmaps: use stored `patch_feats` and compute per-patch distance.
- Replace thresholding: implement KDE or isolation forest on feature bank.

## Notes
- Current Mahalanobis uses a simple covariance and may underperform with very few samples (<5); increase `few_shot_k` for stability.
- If LLM not present, core anomaly predictions still work.
- All computation is CPU-only; generation speed depends on model size.
