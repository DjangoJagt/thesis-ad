# AnomalyGPT CPU-Only Setup Guide

This guide explains how to run AnomalyGPT on CPU-only systems (like WSL without GPU) using a GGUF model.

## Prerequisites

- WSL or Linux system (CPU-only)
- Python 3.8+
- GGUF model (e.g., `mistral-7b-instruct-v0.2.Q4_K_S.gguf`)
- `llama-cpp-python` installed
- MVTec AD dataset

## Quick Start

### 1. Install Dependencies

```bash
# Run the setup script
./setup_cpu.sh
```

Or install manually:

```bash
# Install llama-cpp-python (already done)
pip install llama-cpp-python --no-cache-dir

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify Data Structure

Make sure your data is organized as follows:

```
AnomalyGPT/
├── data/
│   ├── mvtec_anomaly_detection/
│   │   └── bottle/
│   │       ├── train/
│   │       │   └── good/
│   │       └── test/
│   │           ├── good/
│   │           ├── broken_large/
│   │           └── ...
│   └── cognex_data/
│       └── ...
├── pretrained_ckpt/
│   ├── llm/
│   │   └── mistral-7b-instruct-v0.2.Q4_K_S.gguf
│   └── imagebind_ckpt/
│       └── imagebind_huge.pth
└── code/
    └── ...
```

### 3. Run Simple Test

```bash
cd code
python test_cpu_simple.py --class_name bottle
```

This will run a simplified test on a few samples from the bottle class.

## Command Line Options

```bash
python test_cpu_simple.py \
    --class_name bottle \
    --data_path ../data/mvtec_anomaly_detection \
    --gguf_model ../pretrained_ckpt/llm/mistral-7b-instruct-v0.2.Q4_K_S.gguf \
    --max_tokens 128 \
    --n_threads 4
```

**Options:**
- `--class_name`: MVTec class to test (bottle, cable, capsule, etc.)
- `--data_path`: Path to MVTec dataset
- `--gguf_model`: Path to GGUF model file
- `--max_tokens`: Maximum tokens to generate
- `--n_threads`: Number of CPU threads to use
- `--imagebind_ckpt`: Path to ImageBind checkpoint (optional)

## Important Notes

### Limitations of CPU-Only Mode

1. **No GPU acceleration**: Much slower than CUDA version
2. **Simplified pipeline**: The simple test script doesn't include:
   - Vision encoder integration
   - Few-shot learning with normal examples
   - Anomaly map generation
   - Full AnomalyGPT training pipeline

3. **Text-only mode**: Without proper vision-language alignment, the model only sees text descriptions, not the actual images

### What Works

- ✓ GGUF model loading and inference on CPU
- ✓ Basic text-based anomaly detection
- ✓ Testing framework for MVTec dataset
- ✓ No CUDA dependency

### What Doesn't Work (Yet)

- ✗ Vision encoding (ImageBind requires specific setup)
- ✗ Anomaly map generation (needs GPU for efficient computation)
- ✗ Training (original code requires CUDA)
- ✗ Full few-shot learning pipeline

## Next Steps for Full Functionality

To get the full AnomalyGPT working on CPU, you would need to:

1. **Modify ImageBind for CPU**: Currently ImageBind is designed for CUDA
2. **Integrate vision embeddings**: Connect image features to the GGUF model
3. **Port anomaly detection head**: Adapt the pixel-level detection for CPU
4. **Convert checkpoints**: Transform any trained weights to CPU-compatible format

## Alternative: Use Original Code with Smaller Models

If you need full functionality, consider:

1. Getting a smaller LLaMA model in HuggingFace format (not GGUF)
2. Using the original code with `--device cpu` modifications
3. Accepting much slower inference times

## Troubleshooting

### llama-cpp-python errors
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir
```

### PyTorch CUDA errors
Make sure you installed the CPU version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory issues
Reduce the number of threads:
```bash
python test_cpu_simple.py --n_threads 2
```

## Example Output

```
==========================================================
AnomalyGPT CPU-Only Test Script
==========================================================
Class: bottle
Data path: ../data/mvtec_anomaly_detection
GGUF model: ../pretrained_ckpt/llm/mistral-7b-instruct-v0.2.Q4_K_S.gguf
==========================================================

Initializing GGUF model...
✓ GGUF model loaded successfully

Testing samples from ../data/mvtec_anomaly_detection/bottle/test...

Test 1: good/000.png
  Ground truth: Normal
  Response: No
  Prediction: ✓ Correct

Test 2: broken_large/000.png
  Ground truth: Anomaly
  Response: Yes
  Prediction: ✓ Correct
...
```

## Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify data paths are correct
3. Ensure GGUF model is valid and accessible
4. Review error messages carefully
