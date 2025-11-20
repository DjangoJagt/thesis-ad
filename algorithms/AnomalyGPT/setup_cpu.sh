#!/bin/bash
# Setup script for AnomalyGPT CPU-only mode

echo "=========================================="
echo "AnomalyGPT CPU-Only Setup"
echo "=========================================="

# Check if llama-cpp-python is installed
if python -c "import llama_cpp" 2>/dev/null; then
    echo "✓ llama-cpp-python is already installed"
else
    echo "✗ llama-cpp-python not found"
    echo "Installing llama-cpp-python..."
    pip install llama-cpp-python --no-cache-dir
fi

# Install required packages (CPU versions)
echo ""
echo "Installing required packages for CPU-only mode..."

# Install PyTorch CPU version if not already installed
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is already installed"
    python -c "import torch; print(f'  Version: {torch.__version__}')"
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other required packages
echo ""
echo "Installing other dependencies..."
pip install -q \
    easydict \
    einops \
    ftfy \
    kornia \
    matplotlib \
    numpy==1.24.3 \
    opencv-python \
    peft==0.3.0 \
    Pillow \
    PyYAML \
    scikit-learn \
    timm \
    tqdm \
    transformers==4.29.1 \
    sentencepiece

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run a simple test:"
echo "  cd code"
echo "  python test_cpu_simple.py --class_name bottle"
echo ""
