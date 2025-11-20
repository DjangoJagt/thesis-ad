"""
CPU-only test script for AnomalyGPT using GGUF model
This is a simplified version that works without CUDA
"""
import os
import sys
import torch
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

parser = argparse.ArgumentParser("AnomalyGPT CPU Test", add_help=True)
parser.add_argument("--data_path", type=str, default="../data/mvtec_anomaly_detection")
parser.add_argument("--class_name", type=str, default="bottle")
parser.add_argument("--gguf_model", type=str, default="../pretrained_ckpt/llm/mistral-7b-instruct-v0.2.Q4_K_S.gguf")
parser.add_argument("--imagebind_ckpt", type=str, default="../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth")
parser.add_argument("--anomalygpt_ckpt", type=str, default=None, help="Path to AnomalyGPT checkpoint if available")
parser.add_argument("--max_tokens", type=int, default=128)
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--few_shot_k", type=int, default=4, help="Number of normal training images to build a simple pixel baseline")
parser.add_argument("--resize", type=int, default=128, help="Resize square for simple anomaly scoring")

args = parser.parse_args()

# Import the CPU wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from llama_cpp_wrapper import LlamaCppWrapper, SimpleLlamaTokenizer

print("=" * 60)
print("AnomalyGPT CPU-Only Test Script")
print("=" * 60)
print(f"Class: {args.class_name}")
print(f"Data path: {args.data_path}")
print(f"GGUF model: {args.gguf_model}")
print("=" * 60)

# Check if GGUF model exists
if not os.path.exists(args.gguf_model):
    print(f"Error: GGUF model not found at {args.gguf_model}")
    sys.exit(1)

# Initialize GGUF model
print("\nInitializing GGUF model...")
try:
    llm_model = LlamaCppWrapper(
        model_path=args.gguf_model,
        n_ctx=2048,
        n_threads=args.n_threads,
        n_gpu_layers=0  # CPU only
    )
    tokenizer = SimpleLlamaTokenizer(llm_model.llm)
    print("✓ GGUF model loaded successfully")
except Exception as e:
    print(f"Error loading GGUF model: {e}")
    sys.exit(1)

# Skip ImageBind for now (requires deepspeed which is complex to install on CPU-only)
print("\nNote: Running in text-only mode (ImageBind requires GPU dependencies)")
print("The model will use text descriptions instead of visual features.")
visual_encoder = None
llama_proj = None

# Descriptions for MVTec classes
descriptions = {
    'bottle': "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part.",
    'cable': "This is a photo of three cables for anomaly detection, cables cannot be missed or swapped, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'capsule': "This is a photo of a capsule for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part.",
    'carpet': "This is a photo of carpet for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'grid': "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'hazelnut': "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'leather': "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part.",
    'metal_nut': "This is a photo of a metal nut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped.",
    'pill': "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part.",
    'screw': "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part.",
    'tile': "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'toothbrush': "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'transistor': "This is a photo of a transistor for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'wood': "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part.",
    'zipper': "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
}

def create_prompt(description, question, stats_text):
    """Create a prompt for the LLM with injected lightweight vision stats"""
    prompt = f"""<s>[INST] {description}\n\n{question}\n\nImage summary: {stats_text}\n\nRespond ONLY with Yes or No. Do not add extra words. [/INST]"""
    return prompt

def load_image(path, resize):
    img = Image.open(path).convert('RGB').resize((resize, resize))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def build_normal_baseline(train_good_dir, k, resize):
    candidates = sorted([f for f in os.listdir(train_good_dir) if f.endswith('.png')])[:k]
    stack = []
    for f in candidates:
        stack.append(load_image(os.path.join(train_good_dir, f), resize))
    if not stack:
        return None, None
    arr = np.stack(stack, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-5
    return mean, std

def anomaly_score(image_arr, mean, std):
    if mean is None:
        return 0.0, 0.0
    z = np.abs(image_arr - mean) / std
    pixel_score = z.mean()
    top_region = z.max()
    return float(pixel_score), float(top_region)

def format_stats(pixel_score, top_region):
    return f"pixel_score={pixel_score:.3f}, max_region_score={top_region:.3f}"

def enforce_yes_no(text):
    text = text.strip()
    if 'yes' in text.lower():
        return 'Yes'
    if 'no' in text.lower():
        return 'No'
    # fallback heuristic
    return 'Yes' if any(w in text.lower() for w in ['anomaly','defect','damage','broken']) else 'No'

def simple_predict(image_path, description, mean, std):
    question = "Is there any anomaly in the image?"
    img_arr = load_image(image_path, args.resize)
    p_score, r_score = anomaly_score(img_arr, mean, std)
    stats_text = format_stats(p_score, r_score)
    prompt = create_prompt(description, question, stats_text)
    try:
        response = llm_model.generate_text(
            prompt,
            max_tokens=32,
            temperature=0.2,
            top_p=0.9,
            stop=["</s>", "\n"]
        )
        answer = enforce_yes_no(response)
        return answer, stats_text
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Error", stats_text

# Test on a single class
print(f"\nTesting on class: {args.class_name}")
print("=" * 60)

class_path = os.path.join(args.data_path, args.class_name)
test_path = os.path.join(class_path, 'test')

if not os.path.exists(test_path):
    print(f"Error: Test path not found: {test_path}")
    sys.exit(1)

description = descriptions.get(args.class_name, descriptions['bottle'])

# Test on a few samples
results = []
test_count = 0
max_tests = 5  # Limit for initial testing

print(f"\nBuilding few-shot baseline (k={args.few_shot_k}) ...")
train_good_dir = os.path.join(class_path, 'train', 'good')
mean, std = build_normal_baseline(train_good_dir, args.few_shot_k, args.resize)
if mean is None:
    print("⚠ No normal training images found; baseline disabled.")
else:
    print("✓ Baseline statistics computed (mean/std).")

print(f"\nTesting samples from {test_path}...")

for defect_type in os.listdir(test_path):
    defect_path = os.path.join(test_path, defect_type)
    if not os.path.isdir(defect_path):
        continue
    
    is_normal = (defect_type == 'good')
    
    for img_file in os.listdir(defect_path):
        if not img_file.endswith('.png'):
            continue
        
        if test_count >= max_tests:
            break
        
        img_path = os.path.join(defect_path, img_file)
        
        print(f"\nTest {test_count + 1}: {defect_type}/{img_file}")
        print(f"  Ground truth: {'Normal' if is_normal else 'Anomaly'}")
        
        answer, stats_text = simple_predict(img_path, description, mean, std)
        print(f"  Stats: {stats_text}")
        print(f"  Answer: {answer}")
        pred_normal = (answer == 'No')
        pred_anomaly = (answer == 'Yes')
        correct = (is_normal and pred_normal) or ((not is_normal) and pred_anomaly)
        results.append({
            'image': img_path,
            'is_normal': is_normal,
            'response': answer,
            'correct': correct
        })
        
        print(f"  Prediction: {'✓ Correct' if correct else '✗ Wrong'}")
        
        test_count += 1
    
    if test_count >= max_tests:
        break

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
correct_count = sum(1 for r in results if r['correct'])
print(f"Accuracy: {correct_count}/{len(results)} = {100*correct_count/len(results):.1f}%")
print("\nNote: This is a simplified test without the full AnomalyGPT pipeline.")
print("For better results, you would need:")
print("  1. Trained AnomalyGPT checkpoints")
print("  2. Vision-language alignment")
print("  3. Few-shot normal examples")
print("=" * 60)
