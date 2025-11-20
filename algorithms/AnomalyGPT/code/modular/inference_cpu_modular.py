import os
import sys
import argparse
import time
from PIL import Image
import torch
from vision_encoder_clip import VisionEncoderCLIP
from anomaly_scorer import FeatureBank, AdaptiveThreshold

# Ensure model directory (code/model) is on path for llama_cpp_wrapper import
MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
if MODULE_ROOT not in sys.path:
    sys.path.insert(0, MODULE_ROOT)

try:
    from llama_cpp_wrapper import LlamaCppWrapper, SimpleLlamaTokenizer
    HAVE_LLM = True
except Exception as e:
    print(f"[LLM Import Warning] {e}. LLM features disabled.")
    HAVE_LLM = False

DESCRIPTION_TEMPLATE = "This is a photo of a {cls} for anomaly detection; normal samples should be defect-free."
PROMPT_TEMPLATE = """<s>[INST] {description}\n\nGlobal feature cosine distance: {cosine:.3f}\nMahalanobis distance: {maha:.3f}.\nIs this image anomalous? Respond ONLY Yes or No. [/INST]"""

def gather_image_paths(root_dir):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.png'):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)

def main():
    parser = argparse.ArgumentParser("Modular Anomaly Inference CPU")
    parser.add_argument('--data_root', type=str, default='../data/mvtec_anomaly_detection')
    parser.add_argument('--class_name', type=str, default='bottle')
    parser.add_argument('--few_shot_k', type=int, default=8)
    parser.add_argument('--clip_model', type=str, default='ViT-B-32')
    parser.add_argument('--clip_pretrained', type=str, default='openai')
    parser.add_argument('--llm_path', type=str, default='../pretrained_ckpt/llm/mistral-7b-instruct-v0.2.Q4_K_S.gguf')
    parser.add_argument('--use_llm', action='store_true')
    parser.add_argument('--llm_max_samples', type=int, default=5, help='Max images to run LLM on (0=all)')
    parser.add_argument('--quantile', type=float, default=0.9)
    parser.add_argument('--n_threads', type=int, default=4)
    args = parser.parse_args()

    device = 'cpu'
    print('Initializing CLIP encoder...')
    encoder = VisionEncoderCLIP(model_name=args.clip_model, pretrained=args.clip_pretrained, device=device)

    class_dir = os.path.join(args.data_root, args.class_name)
    train_good_dir = os.path.join(class_dir, 'train', 'good')
    test_dir = os.path.join(class_dir, 'test')
    if not os.path.isdir(train_good_dir) or not os.path.isdir(test_dir):
        print('Data directories missing.')
        return

    good_images = sorted([f for f in os.listdir(train_good_dir) if f.endswith('.png')])[:args.few_shot_k]
    print(f'Building feature bank from {len(good_images)} normal samples...')
    bank = FeatureBank()
    for fname in good_images:
        img_path = os.path.join(train_good_dir, fname)
        img = Image.open(img_path).convert('RGB')
        global_feat, _ = encoder.encode_image(img)
        bank.add(global_feat)
    bank.finalize()

    # Fit adaptive threshold using hold-out normal images (reuse bank feats distances)
    distances = []
    for feats in bank.global_feats:
        for row in feats:
            score = bank.score(row.unsqueeze(0))
            distances.append(score['cosine_distance'])
    thresh = AdaptiveThreshold(quantile=args.quantile)
    thresh.fit(distances)
    print(f'Adaptive cosine distance threshold @ q={args.quantile}: {thresh.threshold:.3f}')

    # Optional LLM
    llm = None
    tokenizer = None
    llm_count = 0
    if args.use_llm and HAVE_LLM:
        print('Loading LLM (quantized GGUF)...')
        print(f'  Note: LLM generation is slow on CPU (~30-60s per image with 7B model)')
        if args.llm_max_samples > 0:
            print(f'  Limiting LLM explanations to first {args.llm_max_samples} images')
        llm = LlamaCppWrapper(model_path=args.llm_path, n_ctx=1024, n_threads=args.n_threads, n_gpu_layers=0)
        tokenizer = SimpleLlamaTokenizer(llm.llm)
    elif args.use_llm and not HAVE_LLM:
        print('LLM requested but wrapper not available (skipping).')

    print('Running inference on test set...')
    results = []
    start_time = time.time()
    img_count = 0
    total_imgs = sum(len([f for f in os.listdir(os.path.join(test_dir, d)) if f.endswith('.png')]) 
                     for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
    
    for defect_type in sorted(os.listdir(test_dir)):
        defect_path = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_path):
            continue
        is_normal = defect_type == 'good'
        for fname in sorted(os.listdir(defect_path)):
            if not fname.endswith('.png'):
                continue
            img_count += 1
            img_path = os.path.join(defect_path, fname)
            
            # Progress indicator
            if img_count % 10 == 0 or img_count == 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / img_count if img_count > 0 else 0
                eta = avg_time * (total_imgs - img_count)
                print(f"  [{img_count}/{total_imgs}] {defect_type}/{fname} | Avg: {avg_time:.1f}s/img | ETA: {eta:.0f}s")
            
            img = Image.open(img_path).convert('RGB')
            global_feat, _ = encoder.encode_image(img)
            score = bank.score(global_feat)
            anomaly_flag = thresh.predict(score['cosine_distance'])
            # Override with label for confusion later
            label = 0 if is_normal else 1
            predicted = anomaly_flag
            explanation = ''
            if llm is not None and (args.llm_max_samples == 0 or llm_count < args.llm_max_samples):
                llm_count += 1
                description = DESCRIPTION_TEMPLATE.format(cls=args.class_name)
                prompt = PROMPT_TEMPLATE.format(description=description, cosine=score['cosine_distance'], maha=score['mahalanobis'])
                raw = llm.generate_text(prompt, max_tokens=16, temperature=0.2, top_p=0.95, stop=['</s>', '\n'])
                explanation = raw.strip()
            results.append({
                'path': img_path,
                'defect_type': defect_type,
                'cosine_distance': score['cosine_distance'],
                'mahalanobis': score['mahalanobis'],
                'pred': predicted,
                'label': label,
                'llm_out': explanation
            })
    elapsed = time.time() - start_time

    # Metrics
    if results:
        tp = sum(1 for r in results if r['pred'] == 1 and r['label'] == 1)
        tn = sum(1 for r in results if r['pred'] == 0 and r['label'] == 0)
        fp = sum(1 for r in results if r['pred'] == 1 and r['label'] == 0)
        fn = sum(1 for r in results if r['pred'] == 0 and r['label'] == 1)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        accuracy = (tp + tn) / (len(results) + 1e-9)
        print('\n=== Summary ===')
        print(f'Images processed: {len(results)} in {elapsed:.1f}s')
        print(f'Accuracy: {accuracy*100:.2f}%  Precision: {precision*100:.2f}%  Recall: {recall*100:.2f}%')
        print(f'Threshold (cosine distance): {thresh.threshold:.3f}')
        if llm is not None:
            print('Sample LLM outputs:')
            for r in results[:5]:
                print(f"{os.path.basename(r['path'])}: {r['llm_out']}")

if __name__ == '__main__':
    main()
