import argparse
import csv
import json
import logging
import os

import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
import yaml

from src.utils import get_dataset_info
from src.detection import run_anomaly_detection
from src.backbones import get_model


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("anomalydino_no_pixel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser("AnomalyDINO no-pixel evaluator")
    parser.add_argument("--dataset", type=str, default="Sick")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="Base directory where results_{dataset}/... will be created",
    )
    parser.add_argument("--objects", nargs="+", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--preprocess", type=str, default="agnostic")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--faiss_on_cpu", action="store_true")
    parser.add_argument("--mask_ref_images", action="store_true")
    parser.add_argument("--save_examples", action="store_true")
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of iterations for CUDA warmup")

    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to run directory")

    return parser.parse_args()


def build_results_dir(args):
    shot_dir = f"{args.shots}-shot_preprocess={args.preprocess}"
    if args.tag:
        shot_dir = f"{shot_dir}_{args.tag}"
    return os.path.join(
        args.save_path,
        f"results_{args.dataset}",
        f"{args.model_name}_{args.resolution}",
        shot_dir,
    )


def persist_metadata(results_dir, args, masking_default, rotation_default):
    args_path = os.path.join(results_dir, "args.yaml")
    with open(args_path, "w") as f:
        yaml.dump(vars(args), f)

    preprocess_path = os.path.join(results_dir, "preprocess.yaml")
    with open(preprocess_path, "w") as f:
        yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)


def collect_test_items(data_root: str, object_name: str):
    class_items = []
    test_dir = os.path.join(data_root, object_name, "test")
    for subtype in sorted(os.listdir(test_dir)):
        subtype_dir = os.path.join(test_dir, subtype)
        if not os.path.isdir(subtype_dir):
            continue
        is_anomaly = 0 if subtype == "good" else 1
        for fname in sorted(os.listdir(subtype_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            name_no_ext, _ = os.path.splitext(fname)
            class_items.append(
                {
                    "path": os.path.join(subtype_dir, fname),
                    "label": is_anomaly,
                    "subfolder": subtype,
                    "id": name_no_ext,
                    "filename": fname,
                    "score_key": f"{subtype}/{fname}",
                }
            )
    return class_items


def save_results_csv(csv_path: str, per_sample_data):
    fieldnames = [
        "class_name",
        "subfolder",
        "filename",
        "anomaly_score",
        "gt_label",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample_data:
            writer.writerow(row)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    divider = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(divider)
    for row in rows:
        print(fmt(row))


def main():
    args = parse_args()

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        args.dataset, args.preprocess
    )
    if args.objects:
        objects = [obj for obj in objects if obj in args.objects]
        if not objects:
            raise ValueError("No matching objects for selection")

    results_dir = build_results_dir(args)
    os.makedirs(results_dir, exist_ok=True)
    persist_metadata(results_dir, args, masking_default, rotation_default)

    log_path = os.path.join(results_dir, "log.txt")
    logger = build_logger(log_path)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

    per_class_rows = []
    sample_rows = []
    aucs = []
    time_measurements = []

    time_file = os.path.join(results_dir, "time_measurements.csv")
    with open(time_file, "w", newline="") as time_csv:
        time_writer = csv.writer(time_csv)
        time_writer.writerow([
            "Object",
            "Sample",
            "Anomaly_Score",
            "MemoryBank_Time",
            "Inference_Time",
        ])

        for object_name in objects:
            logger.info(f"Processing {object_name}")
            items = collect_test_items(args.data_root, object_name)
            if not items:
                logger.warning(f"No test images for {object_name}, skipping")
                continue

            if args.save_examples:
                object_dir = os.path.join(results_dir, object_name)
                os.makedirs(object_dir, exist_ok=True)
                os.makedirs(os.path.join(object_dir, "examples"), exist_ok=True)

            train_good_dir = os.path.join(args.data_root, object_name, "train", "good")
            if args.warmup_iters > 0:
                try:
                    first_image = sorted(os.listdir(train_good_dir))[0]
                    first_image_path = os.path.join(train_good_dir, first_image)
                    for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                        img_tensor, grid_size = model.prepare_image(first_image_path)
                        model.extract_features(img_tensor)
                except (IndexError, FileNotFoundError):
                    logger.warning(f"Warmup skipped for {object_name}: no training images found")

            anomaly_scores, time_memorybank, time_inference = run_anomaly_detection(
                model,
                object_name,
                data_root=args.data_root,
                n_ref_samples=args.shots,
                object_anomalies=object_anomalies,
                plots_dir=results_dir,
                save_examples=args.save_examples,
                knn_metric=args.knn_metric,
                knn_neighbors=args.k_neighbors,
                faiss_on_cpu=args.faiss_on_cpu,
                masking=masking_default.get(object_name, False),
                mask_ref_images=args.mask_ref_images,
                rotation=rotation_default.get(object_name, False),
                seed=args.seed,
                save_patch_dists=True,
                save_tiffs=False,
            )

            for sample_key in sorted(anomaly_scores.keys()):
                anomaly_score = anomaly_scores[sample_key]
                inference_time = time_inference.get(sample_key, 0.0)
                time_writer.writerow(
                    [
                        object_name,
                        sample_key,
                        f"{anomaly_score:.5f}",
                        f"{time_memorybank:.5f}",
                        f"{inference_time:.5f}",
                    ]
                )
                time_measurements.append(inference_time)

            preds = []
            gts = []
            missing_scores = 0
            for sample in items:
                score = anomaly_scores.get(sample["score_key"])
                if score is None:
                    missing_scores += 1
                    score = 0.0
                preds.append(score)
                gts.append(sample["label"])
                sample_rows.append(
                    {
                        "class_name": object_name,
                        "subfolder": sample["subfolder"],
                        "filename": sample["id"],
                        "anomaly_score": score,
                        "gt_label": sample["label"],
                    }
                )

            if missing_scores:
                logger.warning(
                    f"{missing_scores} samples in {object_name} missing scores (likely key mismatch)."
                )

            if len(set(gts)) < 2:
                logger.warning(f"Not enough anomaly vs normal samples for {object_name}")
                continue

            auroc = roc_auc_score(gts, preds)
            aucs.append(auroc)
            per_class_rows.append([object_name, f"{auroc:.4f}", str(len(preds))])
            logger.info(f"AUROC ({object_name}): {auroc:.4f}")

    if time_measurements:
        logger.info(
            f"Mean inference time: {np.mean(time_measurements):.5f} s/sample"
        )
    logger.info(f"Saved timing measurements to {time_file}")

    if per_class_rows:
        print_table(["object", "AUROC", "#samples"], per_class_rows)
        logger.info(f"Mean AUROC: {np.mean(aucs):.4f}")

    csv_path = os.path.join(results_dir, "detailed_results.csv")
    save_results_csv(csv_path, sample_rows)
    logger.info(f"Saved per-sample results to {csv_path}")

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "objects": per_class_rows,
                "mean_auroc": np.mean(aucs) if aucs else None,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
