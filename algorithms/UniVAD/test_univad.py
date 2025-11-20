import argparse
import logging
import os
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math
from PIL import Image
from prefetch_generator import BackgroundGenerator
import cv2
from UniVAD import UniVAD
import matplotlib.pyplot as plt
from utils.visualize import save_anomaly_map, save_visualization_plot

from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from datasets.mvtec_loco import MVTecLocoDataset
from datasets.brainmri import BrainMRIDataset
from datasets.his import HISDataset
from datasets.resc import RESCDataset
from datasets.liverct import LiverCTDataset
from datasets.chestxray import ChestXrayDataset
from datasets.oct17 import OCT17Dataset
from torch.utils.data import Subset


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def resize_tokens(x):
    B, N, C = x.shape
    x = x.view(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
    return x


def cal_score(obj):
    table = []
    gt_px = []
    pr_px = []
    gt_sp = []
    pr_sp = []

    table.append(obj)
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)

    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())

    table.append(str(np.round(auroc_sp * 100, decimals=1)))
    table.append(str(np.round(auroc_px * 100, decimals=1)))

    table_ls.append(table)
    auroc_sp_ls.append(auroc_sp)
    auroc_px_ls.append(auroc_px)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=448, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument(
        "--dataset", type=str, default="mvtec", help="train dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/mvtec",
        help="path to test dataset",
    )
    parser.add_argument(
        "--save_path", type=str, default=f"./results/", help="path to save results"
    )
    parser.add_argument(
        "--round", type=int, default=0, help="round"
    )
    parser.add_argument("--class_name", type=str, default="None", help="class_name")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--light",
        action="store_true",
        help="Enable lightweight CPU-only mode to reduce RAM usage (uses smaller models and skips heavy components)",
    )
    parser.add_argument(
        "--save_anomaly_maps",
        action="store_true",
        help="Save anomaly maps as PNG images to save_path/anomaly_maps/",
    )
    parser.add_argument(
        "--save_examples",
        action="store_true",
        help="Save 4-panel example plots (original, PCA+mask, patch-dist heatmap, histogram)",
    )
    parser.add_argument(
        "--disable_cfa",
        action="store_true",
        help="Disable Component Feature Aggregator (CFA) for testing without feature aggregation. Default: enabled.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=-1,
        help="Limit number of test images per class (-1 for all, or specify number like 3, 5, 10)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device
    k_shot = args.k_shot

    # If lightweight CPU mode requested, limit OpenMP/MKL and PyTorch threads to reduce RAM/CPU pressure
    if args.light and device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    image_size = args.image_size
    save_path = args.save_path + "/" + dataset_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, "log.txt")

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    UniVAD_model = UniVAD(image_size=args.image_size, lightweight=args.light, enable_cfa=not args.disable_cfa).to(device)

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    gaussion_filter = torchvision.transforms.GaussianBlur(3, 4.0)

    if dataset_name == "mvtec":
        test_data = MVTecDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "visa":
        test_data = VisaDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            mode="test",
        )
    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(
            root=dataset_dir,
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(
            root="./data/BrainMRI",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "his":
        test_data = HISDataset(
            root="./data/HIS",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "resc":
        test_data = RESCDataset(
            root="./data/RESC",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(
            root="./data/ChestXray",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(
            root="./data/OCT17",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(
            root="./data/LiverCT",
            transform=transform,
            target_transform=transform,
            aug_rate=-1,
            mode="test",
        )
    else:
        raise NotImplementedError("Dataset not supported")
    

    # test_dataloader = DataLoaderX(
    #     test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    # )
    # test_dataloader = DataLoaderX(
    #     test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    # )

    # Limit test images if requested
    if args.num_images > 0:
        test_subset = torch.utils.data.Subset(test_data, range(min(args.num_images, len(test_data))))
    else:
        test_subset = test_data

    # Optimize DataLoader for device type: CPU benefits from no workers/no pinning
    num_workers = 0 if device == "cpu" else 8
    pin_memory = False if device == "cpu" else True
    test_dataloader = DataLoaderX(
        test_subset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in test_data.get_cls_names()]

    results = {}
    results["cls_names"] = []
    results["imgs_masks"] = []
    results["anomaly_maps"] = []
    results["gt_sp"] = []
    results["pr_sp"] = []

    cls_last = None

    image_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
   
    for items in tqdm(test_dataloader):
        image = items["img"].to(device)
        image_pil = items["img_pil"]
        image_path = items["img_path"][0]

        if args.class_name != "None":
            if args.class_name not in image_path:
                continue

        cls_name = items["cls_name"][0]
        results["cls_names"].append(cls_name)
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)  # px
        results["gt_sp"].append(items["anomaly"].item())

        if cls_name != cls_last:
            if dataset_name == "mvtec":
                normal_image_paths = [
                    "./data/mvtec/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "mvtec_loco":
                normal_image_paths = [
                    "./data/mvtec_loco_caption/"
                    + cls_name.replace(" ", "_")
                    + "/train/good/"
                    + str(i).zfill(3)
                    + ".png"
                    for i in range(args.round, args.round + k_shot)
                ]
            elif dataset_name == "visa":
                if cls_name.replace(" ", "_") in [
                    "capsules",
                    "cashew",
                    "chewinggum",
                    "fryum",
                    "pipe_fryum",
                ]:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(3)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
                else:
                    normal_image_paths = [
                        "./data/VisA_pytorch/1cls/"
                        + cls_name.replace(" ", "_")
                        + "/train/good/"
                        + str(i).zfill(4)
                        + ".JPG"
                        for i in range(args.round, args.round + k_shot)
                    ]
            elif dataset_name in [
                "his",
                "oct17",
                "chestxray",
                "brainmri",
                "liverct",
                "resc",
            ]:
                dir = (
                    "./data/"
                    + cls_name.replace(" ", "_")
                    + "/train/good"
                )
                files = sorted(os.listdir(dir))[:k_shot]
                normal_image_paths = [os.path.join(dir, file) for file in files]

            # normal_image_path = normal_image_paths[:k_shot]
            normal_images = torch.cat(
                [
                    image_transform(Image.open(x).convert("RGB")).unsqueeze(0)
                    for x in normal_image_paths
                ],
                dim=0,
            ).to(device)

            setup_data = {
                "few_shot_samples": normal_images,
                "dataset_category": cls_name.replace(" ", "_"),
                "image_path": normal_image_paths,
            }
            UniVAD_model.setup(setup_data)
            cls_last = cls_name

        with torch.no_grad():

            pred_value = UniVAD_model(image, image_path, image_pil)
            anomaly_score, anomaly_map = (
                pred_value["pred_score"],
                pred_value["pred_mask"],
            )
            results["anomaly_maps"].append(anomaly_map.detach().cpu().numpy())
            overall_anomaly_score = anomaly_score.item()
            results["pr_sp"].append(overall_anomaly_score)

            # Optionally save AnomalyDINO-style 4-panel visualization plots using UniVAD's native data
            if args.save_examples:
                # Determine how many examples already saved for this class
                examples_dir = os.path.join(save_path, "examples", cls_name)
                os.makedirs(examples_dir, exist_ok=True)
                existing = len([f for f in os.listdir(examples_dir) if f.endswith('.png')])
                
                if existing < 3:
                    out_path = os.path.join(examples_dir, f"example_{existing}.png")
                    
                    # Only load mask if CFA is enabled (component segmentation active)
                    if not args.disable_cfa:
                        # Get the corresponding mask path for this image
                        # Convert image path ./data/mvtec/bottle/test/broken_large/000.png
                        # to mask path ./masks/mvtec/bottle/test/broken_large/000/grounding_mask.png
                        relative_path = image_path.replace("./data/", "")  # mvtec/bottle/test/broken_large/000.png
                        mask_dir = relative_path.replace(".png", "").replace(".JPG", "").replace(".jpeg", "")  # mvtec/bottle/test/broken_large/000
                        mask_path = f"./masks/{mask_dir}/grounding_mask.png"
                    else:
                        # CFA disabled - no component segmentation, so no mask to display
                        mask_path = None
                    
                    # Use UniVAD's native outputs: anomaly_map tensor and overall_anomaly_score
                    save_visualization_plot(
                        original_image_pil=image_pil[0],
                        anomaly_map_tensor=anomaly_map,
                        image_score=overall_anomaly_score,
                        output_path=out_path,
                        mask_image_path=mask_path
                    )
                    logger.info(f"Saved visualization plot: {out_path}")

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []

    threads = [None] * 20
    idx = 0
    for obj in tqdm(obj_list):
        threads[idx] = threading.Thread(target=cal_score, args=(obj,))
        threads[idx].start()
        idx += 1

    for i in range(idx):
        threads[i].join()

    # Save anomaly maps as PNG if requested
    if args.save_anomaly_maps:
        anomaly_maps_dir = os.path.join(save_path, "anomaly_maps")
        os.makedirs(anomaly_maps_dir, exist_ok=True)
        logger.info(f"Saving anomaly maps to {anomaly_maps_dir}")
        
        for img_idx, (cls_name, anomaly_map) in enumerate(
            zip(results["cls_names"], results["anomaly_maps"])
        ):
            # Create subdirectory for each class
            class_dir = os.path.join(anomaly_maps_dir, cls_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save anomaly map with index
            map_path = os.path.join(class_dir, f"anomaly_map_{img_idx:04d}.png")
            save_anomaly_map(anomaly_map, map_path)
            logger.info(f"  Saved {map_path}")

    # logger
    table_ls.append(
        [
            "mean",
            str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
            str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
        ]
    )
    results = tabulate(
        table_ls,
        headers=[
            "objects",
            "auroc_sp",
            "auroc_px",
        ],
        tablefmt="pipe",
    )
    logger.info("\n%s", results)
