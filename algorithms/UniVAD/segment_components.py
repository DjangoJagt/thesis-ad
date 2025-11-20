from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml
import argparse


mask_path = "./masks/mvtec"
categories = [
    "bottle"]
# categories = [
#     "bottle",
#     "cable",
#     "capsule",
#     "carpet",
#     "grid",
#     "hazelnut",
#     "leather",
#     "metal_nut",
#     "pill",
#     "screw",
#     "tile",
#     "toothbrush",
#     "transistor",
#     "wood",
#     "zipper",
# ]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def apply_cli_sam_overrides(config, cli_args):
    """Helper function to apply CLI SAM variant/checkpoint overrides to config."""
    gc = config.setdefault("grounding_config", {})

    # If user provided a variant override, set it
    if cli_args.sam_variant is not None:
        gc["sam_variant"] = cli_args.sam_variant

    # If user provided an explicit checkpoint, it overrides everything
    if cli_args.sam_checkpoint is not None:
        gc["sam_checkpoint"] = cli_args.sam_checkpoint
    else:
        # If no explicit checkpoint was provided, prefer any checkpoint present in config,
        # otherwise pick a sensible default based on the requested variant (if any).
        if "sam_checkpoint" not in gc or gc.get("sam_checkpoint") in (None, ""):
            # choose default checkpoint by variant
            variant = gc.get("sam_variant", None)
            default_ckpt_map = {
                "vit_b": "pretrained_ckpts/sam_hq_vit_b.pth",
                "vit_h": "pretrained_ckpts/sam_hq_vit_h.pth",
            }
            if variant in default_ckpt_map:
                gc["sam_checkpoint"] = default_ckpt_map[variant]
            else:
                # if no variant requested and config lacks checkpoint, leave as-is
                gc.setdefault("sam_checkpoint", gc.get("sam_checkpoint", None))

    # If we resolved a checkpoint path, make it explicit (helpful for logs)
    return config


parser = argparse.ArgumentParser(description="Generate grounding+SAM masks for datasets")
parser.add_argument("--sam_variant", type=str, default=None,
                    help="SAM variant to use (vit_b, vit_h). If not set, uses config value or default vit_h")
parser.add_argument("--sam_checkpoint", type=str, default=None,
                    help="Path to SAM checkpoint to use (overrides config)")
parser.add_argument("--num_images", type=int, default=None,
                    help="Maximum number of images to process per category (default: all images). Use None or -1 for all, or specify e.g. 3, 10, 100")
args = parser.parse_args()

# Convert num_images: None or -1 means all images; positive int means limit
num_images_limit = None if (args.num_images is None or args.num_images == -1) else args.num_images

for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec/{category}/test/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec/{category}/train/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks/VisA_pytorch/1cls"
categories = [
    "candle",
    "capsules",
    "chewinggum",
    "cashew",
    "fryum",
    "pipe_fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/VisA_pytorch/1cls/{category}/test/*/*.JPG"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/VisA_pytorch/1cls/{category}/train/*/*.JPG"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks/mvtec_loco_caption"
categories = [
    'breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors'
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec_loco_caption/{category}/test/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec_loco_caption/{category}/train/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks"
categories = [
    "LiverCT",
    "BrainMRI",
    "RESC",
    "HIS",
    "ChestXray"
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/test/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/train/*/*.png"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks"
categories = [
    "OCT17"
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/test/*/*.jpeg"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/train/*/*.jpeg"))
    if num_images_limit is not None:
        image_paths = image_paths[:num_images_limit]
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    config = apply_cli_sam_overrides(config, args)
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

