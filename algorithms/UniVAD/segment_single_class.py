from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml
import argparse

parser = argparse.ArgumentParser(description="Generate grounding+SAM masks for datasets")
parser.add_argument("--dataset", type=str, default="mvtec",
                    choices=["mvtec", "visa", "mvtec_loco", "medical", "oct17", "cognex", "sick"],
                    help="Dataset to process (default: mvtec)")
parser.add_argument("--class_name", type=str, default=None,
                    help="Specific class to process. If not provided, processes all classes in the dataset")
parser.add_argument("--dataset_root", type=str, default=None,
                    help="Custom dataset root path (for cognex or custom datasets)")
parser.add_argument("--sam_variant", type=str, default=None,
                    help="SAM variant to use (vit_b, vit_h). If not set, uses config value or default vit_h")
parser.add_argument("--sam_checkpoint", type=str, default=None,
                    help="Path to SAM checkpoint to use (overrides config)")
parser.add_argument("--num_images", type=int, default=None,
                    help="Maximum number of images to process per category (default: all images). Use None or -1 for all")
parser.add_argument("--phases", type=str, nargs="+", default=["train", "test"],
                    help="Which phases to process (default: train test)")
args = parser.parse_args()

# Convert num_images: None or -1 means all images; positive int means limit
num_images_limit = None if (args.num_images is None or args.num_images == -1) else args.num_images


# Dataset configurations
DATASET_CONFIGS = {
    "mvtec": {
        "root": "./data/mvtec",
        "mask_path": "./masks/mvtec",
        "categories": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", 
                      "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", 
                      "transistor", "wood", "zipper"],
        "extensions": ["png"],
    },
    "visa": {
        "root": "./data/VisA_pytorch/1cls",
        "mask_path": "./masks/VisA_pytorch/1cls",
        "categories": ["candle", "capsules", "chewinggum", "cashew", "fryum", 
                      "pipe_fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4"],
        "extensions": ["JPG"],
    },
    "mvtec_loco": {
        "root": "./data/mvtec_loco_caption",
        "mask_path": "./masks/mvtec_loco_caption",
        "categories": ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
        "extensions": ["png"],
    },
    "medical": {
        "root": "./data",
        "mask_path": "./masks",
        "categories": ["LiverCT", "BrainMRI", "RESC", "HIS", "ChestXray"],
        "extensions": ["png"],
    },
    "oct17": {
        "root": "./data",
        "mask_path": "./masks",
        "categories": ["OCT17"],
        "extensions": ["jpeg"],
    },
}


def read_config(config_path):
    """Load YAML config if it exists, otherwise return default config."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config
    else:
        # Default grounding config
        return {
            "grounding_config": {
                "sam_variant": "vit_h",
                "sam_checkpoint": "./pretrained_ckpts/sam_hq_vit_h.pth",
                "text_prompt": "object",
                "background_prompt": "",
                "box_threshold": 0.25,
                "text_threshold": 0.25,
            }
        }


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
        # Determine the variant to use (CLI override or config value or default)
        variant = gc.get("sam_variant", "vit_h")
        
        # If variant was changed via CLI, update the checkpoint to match
        default_ckpt_map = {
            "vit_b": "./pretrained_ckpts/sam_hq_vit_b.pth",
            "vit_h": "./pretrained_ckpts/sam_hq_vit_h.pth",
        }
        
        # Always set checkpoint based on variant (CLI override or config)
        if variant in default_ckpt_map:
            gc["sam_checkpoint"] = default_ckpt_map[variant]
        
    return config

def process_dataset(dataset_name, categories, data_root, mask_root, extensions, phases):
    """Process all categories in a dataset."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Data root: {data_root}")
    print(f"Mask output: {mask_root}")
    print(f"Categories: {len(categories)} - {', '.join(categories)}")
    print(f"{'='*60}\n")
    
    for category in categories:
        for phase in phases:
            # Build image search patterns for all extensions
            image_paths = []
            for ext in extensions:
                pattern = f"{data_root}/{category}/{phase}/*/*.{ext}"
                image_paths.extend(glob.glob(pattern))
            
            image_paths = sorted(set(image_paths))  # Remove duplicates
            
            if num_images_limit is not None:
                image_paths = image_paths[:num_images_limit]
            
            if not image_paths:
                continue
                
            config = read_config(f"./configs/class_histogram/{category}.yaml")
            config = apply_cli_sam_overrides(config, args)
            
            print(f"📸 {category}/{phase}: {len(image_paths)} images")
            
            os.makedirs(f"{mask_root}/{category}", exist_ok=True)
            grounding_segmentation(
                image_paths, f"{mask_root}/{category}", config["grounding_config"]
            )


# Main execution
if __name__ == "__main__":
    if args.dataset == "cognex":
        # Custom Cognex dataset
        if args.dataset_root is None:
            args.dataset_root = "./data/cognex_data"
        
        data_root = args.dataset_root
        dataset_name = os.path.basename(data_root.rstrip("/"))
        mask_root = f"./masks/{dataset_name}"
        
        if args.class_name:
            # Single class specified
            categories = [args.class_name]
        else:
            # Auto-discover all classes (subdirectories)
            categories = [d for d in os.listdir(data_root) 
                         if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]
        
        # Cognex uses mixed extensions
        extensions = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
        
        process_dataset("cognex", categories, data_root, mask_root, extensions, args.phases)
    
    elif args.dataset == "sick":
        # SICK dataset (same structure as Cognex)
        if args.dataset_root is None:
            args.dataset_root = "./data/sick_data"
        
        data_root = args.dataset_root
        dataset_name = os.path.basename(data_root.rstrip("/"))
        mask_root = f"./masks/{dataset_name}"
        
        if args.class_name:
            # Single class specified
            categories = [args.class_name]
        else:
            # Auto-discover all product IDs (subdirectories)
            categories = [d for d in os.listdir(data_root) 
                         if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]
        
        # SICK uses mixed extensions
        extensions = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
        
        process_dataset("sick", categories, data_root, mask_root, extensions, args.phases)
    
    else:
        # Standard datasets
        dataset_config = DATASET_CONFIGS[args.dataset]
        data_root = dataset_config["root"]
        mask_root = dataset_config["mask_path"]
        
        if args.class_name:
            # Single class specified
            if args.class_name not in dataset_config["categories"]:
                print(f"⚠️  Warning: '{args.class_name}' not in known categories for {args.dataset}")
                print(f"Available categories: {', '.join(dataset_config['categories'])}")
                print(f"Attempting to process anyway...")
            categories = [args.class_name]
        else:
            # All classes
            categories = dataset_config["categories"]
        
        process_dataset(args.dataset, categories, data_root, mask_root, 
                       dataset_config["extensions"], args.phases)

print("\n✅ Mask generation complete!")