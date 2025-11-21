import os
import json


class SickSolver(object):
    # SICK product IDs - will be auto-discovered from directory structure
    
    def __init__(self, root="./sick_data"):
        self.root = root
        self.meta_path = f"{root}/meta.json"

    def run(self):
        info = dict(train={}, test={})
        
        # Auto-discover product directories (numeric folder names)
        product_dirs = [d for d in os.listdir(self.root) 
                       if os.path.isdir(os.path.join(self.root, d)) and d.isdigit()]
        product_dirs.sort()
        
        for cls_name in product_dirs:
            cls_dir = f"{self.root}/{cls_name}"
            
            for phase in ["train", "test"]:
                phase_dir = f"{cls_dir}/{phase}"
                if not os.path.exists(phase_dir):
                    continue
                    
                cls_info = []
                species = os.listdir(phase_dir)
                
                for specie in species:
                    specie_dir = f"{phase_dir}/{specie}"
                    if not os.path.isdir(specie_dir):
                        continue
                        
                    is_abnormal = True if specie not in ["good"] else False
                    img_names = [f for f in os.listdir(specie_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
                    mask_names = os.listdir(f"{cls_dir}/ground_truth/{specie}") if is_abnormal and os.path.exists(f"{cls_dir}/ground_truth/{specie}") else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f"{cls_name}/{phase}/{specie}/{img_name}",
                            mask_path=f"{cls_name}/ground_truth/{specie}/{mask_names[idx]}" if is_abnormal and mask_names else "",
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                
                if cls_info:
                    info[phase][cls_name] = cls_info
                    
        with open(self.meta_path, "w") as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == "__main__":
    runner = SickSolver(root="./data/sick_data")
    runner.run()
