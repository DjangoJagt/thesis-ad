import os
import json


class CognexSolver(object):
    CLSNAMES = [
        "Arla_Halfvolle_melk_lactofree_10760273",
        "Elinas_Yoghurt_Griekse_stijl_aardbei_11400153",
        "Heemskerk_Zoete_aardappelblokjes_11463862",
        "Merkloos_Bio_winterpeen_10691308",
        "Merkloos_Elstar_appels_11298357",
        "Merkloos_Pink_Lady_appels_90006052",
        "Merkloos_Rozemarijn_10074790",
        "Picnic_Geraspte_jong_belegen_kaas_48plus_11695484",
        "'t_Slagershuys_Kipdij_ovenschotel_teriyaki_11738318",
        "Vischmeesters_Kabeljauwhaas_11829912",
    ]

    def __init__(self, root="./cognex_data"):
        self.root = root
        self.meta_path = f"{root}/meta.json"

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f"{self.root}/{cls_name}"
            
            # Skip if class directory doesn't exist
            if not os.path.exists(cls_dir):
                print(f"Warning: {cls_name} directory not found, skipping...")
                continue
                
            for phase in ["train", "test"]:
                phase_dir = f"{cls_dir}/{phase}"
                if not os.path.exists(phase_dir):
                    print(f"Warning: {phase_dir} not found, skipping...")
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
        
        print(f"Created meta.json with {len(info['train'])} train classes and {len(info['test'])} test classes")


if __name__ == "__main__":
    runner = CognexSolver(root="./data/cognex_data")
    runner.run()
