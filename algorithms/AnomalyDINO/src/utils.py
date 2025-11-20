import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def augment_image(img_ref, augmentation = "rotate", angles = [0, 45, 90, 135, 180, 225, 270, 315]):
    """
    Simply augmentation of images, currently just rotation.
    """
    imgs = []
    if augmentation == "rotate":
        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    return result


def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    mask = np.repeat(mask, imgd2, axis=1)
    return mask


def plot_ref_images(img_list, mask_list, vis_background_list, grid_size, save_path, title = "Reference Images", img_names = None):
    k = min(len(img_list), 32)  # reduce max number of ref samples to plot to 32

    n_aug = len(img_list)//len(img_names)

    fig, axs = plt.subplots(k, 3, figsize=(10, 3.5*k))
    if k == 1:
        axs = axs.reshape(1, -1)
    for i in range(k):
        axs[i, 0].imshow(img_list[i])
        axs[i, 1].imshow(vis_background_list[i])
        axs[i, 2].imshow(img_list[i])
        axs[i, 2].imshow(resize_mask_img(mask_list[i], img_list[i].shape, grid_size), alpha=0.5)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        if i % n_aug == 0:
            axs[i, 0].title.set_text(f"Image: {img_names[i // n_aug]}")
        else:
            axs[i, 0].title.set_text(f"Augmentation of Image {img_names[i // n_aug]}")
        axs[i, 1].title.set_text("PCA + Mask")
        axs[i, 2].title.set_text("Mask")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "reference_samples.png")
    plt.close()


def get_dataset_info(dataset, preprocess):

    if preprocess not in ["informed", "agnostic", "masking_only", "informed_no_mask", "agnostic_no_mask", "force_no_mask_no_rotation", "force_mask_no_rotation", "force_no_mask_rotation", "force_mask_rotation"]:
        # masking only: deactivate rotation, apply masking like in informed/agnostic
        raise ValueError(f"Preprocessing '{preprocess}' not yet covered!")
    
    if dataset == "MVTec":
        objects = ['bottle']
        # objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        object_anomalies = {"bottle": ["broken_large", "broken_small", "contamination"],
                            "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_wire", "missing_cable", "poke_insulation"],
                            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
                            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
                            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
                            "hazelnut": ["crack", "cut", "hole", "print"],
                            "leather": ["color", "cut", "fold", "glue", "poke"],
                            "metal_nut": ["bent", "color", "flip", "scratch"],
                            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
                            "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
                            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
                            "toothbrush": ["defective"], 
                            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
                            "wood": ["color", "combined", "hole", "liquid", "scratch"],
                            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]
                            }

        if preprocess in ["agnostic", "informed", "masking_only"]:
            # Define Masking for the different objects -> determine with Masking Test (see Fig. 2 and discussion in the paper)
            # True: default masking (threshold the first PCA component > 10)
            # False: No masking will be applied
            masking_default = {"bottle": False,      
                                "cable": False,         # no masking
                                "capsule": True,        # default masking
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,
                                "leather": False,
                                "metal_nut": False,
                                "pill": True,
                                "screw": True,
                                "tile": False,
                                "toothbrush": True,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }
            
        if preprocess in ["informed", "informed_no_mask"]:
            rotation_default = {"bottle": False,
                                "cable": False, 
                                "capsule": False,
                                "carpet": False,
                                "grid": False,
                                "hazelnut": True,       # informed: hazelnut is rotated
                                "leather": False,
                                "metal_nut": False,
                                "pill": False,          # informed: all pills in train are oriented just the same
                                "screw": True,          # informed: screws in train are oriented differently
                                "tile": False,
                                "toothbrush": False,
                                "transistor": False,
                                "wood": False,
                                "zipper": False
                                }

        elif preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess == "masking_only":
            rotation_default = {o: False for o in objects}

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}

    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        object_anomalies = {"candle": ["bad"],
                            "capsules": ["bad"],
                            "cashew": ["bad"],
                            "chewinggum": ["bad"],
                            "fryum": ["bad"],
                            "macaroni1": ["bad"],
                            "macaroni2": ["bad"],
                            "pcb1": ["bad"],
                            "pcb2": ["bad"],
                            "pcb3": ["bad"],
                            "pcb4": ["bad"],
                            "pipe_fryum": ["bad"],
                            }

        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}

        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}

    # added cognex dataset
    elif dataset == "Cognex":
        # Exact folder name(s) of your product(s)
        objects = [ "'t_Slagershuys_Kipdij_ovenschotel_teriyaki_11738318",
                    "Arla_Halfvolle_melk_lactofree_10760273",
                    "Elinas_Yoghurt_Griekse_stijl_aardbei_11400153",
                    "Heemskerk_Zoete_aardappelblokjes_11463862",
                    "Merkloos_Bio_winterpeen_10691308",
                    "Merkloos_Pink_Lady_appels_90006052",
                    "Merkloos_Rozemarijn_10074790",
                    "Vischmeesters_Kabeljauwhaas_11829912"
                ]
        # objects = ["Merkloos_Elstar_appels_11298357", "Picnic_Geraspte_jong_belegen_kaas_48plus_11695484" ]
        # objects = ["Picnic_Geraspte_jong_belegen_kaas_48plus"]
        # Automatically map all objects to the 'tote_photos' anomaly type
        object_anomalies = {obj: ["tote_photos"] for obj in objects}

        # --- Preprocessing Configuration ---
        
        # Masking Logic (matches VisA style):
        # Default is True, unless explicitly disabled with '_no_mask'
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects} 

        # Rotation Logic:
        # Agnostic = try rotation. Informed/others = no rotation (safer start).
        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        else:
            rotation_default = {o: False for o in objects}

    # added sick dataset
    elif dataset == "Sick":
        # Sick dataset products
        objects = ["10074656", "10074666", "10074790", "10762299", 
                   "11478299", "90006036", "90006124"]
        # Automatically map all objects to the 'issue' anomaly type
        object_anomalies = {obj: ["issue"] for obj in objects}

        # --- Preprocessing Configuration ---
        
        # Masking Logic (matches VisA style):
        # Default is True, unless explicitly disabled with '_no_mask'
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects} 

        # Rotation Logic:
        # Agnostic = try rotation. Informed/others = no rotation (safer start).
        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        else:
            rotation_default = {o: False for o in objects}

    else:
        raise ValueError(f"Dataset '{dataset}' not yet covered!")

    if preprocess == "force_no_mask_no_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_mask_no_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: False for o in objects}
    elif preprocess == "force_no_mask_rotation":
        masking_default = {o: False for o in objects}
        rotation_default = {o: True for o in objects}
    elif preprocess == "force_mask_rotation":
        masking_default = {o: True for o in objects}
        rotation_default = {o: True for o in objects}

    return objects, object_anomalies, masking_default, rotation_default