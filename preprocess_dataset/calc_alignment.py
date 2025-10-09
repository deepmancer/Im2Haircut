import os
import sys
import cv2
import torch
import numpy as np
import imageio
import argparse
import face_alignment
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize


# --- Utils --------------------------------------------------------------------
def load_apple_pro_depth_and_normalize(path, hair_mask):
    """Load depth from Apple Pro dataset and normalize inside hair silhouette."""
    depth = np.load(path)['depth']
    hair_silh = hair_mask > 0.5

    if not np.any(hair_silh):
        return np.zeros_like(depth)

    depth_inside = depth[hair_silh]
    min_depth, max_depth = np.min(depth_inside), np.max(depth_inside)

    normalized = depth.copy()
    normalized[hair_silh] = (depth_inside - min_depth) / (max_depth - min_depth)
    normalized[~hair_silh] = 0
    return normalized


def ensure_dir(path):
    """Create a directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


def align_image(img, matrix, size=None):
    """Apply affine alignment to an image or array."""
    h, w = img.shape[:2]
    out_w, out_h = (w, h) if size is None else size
    return cv2.warpAffine(img, matrix, (out_w, out_h))


# --- Main ---------------------------------------------------------------------
def main(args, device="cuda"):
    img_path = args.img_path
    hair_silh_path = args.hair_path
    gt_img_path = args.gt_img_path

    # Output root folder
    last = os.path.basename(img_path.rstrip("/"))
    path_to_save = img_path.replace(last, f"{last}_aligned")
    ensure_dir(path_to_save)


    # Face alignment setup
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # Load ground-truth target landmarks
    img_gt = Image.open(gt_img_path).resize((512, 512))
    img_gt_np = np.array(img_gt)
    lmks_gt = fa.get_landmarks_from_image(img_gt_np)[0]

    images_list = sorted(os.listdir(img_path))

    for idx, img_name in enumerate(images_list):
        try:
            # Load input image + mask
            pil_img = Image.open(os.path.join(img_path, img_name))
            img = np.array(pil_img)
            if img.ndim == 2:  # grayscale
                img = img[..., None]

            pil_hair = os.path.join(hair_silh_path, img_name)

            # Detect landmarks
            lmks = fa.get_landmarks_from_image(img[:, :, :3])[0]

            # Compute alignment
            matrix, _ = cv2.estimateAffinePartial2D(lmks, lmks_gt)
            aligned_image = align_image(img, matrix)

            # Save aligned face image
            cv2.imwrite(os.path.join(path_to_save, img_name), aligned_image[:, :, ::-1])

            # Process auxiliary folders
            for folder in args.all_paths_for_processing:
                try:
                    folder_src = img_path.replace(last, folder)
                    folder_dst = path_to_save.replace(last, f"{folder}")
                    
                    ensure_dir(folder_dst)

                    if folder == "depth_apple_pro":
                        depth_path = os.path.join(folder_src, img_name.replace(".png", ".npz").replace(".jpg", ".npz"))
                        mask_img = np.array(Image.open(pil_hair).resize((512, 512)))
                        mask_hair = (mask_img[..., 0] if mask_img.ndim > 2 else mask_img) / 255.
                        img_folder = resize(load_apple_pro_depth_and_normalize(depth_path, mask_hair), (512, 512), anti_aliasing=True)

                    elif folder == "confidence_maps":
                        npy_path = os.path.join(folder_src, img_name.split(".")[0] + ".npy")
                        img_folder = np.load(npy_path)

                    elif folder == "strand_map":
                        pil_strand = Image.open(os.path.join(folder_src, img_name))
                        mask_img = np.array(Image.open(pil_hair).resize((512, 512))) / 255.
                        if mask_img.ndim > 2:
                            mask_img = mask_img[..., 0]
                        img_folder = np.array(pil_strand) * mask_img[..., None]

                    else:
                        pil_generic = Image.open(os.path.join(folder_src, img_name)).resize((512, 512))
                        img_folder = np.array(pil_generic)
                        if img_folder.ndim > 2:
                            if folder in ("seg", "body_img"):
                                img_folder = img_folder[..., 0]
                            else:
                                img_folder = img_folder[..., :3]

                    # Align and save
                    aligned_folder_img = align_image(img_folder, matrix)

                    if folder == "confidence_maps":
                        np.save(os.path.join(folder_dst, img_name.split(".")[0] + ".npy"), aligned_folder_img)
                    elif folder == "depth_apple_pro":
                        out_name = img_name.replace(".png", ".npz").replace(".jpg", ".npz")
                        np.savez_compressed(os.path.join(folder_dst, out_name), depth=aligned_folder_img)
                    else:
                        if aligned_folder_img.ndim > 2 and folder != "depth_vis_map_aligned":
                            cv2.imwrite(os.path.join(folder_dst, img_name), aligned_folder_img[:, :, ::-1])
                        else:
                            cv2.imwrite(os.path.join(folder_dst, img_name), aligned_folder_img)

                except Exception as e:
                    print(f"[WARN] Error processing {folder}/{img_name}: {e}")

        except Exception as e:
            print(f"[WARN] Skipping {img_name}: {e}")


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--img_path', default= './resized_img', type=str)
    parser.add_argument('--hair_path', default= './seg', type=str)
    parser.add_argument('--all_paths_for_processing', nargs='+', type=str, help="List of image paths")
    parser.add_argument('--gt_img_path', default= './data/aligned_image.png', type=str)



    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)  