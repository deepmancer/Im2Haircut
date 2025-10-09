import numpy as np
import trimesh
import sys
import trimesh
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import pickle
import torch
import cv2
import argparse

path = '/fast/vsklyarova/Projects/SynthHair/utils'
path2 = '/fast/vsklyarova/Projects/SynthHair'
sys.path.append(path)
sys.path.append(path2)

from geometry import load_K_Rt_from_P, project_orient_to_camera, soft_interpolate, hard_interpolate

sys.path.append('/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/Validation_CVPR')
from hair_rasterizer_opengl import HairRasterizer

sys.path.append('/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/')
from visual_utils.color_utils import vis_orient

def postprocess_with_shifted_principal_point(image, c_x, c_y, image_size):
    """
    Postprocess the image to account for the shifted principal point after projection.

    Args:
        image (np.ndarray): The rendered image (2D array).
        c_x (float): x-coordinate of the principal point.
        c_y (float): y-coordinate of the principal point.
        image_size (Tuple[int, int]): Resolution (width, height) of the image.

    Returns:
        np.ndarray: The postprocessed image.
    """
    h, w = image.shape[:2]

    # Shift the image back by subtracting the principal point offset
    # Convert the principal point from image coordinates to normalized device coordinates
    c_x_norm = (c_x - w / 2) / (w / 2)
    c_y_norm = (c_y - h / 2) / (h / 2)

    # Create a translation matrix to shift the image back
    translation_matrix = np.float32([[1, 0, c_x_norm * w], [0, 1, c_y_norm * h]])

    # Apply the translation to the image
    postprocessed_image = cv2.warpAffine(image, translation_matrix, (w, h))

    return postprocessed_image

def compute_orientation_error_with_mask(gt_image, pred_image, mask):
    """
    Compute orientation map error using MAE and MSE with a mask.

    Args:
        gt_image (np.ndarray): Ground truth orientation map.
        pred_image (np.ndarray): Predicted orientation map.
        mask (np.ndarray): Binary mask (0 or 1) indicating valid regions.

    Returns:
        float: Mean Absolute Error (MAE) within the masked region.
        float: Mean Squared Error (MSE) within the masked region.
    """
    # Ensure same dimensions
    assert gt_image.shape == pred_image.shape == mask.shape, "All inputs must have the same shape"

    # Apply mask
    valid_gt = gt_image[mask > 0]
    valid_pred = pred_image[mask > 0]

    # Compute MAE and MSE
    mae = np.mean(np.abs(valid_gt - valid_pred))
    mse = np.mean((valid_gt - valid_pred) ** 2)

    return mae, mse


def save_img(img, path):
    image = Image.fromarray((img * 255).astype(np.uint8))
    image.save(path)

#######################


def run_inference(exp_name, root, recalculate, savedir='visuals_real_dataset'):
    image_size = (512, 512)
    n_points = 200

    head_main =f'/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/data/head_prior.obj'
    head = trimesh.load(head_main)
   

    savedir = f'/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/{savedir}'

    exp_root = savedir


    print('start processing', exp_name)

    ckpt_names = sorted(os.listdir(os.path.join(exp_root, exp_name)))
#     print(ckpt_names)
    ckpt_names = [ckp for ckp in ckpt_names if 'ckpt' in ckp][::-1][:10]

    print(ckpt_names)

    for ckpt_name in ckpt_names:
        save_root = os.path.join(savedir, exp_name, ckpt_name)

        os.makedirs(save_root, exist_ok=True) 


        if os.path.exists(os.path.join(save_root, f'orient_loss_mae.txt')) and recalculate is False:
            print(f"File exists. Continuing...")
            # Perform your operation here
            continue  # This explicitly continues to the next iteration (optional in this case)

        else:
            exp_path = os.path.join(exp_root, exp_name, ckpt_name, 'pc' )


            os.makedirs(os.path.join(save_root, 'pred_mask'), exist_ok=True) 
            os.makedirs(os.path.join(save_root, 'pred_ormap'), exist_ok=True) 
            os.makedirs(os.path.join(save_root, 'pred_ormap_'), exist_ok=True) 

            exps_orients_mse = 0
            exps_orients_mae = 0
            exps_silh = 0 
            exps_orients_pi=0
            exps_results = []
            n_samples = 0

            for names in os.listdir(os.path.join(root, 'resized_img_aligned')):
                name = names.split('.')[0]
                
                try:
                    strands = trimesh.load(os.path.join(exp_path, f'{name}.ply'))
               
                    proj_matrix = np.loadtxt(f'{root}/proj_matx_inv/{name}.txt')
                    path_to_img = f'{root}/resized_img_aligned/{name}.png'
                    path_to_normal_map = f'{root}/bfm_normals/{name}.jpg'
                    path_to_gt_hair_mask= f'{root}/seg_aligned_aligned/{name}.png'
                    path_to_gt_hair_ormap =  f'{root}/orientation_maps_aligned_aligned/{name}.png'


                    strands_origins = torch.tensor(strands.vertices.reshape(-1, n_points, 3), device='cuda').float()
                    raster = HairRasterizer(strands_origins.shape[0], n_points, (np.array(head.vertices), np.array(head.faces)), resolution=image_size, line_width=0.01)

                    intrinsics, pose = load_K_Rt_from_P(None, proj_matrix)
                    pose_all_inv = torch.tensor(np.linalg.inv(pose), device='cuda').float()
                    intrinsics_all = torch.tensor(intrinsics, device='cuda').float()

                    orients = torch.zeros_like(strands_origins)
                    orients[:, :orients.shape[1] - 1] = (strands_origins[:, 1:] - strands_origins[:, :-1])
                    orients[:, orients.shape[1] - 1: ] = orients[:, orients.shape[1] - 2: orients.shape[1] - 1]
                    orients = orients.reshape(-1, 3)


                    strands_origins_copy = strands_origins.clone()
                    res = raster.rasterize(
                                            strands_origins_copy.cpu(),
                                            (intrinsics_all.cpu(), pose_all_inv.cpu()),
                                            a = torch.randn(strands_origins_copy.shape[0], 1),
                                            return_idx=True
                                          )

                    occl_idx = res[1][0]

                    valid_pixels = torch.tensor(occl_idx[occl_idx != -1], device='cuda')


                    strands_origins_ = soft_interpolate(valid_pixels, strands_origins.view(-1, 3))

                    hard_orients = hard_interpolate(valid_pixels, orients)

                    pred_loss_orients = project_orient_to_camera(
                                                                hard_orients.unsqueeze(1),
                                                                strands_origins_.unsqueeze(1),
                                                                cam_intr=intrinsics_all[None],
                                                                cam_extr=pose_all_inv[None]
                                                                 )

                    plane_orients = torch.zeros(image_size[0], image_size[1], 1, device=hard_orients.device)
                    plane_orients[occl_idx != -1, :] = pred_loss_orients
                    pred_orients = plane_orients.permute(2, 0, 1)


                    # Example usage

                    c_x = intrinsics_all[0, 2].detach().cpu().numpy()
                    c_y = intrinsics_all[1, 2].detach().cpu().numpy()

                    postprocessed_image = postprocess_with_shifted_principal_point(pred_orients.detach().cpu().numpy()[0], c_x, c_y, image_size) / 3.14

                    img = np.array(Image.open(path_to_img)) / 255.
                    img_normals = np.array(Image.open(path_to_normal_map)) / 255.

                    # Expand grayscale image to 3 channels to match RGB
                    pred_ormap = np.repeat(postprocessed_image[:, :, np.newaxis], 3, axis=2)  # Shape becomes (512, 512, 3)
                    pred_mask = pred_ormap > 0
                    gt_mask = np.array(Image.open(path_to_gt_hair_mask)) / 255.
                    gt_mask = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=2)

                    gt_ormap =  np.array(Image.open(path_to_gt_hair_ormap)) / 255.
                    gt_ormap= np.repeat(gt_ormap[:, :, np.newaxis], 3, axis=2)


                    gt_ormap_color = (vis_orient(gt_ormap[..., 0], mask=gt_mask[..., 0]).transpose(1, 2, 0)).clip(0, 1)
                    pred_ormap_color = (vis_orient(pred_ormap[..., 0], mask=pred_mask[..., 0]).transpose(1, 2, 0)).clip(0, 1)

                    save_img(pred_mask,  os.path.join(save_root, 'pred_mask', names))
                    save_img(pred_ormap_color, os.path.join(save_root,'pred_ormap', names))
                    save_img(gt_ormap_color, os.path.join(save_root,'pred_ormap', 'gt2_'+names))
                    save_img(pred_ormap, os.path.join(save_root,'pred_ormap_', names))
                    save_img(gt_ormap, os.path.join(save_root,'pred_ormap_', 'gt_'+names))
                    overlay_normals = (img_normals * (1-pred_mask) + pred_mask).clip(0, 1)
                    # Concatenate images horizontally
                    concatenated_image = np.concatenate((img, img_normals, gt_mask, pred_mask, gt_ormap_color, pred_ormap_color, overlay_normals), axis=1)  # Horizontal concatenation


                    # Example usage
                    L_orient_mae, L_orient_mse = compute_orientation_error_with_mask(gt_ormap, pred_ormap, gt_mask)

                    L_orient_mae, L_orient_mse = round(L_orient_mae, 4), round(L_orient_mse, 4)
                    L_silh = round(((gt_mask-pred_mask)**2).sum() / gt_mask.sum(), 4)

                    L_orient_pi = torch.minimum(
                        (torch.tensor(pred_ormap) - torch.tensor(gt_ormap)).abs(),
                        torch.minimum(
                            (torch.tensor(pred_ormap) - torch.tensor(gt_ormap) - 1).abs(), 
                            (torch.tensor(pred_ormap) - torch.tensor(gt_ormap) + 1).abs()
                        ))
                    L_orient_pi =(L_orient_pi * torch.pi * torch.tensor(gt_mask)).sum() / torch.tensor(gt_mask).sum()

#                     print(L_orient_pi)
                    exps_orients_mse += L_orient_mse
                    exps_orients_mae += L_orient_mae
                    exps_orients_pi += L_orient_pi
                    exps_silh += L_silh

                    exps_results.append(concatenated_image)
                    n_samples += 1
                    
                except Exception as e:
                    print('Not found', name, e)
                    continue



            fo = open(os.path.join(save_root, f'orient_loss_mae.txt'), "w")
            fo.write("orient_mae_loss:\t%5f" % (exps_orients_mae/n_samples))
            fo.close()

            fo = open(os.path.join(save_root, f'orient_loss_mse.txt'), "w")
            fo.write("orient_mse_loss:\t%5f" % (exps_orients_mse/n_samples))
            fo.close()

            fo = open(os.path.join(save_root, f'seg_loss.txt'), "w")
            fo.write("mask_loss:\t%5f" % (exps_silh/n_samples))
            fo.close()
            
            fo = open(os.path.join(save_root, f'pi_loss.txt'), "w")
            fo.write("pi_loss:\t%5f" % (exps_orients_pi/n_samples))
            fo.close()



            title = f'mean silh loss is: {exps_silh/n_samples}, orient loss mae: {exps_orients_mae/n_samples}, mse: {exps_orients_mse/n_samples}, pi_loss: {exps_orients_pi/n_samples}'

            concatenated_images = np.concatenate(exps_results, axis=0)

            plt.imshow(concatenated_images)  # Display the image
            plt.title(title, fontsize=2)  # Add title with desired fontsize and padding
            plt.axis('off')  # Turn off axes

            # Save the figure
            plt.savefig(os.path.join(save_root, "result.png"), bbox_inches="tight", pad_inches=0, dpi=1000)
            plt.show()
        
        
if __name__ == '__main__':
    print('Hello Wooden')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--root', type=str, default='/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/Validation_CVPR/dataset')
    parser.add_argument('--recalculate', type=bool, default=False)
    parser.add_argument('--savedir', type=str, default='visuals_real_dataset')
    
    args = parser.parse_args()

    print('infer on dataset', args.root)   
    run_inference(args.exp_name, args.root, args.recalculate, args.savedir)
