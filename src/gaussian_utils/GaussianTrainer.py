import torch
import sys
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import math


sys.path.append('./submodules/external/GaussianHaircut')
from src.utils.image_utils import vis_orient
from src.utils.loss_utils import l1_loss,  or_loss, ssim

import torch
import torch.nn.functional as F

from src.loss_utils.losses import or_loss_directed
from src.gaussian_utils.gaussian_renderer import render, render_hair, network_gui
from src.gaussian_utils.scene.gaussian_model import GaussianModel
from src.gaussian_utils.scene.gaussian_model_strands import GaussianModelCurves
from src.gaussian_utils.scene.cameras import CameraMini
from src.gaussian_utils.image_utils import vis_directed_orient
from src.camera_utils.projection import load_K_Rt_from_P


def normalize_depth(depth, eps=1e-6):
    min_val = depth.min()
    max_val = depth.max()
    range_val = max_val - min_val
    if range_val < eps:
        return torch.zeros_like(depth)  # or torch.ones_like(depth), or depth itself
    return (depth - min_val) / range_val


def scale_matrix(mat, scale_factor):
    mat[0, 0] /= scale_factor
    mat[1, 1] /= scale_factor
    mat[0, 2] /= scale_factor
    mat[1, 2] /= scale_factor
    return mat


def flip_hairstyle(strands):
    # Flip the x-coordinate (horizontal axis)
    strands_flipped = strands.clone()  # Ensure a copy for gradient preservation
    strands_flipped[:, :, 0] = -strands[:, :, 0]  # Flip x-axis
    # y and z coordinates remain unchanged
    return strands_flipped


def obtain_camera(cam, scaling_factor, resolution): 

    # suppose images are square

    intrinsics, pose = load_K_Rt_from_P(None, cam)
    
    intrinsics = (scale_matrix(torch.from_numpy(intrinsics).float(), scaling_factor))
    pose = torch.from_numpy(pose).float()
    extrinsics = torch.inverse(pose)
    
    
    R = np.transpose(extrinsics[:3,:3].cpu().numpy())  # R is stored transposed due to 'glm' in CUDA code
    T = extrinsics[:3, 3].cpu().numpy()
    

    fx = intrinsics[0][0].cpu()
    fy = intrinsics[1][1].cpu()
    
    cx = intrinsics[0][2].cpu()
    cy = intrinsics[1][2].cpu()

    FoVx = 2 * math.atan(resolution[0] / 2 / fx)
    FoVy = 2 * math.atan(resolution[1] / 2 / fy)
    
    return CameraMini(R=R, T=T, FoVx=FoVx, FoVy=FoVy, width=resolution[0], height=resolution[1], cx=cx, cy=cy)


class GaussianTrainer(nn.Module):
    def __init__(self,
                 dataset=None,
                 opt=None,
                 pipe=None,
                 pointcloud_path_head=None, 
                 ip=None,
                 port=None,
                 scale_matx_path=None,
                 use_conf=False,
                 gaussian_width=0.008,
                 use_directed_loss=False, 
                 loss_type='l1',
                 optimize_appearance=False,
                device=None):
        
        
        super().__init__()
        
        with open(scale_matx_path, 'rb') as f:
            transform = pickle.load(f)
            
        print('gaussian width is', gaussian_width)
        self.loss_type = loss_type
        self.use_directed_loss = use_directed_loss
        self.transform = transform
        self.translate_to_sphere = torch.tensor(transform['translation'], device=device).float()
        self.scale_to_sphere = torch.tensor(transform['scale'], device=device).float()
        
        self.use_conf = use_conf
#         print('confidence', use_conf)
        self.optimize_appearance = optimize_appearance
#         print('gaussian width', gaussian_width)
        self.first_iter = 0
        
        self.pipe = pipe
        self.dataset = dataset
        
        self.opt = opt
        
        self.gaussians = GaussianModel(dataset.sh_degree, device=device)

        self.gaussians_hair = GaussianModelCurves(dataset.sh_degree, scale=gaussian_width, device=device)

        self.gaussians.load_ply(pointcloud_path_head)

        self.spatial_lr_scale = 1. if self.gaussians.spatial_lr_scale == 0 else self.gaussians.spatial_lr_scale

        self.device = device
        
        with torch.no_grad():
            # Head gaussians data
            self.gaussians.mask_precomp = self.gaussians.get_label[..., 0] < 0.5
            self.gaussians.xyz_precomp = self.gaussians.get_xyz[self.gaussians.mask_precomp].detach()
            self.gaussians.opacity_precomp = self.gaussians.get_opacity[self.gaussians.mask_precomp].detach()
            self.gaussians.scaling_precomp = self.gaussians.get_scaling[self.gaussians.mask_precomp].detach()
            self.gaussians.rotation_precomp = self.gaussians.get_rotation[self.gaussians.mask_precomp].detach()
            self.gaussians.cov3D_precomp = self.gaussians.get_covariance(1.0)[self.gaussians.mask_precomp].detach()
            self.gaussians.shs_view = self.gaussians.get_features[self.gaussians.mask_precomp].detach().transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1)**2)

        bg_color = [1, 1, 1, 0, 0, 0, 0, 0, 0, 7] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        # Start GUI server, configure and run training
        network_gui.init(ip, port)
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_hair(custom_cam, self.gaussians, self.gaussians_hair, self.pipe, self.background, self.scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
            except Exception as e:
                network_gui.conn = None

                
    def render(self, cam_sample, selected_strands, scaling_factor, sample_flip, resolution, gt_mask, appearance=None):


        selected_strands = (selected_strands - self.translate_to_sphere ) / self.scale_to_sphere 
        
        features_dc = None
        features_rest = None
        

        if self.optimize_appearance:

            selected_app = appearance
            if sample_flip.item() > 0:
                flipped_app = flip_hairstyle(selected_app)
                
                    
            input_app = flipped_app if sample_flip.item() > 0 else selected_app
#             features_dc = input_app
            features_dc, features_rest = torch.split(input_app, [3, 45], dim=-1)
    
    
        
        if sample_flip.item() > 0:
            flipped_strands = flip_hairstyle(selected_strands)
        # strands, baldness mask define which ot optimize
        
        
        input_strands = flipped_strands if sample_flip.item() > 0 else selected_strands
        self.gaussians_hair.update_gaussians_hair(input_strands, features_dc, features_rest)

        
        
        imgs_all = []
        masks_all = []
        orient_angle_all = []
        orient_conf_all = []
        depths_all = []
        
        
        n_views= cam_sample.shape[0]
        
        for view in range(n_views):
            camera_gs = obtain_camera(cam_sample[view].detach().cpu().numpy(), scaling_factor, resolution=resolution)

            render_pkg = render_hair(camera_gs, self.gaussians, self.gaussians_hair, self.pipe, self.background, render_direction=self.opt.render_direction, use_directed_loss=self.use_directed_loss)

            image = render_pkg["render"]
            mask = render_pkg["mask"]
            orient_angle = render_pkg["orient_angle"]
            orient_conf = render_pkg["orient_conf"]            
            depth_pred = render_pkg['depth']

        
            if sample_flip.item() > 0:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                orient_angle = torch.flip(orient_angle, dims=[-1])
                orient_conf = torch.flip(orient_conf, dims=[-1])
                depth_pred = torch.flip(depth_pred, dims=[-1])
                

            normalized_depth = normalize_depth(depth_pred[0]) * gt_mask[0][0]
            
            imgs_all.append(image)
            masks_all.append(mask)
            orient_angle_all.append(orient_angle)
            orient_conf_all.append(orient_conf)
            depths_all.append(normalized_depth[None])

        return torch.stack(imgs_all), torch.stack(masks_all), torch.stack(orient_angle_all), torch.stack(orient_conf_all), torch.stack(depths_all)
        
  
    def log_to_tensorboard(self, tb_writer, gt, pred, iteration, mode):
        def clamp(x): return torch.clamp(x, 0.0, 1.0)

        tb_writer.add_images(f"{mode}/render", clamp(pred["image"][0:1]), global_step=iteration)
        tb_writer.add_images(f"{mode}/render_mask", F.pad(pred["mask"][0], (0, 0, 0, 0, 0, 3 - pred["mask"].shape[1]), 'constant', 0)[None], global_step=iteration)
        tb_writer.add_images(f"{mode}/render_depth", clamp(pred["depth"][0:1]), global_step=iteration)

        tb_writer.add_images(f"{mode}/ground_truth", clamp(gt["image"][0:1]), global_step=iteration)
        tb_writer.add_images(f"{mode}/ground_truth_mask", F.pad(gt["mask"][0], (0, 0, 0, 0, 0, 3 - gt["mask"].shape[1]), 'constant', 0)[None], global_step=iteration)
        tb_writer.add_images(f"{mode}/ground_truth_depth", clamp(gt["depth"][0:1]), global_step=iteration)

        if self.use_conf:
            conf_vis = (1 - 1 / (pred["orient_conf"][0][0] + 1)) * pred["mask"][0, :1]
            gt_conf_vis = (1 - 1 / (gt["orient_conf"][0][0] + 1)) * gt["mask"][0, :1]
            tb_writer.add_images(f"{mode}/render_conf", vis_orient(pred["orient_angle"][0], conf_vis)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_conf", vis_orient(gt["orient_angle"][0], gt_conf_vis)[None], global_step=iteration)

        if self.use_directed_loss and gt["directed_map"] is not None:
            tb_writer.add_images(f"{mode}/ground_truth_dir_orient", torch.cat((gt["mask"][0, :1], gt["directed_map"][0, 1:]), 0)[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/pred_dir_orient", pred["mask"][0, :1] * torch.cat((pred["mask"][0, :1], vis_directed_orient(pred["orient_angle"])[0]), 0)[None], global_step=iteration)
        else:
            tb_writer.add_images(f"{mode}/render_orient", vis_orient(pred["orient_angle"][0], pred["mask"][0, :1])[None], global_step=iteration)
            tb_writer.add_images(f"{mode}/ground_truth_orient", vis_orient(gt["orient_angle"][0], gt["mask"][0, :1])[None], global_step=iteration)

            
    def render_and_parse_feats(self, strands, gt_cam, feats, scaling_factor, flip, appearance=None):
        
        gt_directed_map = None
        if feats.shape[2] == 8:
            gt_image, gt_mask, gt_orient_angle, gt_orient_conf, gt_depth = torch.split(feats, [3, 2, 1, 1, 1], dim=2)
        else:
            gt_image, gt_mask, gt_orient_angle, gt_orient_conf, gt_depth, gt_directed_map = torch.split(feats, [3, 2, 1, 1, 1, 3], dim=2)

        bs, nviews = gt_image.shape[0], gt_cam.shape[1]
        image_preds, mask_preds, angle_preds, conf_preds, depth_preds = [], [], [], [], []

        for idx in range(bs):
            resolution = gt_image.shape[-2:][::-1]  # (W, H)
            cam_sample = gt_cam[idx]
            strands_sample = strands[idx]
            sample_flip = flip[idx]

            image, mask, angle, conf, depth = self.render(
                cam_sample, strands_sample, scaling_factor, sample_flip,
                resolution, gt_mask[idx], appearance=appearance
            )

            image_preds.append(image.unsqueeze(0))
            mask_preds.append(mask.unsqueeze(0))
            angle_preds.append(angle.unsqueeze(0))
            conf_preds.append(conf.unsqueeze(0))
            depth_preds.append(depth.unsqueeze(0))

        def reshape(x): return x.reshape(bs * nviews, *x.shape[-3:])

        pred = {
            "image": reshape(torch.cat(image_preds, dim=0)),
            "mask": reshape(torch.cat(mask_preds, dim=0)),
            "orient_angle": reshape(torch.cat(angle_preds, dim=0)),
            "orient_conf": reshape(torch.cat(conf_preds, dim=0)),
            "depth": reshape(torch.cat(depth_preds, dim=0)),
        }

        gt = {
            "image": reshape(gt_image),
            "mask": reshape(gt_mask),
            "orient_angle": reshape(gt_orient_angle),
            "orient_conf": reshape(gt_orient_conf),
            "depth": reshape(gt_depth),
            "directed_map": reshape(gt_directed_map) if gt_directed_map is not None else None,
        }

        return {"gt": gt, "pred": pred}
        
        
    def compute_losses(self, gt, pred):
        L1 = l1_loss(pred["image"], gt["image"])
        SSIM = 1.0 - ssim(pred["image"], gt["image"])
        Lmask = l1_loss(pred["mask"], gt["mask"])
        Ldepth = l1_loss(pred["depth"], gt["depth"])

        orient_weight = torch.ones_like(gt["mask"][:, :1])
        if self.use_conf:
            orient_weight *= gt["orient_conf"]
        else:
            pred["orient_conf"] = None

        if self.use_directed_loss and gt["directed_map"] is not None:
            Lorient = or_loss_directed(
                vis_directed_orient(pred["orient_angle"]),
                gt["directed_map"][:, 1:],
                pred["orient_conf"],
                weight=orient_weight,
                mask=gt["mask"][:, :1],
                type=self.loss_type
            )
        else:
            Lorient = or_loss(
                pred["orient_angle"],
                gt["orient_angle"],
                pred["orient_conf"],
                weight=orient_weight,
                mask=gt["mask"][:, :1]
            )

        return {"l1": L1, "ssim": SSIM, "mask": Lmask, "orient": Lorient, "depth": Ldepth}
       
            
    def step(self, strands, gt_cam, feats, scaling_factor, idx=0, iteration=0, tb_writer=None, mode='train', flip=None, appearance=None, cam_idxes=None):
    #         if cam_idxes <= 18 or cam_idxes>=62:
#             self.use_directed_loss=True
#         else:
#             self.use_directed_loss=False
#         print(strands.shape)
        parsed = self.render_and_parse_feats(strands, gt_cam, feats, scaling_factor, flip, appearance)
        gt, pred = parsed['gt'], parsed['pred']

        losses = self.compute_losses(gt, pred)

        if tb_writer is not None:
            self.log_to_tensorboard(tb_writer, gt, pred, iteration, mode)

        return losses["l1"], losses["ssim"], losses["mask"], losses["orient"], losses["depth"]

