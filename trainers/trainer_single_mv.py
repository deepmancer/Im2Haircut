# === Standard Library ===
import os
import sys
import random
import argparse
import pickle

# === Third-Party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import cv2
import yaml
from pyhocon import ConfigFactory

# === Project-Specific Modules ===
sys.path.append("./submodules/external/GaussianHaircut")
from src.utils.general_utils import safe_state

from src.model_utils.geometry import decode_pca
from src.datasets.real_dataset_multiview import HairstyleRealDatasetMultiView

# Losses and model utils
from src.loss_utils.head_sdf_prior import SDFHeadPrior
from src.loss_utils.scalp_renderer import ScalpRenderer

from src.gaussian_utils.GaussianTrainerMV import GaussianTrainerMV

from src.processing_utils.upsampling import calc_strands_similarity
from src.model_utils.file_utils import file_backup
from src.model_utils.save_utils import save_strands
from src.model_utils.geometry import compute_similarity_transform, can2world_transform

from src.model_utils.get_projector import create_projector_backbone

# Arguments / config handling
from src.gaussian_utils.arguments import ModelParams, PipelineParams, OptimizationParams

# Distributed training utilities
from src.model_utils import distributed as dist

# === Environment and Torch Settings ===
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import trimesh


class BaseTrainerMultiView(nn.Module):
    """
    Base trainer class for multi-view hair reconstruction.
    
    Key differences from single-view trainer:
    1. Uses HairstyleRealDatasetMultiView which loads all views of a subject
    2. The encoder uses only the frontal view for prediction
    3. All views are used for 3DGS supervision loss
    """
    
    def _init_basic_config(self, num_steps_coarse, device, ngpus, accumulate_gradients,
                       upsample_hairstyle, upsample_resolution, optimize_appearance, config,
                       unfreeze_time_for_pca):

        self.num_steps_coarse = num_steps_coarse
        self.device = device
        self.ngpus = ngpus
        print('device:', self.device, '| GPUs:', ngpus, '| accumulate_gradients:', accumulate_gradients)

        self.upsample_hairstyle = upsample_hairstyle
        self.blend_func = lambda x: torch.where(x <= 0.9, 1 - 1.63 * x**5, 0.4 - 0.4 * x)
        self.roots_origins_up = torch.load(
            f'./data/coords_for_each_origin_{upsample_resolution}x{upsample_resolution}.pth'
        ).float().to(self.device)[None]

        self.optimize_appearance = optimize_appearance
        self.config = config
        self.num_points = config['dataset'].get('num_points', 200)
        self.num_components = config['dataset'].get('num_components', 64)
        self.texture_size = config['dataset'].get('texture_size', 64)
        
        self.resolution_upsample = upsample_resolution
        self.scalp_render = ScalpRenderer(size=(512, 512))
        self.accumulate_gradients = accumulate_gradients
        self.unfreeze_time_for_pca = unfreeze_time_for_pca

        self.all_steps = config['visuals_config'].get('num_epochs', 1)

        self.path_to_coords_for_each_origin = f'./data/coords_for_each_origin_64x64.pth'
        self.global_mean_path = config['pca_basis'].get('global_mean_path', '')
        self.mean_shape_path = config['pca_basis'].get('mean_shape_path', '')
        self.blend_shape_path = config['pca_basis'].get('blend_shape_path', '')

   
    def _init_config(self, config):
        self.scalp_mask_prediction = None
        visuals_cfg = config['visuals_config']
        self.use_scale = config['dataset'].get('use_scale', False)
        self.logging_freq = visuals_cfg['logging_freq']
        self.save_freq = visuals_cfg['save_freq']
        self.eval_freq = visuals_cfg['eval_freq']
        self.pc_freq = visuals_cfg['pc_freq']
        self.num_epochs = visuals_cfg['num_epochs']

        loss_cfg = config['loss_config']
        self.finetune_coarse_model = loss_cfg.get('finetune_coarse_model', False)
        self.penetration_weight = loss_cfg.get('penetration_weight', 0.0)
        self.gaus_l1 = loss_cfg.get('gaus_l1_loss', 0)
        self.gaus_ssim = loss_cfg.get('gaus_ssim_loss', 0)
        self.gaus_mask = loss_cfg.get('gaus_mask_loss', 0)
        self.gaus_orient = loss_cfg.get('gaus_orient_loss', 0)
        self.gaus_depth = loss_cfg.get('gaus_depth_loss', 0)
        self.gaus_bald_mask = loss_cfg.get('gaus_bald_mask', 0)
        self.sdf_penalty = loss_cfg.get('sdf_penalty', 0)
        self.scale_output = loss_cfg.get('scale_output', False)
        self.dilate_mask = loss_cfg.get('dilate_mask', False)
        self.transformer_mask_size = loss_cfg.get('transformer_mask_size', 32)
        self.learning_rate = config['optconfig']['lr']
        self.weight_decay = config['optconfig'].get('weight_decay', 0.001)
        self.optimizer_type = config['optconfig'].get('optimizer_type', 'adam')
        
        scale_stats_path = loss_cfg.get('scale_stats_path', '')
        try:
            if scale_stats_path:
                with open(scale_stats_path, "rb") as file:
                    scale_stats = pickle.load(file)
                self.scale_stats_mean = torch.tensor(scale_stats['mean'], device=self.device).float()
                self.scale_stats_std = torch.tensor(scale_stats['std'], device=self.device).float()
        except Exception as e:
            print(f"Failed to load scale stats from {scale_stats_path}: {e}")
            
        if self.sdf_penalty > 0:
            ckpt = './pretrained_models/neus.pth'
            self.sdf = SDFHeadPrior(ckpt, apply_relu=False, device=self.device)
                 
        self.colors_save = None
        self.edited_uvmap = None

        
    def _init_dataset_multiview(self, config, subject, folder_name, world_size, rank, num_workers):
        """Initialize multi-view dataset for a specific subject."""
        data_path = config['dataset'].get('data_path', '')
        
        self.data_path = data_path
        self.subject = subject
        
        # Use multi-view dataset
        self.real_set_train = HairstyleRealDatasetMultiView(
            **config['dataset_real'],
            infer_path=f'{data_path}/{folder_name}/{subject}',
            subject=subject
        )

        self.num_workers = num_workers

        self.train_dl = data.DataLoader(
            self.real_set_train,
            config['optconfig']['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=num_workers
        )
        
        # Store number of views for logging
        self.n_views = self.real_set_train.n_views
        
        # Print multi-view setup summary
        print('\n' + '='*60)
        print('MULTI-VIEW DATASET CONFIGURATION')
        print('='*60)
        print(f'  Subject:       {subject}')
        print(f'  Data path:     {data_path}/{folder_name}/{subject}')
        print(f'  Num views:     {self.n_views}')
        print(f'  View files:    {self.real_set_train.view_list}')
        print(f'  Frontal view:  {self.real_set_train.view_list[self.real_set_train.frontal_idx]}')
        print('='*60 + '\n')

        
    def _init_roots_and_blend_shapes(self):
        self.roots_origins = torch.load(self.path_to_coords_for_each_origin)[None].float().to(self.device)
        global_mean_shape = torch.tensor(np.load(self.global_mean_path), device=self.device).float()
        mean_shape_local = torch.tensor(np.load(self.mean_shape_path), device=self.device).float()
        self.mean_shape = global_mean_shape + mean_shape_local
        self.blend_shapes = torch.tensor(np.load(self.blend_shape_path), device=self.device).float()
        

    def _init_encoders(self, config, device, rank):
        self.projector_type = config['projector'].get('projector_type', '')
        self.lp_enc = create_projector_backbone(self.projector_type, config).to(device)

        projector_type_elow = config['projector_type_elow'].get('projector_type', '')
        ckpt_path_elow = config['lp_encoder_fine']['ckpt_path_elow']
        lp_enc_elow = self.create_coarse_model(projector_type_elow, config, ckpt_path_elow, device, finetune_coarse_model=self.finetune_coarse_model)

        if self.finetune_coarse_model:
            self.lp_enc_elow = lp_enc_elow.to(device)
        else:
            self.lp_enc_elow = lp_enc_elow

            
    def _init_gaussian_trainer(self, dataset, opt, pipe, pointcloud_path_head, ip, port, rank, config):
        self.gaus_trainer = GaussianTrainerMV(
            dataset=dataset,
            opt=opt,
            pipe=pipe,
            pointcloud_path_head=pointcloud_path_head,
            ip=ip,
            port=port + rank,
            gaussian_width=config['gaussians'].get('gaussian_width', 0.008),
            scale_matx_path=config['dataset_real'].get('gs_scale_path', ''),
            use_conf=config['loss_config'].get('use_conf', False),
            use_directed_loss=config['gaussians'].get('use_directed_loss', False),
            loss_type=config['gaussians'].get('loss_type', "min"),
            optimize_appearance=self.optimize_appearance,
            device=self.device
        )

        
    def create_coarse_model(self, projector_type_elow, config, ckpt_path_elow, device, finetune_coarse_model):
        elow = create_projector_backbone(projector_type_elow, config)
        checkpoint = torch.load(ckpt_path_elow, map_location=device)
        state_dict = checkpoint['lp_enc']

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v

        elow.load_state_dict(new_state_dict)
        elow.to(device)

        if finetune_coarse_model:
            print('finetune coarse model as well')
            elow.train()
        else:
            elow.eval()

        params_number = sum(param.numel() for param in elow.parameters())
        print(f'load ckpt {ckpt_path_elow} in coarse model with {params_number}')

        return elow     

        
    def _setup_dirs_and_writer(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        for mode in ['train']:
            os.makedirs(os.path.join(savedir, f'pointclouds_{mode}'), exist_ok=True)

        self.savedir = savedir
        self.writer = None
        self.step = 0
        self.epoch = 0
       
    
    def single_step(self, pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):
        """
        Compute losses using a randomly sampled view per iteration.
        
        At each optimization step:
        1. Randomly sample ONE view from the available views
        2. Compute loss using only that sampled view
        3. 3D strands are always generated from view_000 (frontal) in update_hairstyle()
        """
        # Unpack batch
        img, baldness_mask, feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal, gaus_cam_frontal = batch
        
        # Move inputs to the appropriate device
        device_inputs = [img, baldness_mask, cam, flip, transformer_mask]
        device_inputs = [x.to(self.device).to(rank) for x in device_inputs]
        img, baldness_mask, cam, flip, transformer_mask = device_inputs
        feats = feats.to(self.device).to(rank)

        # Initialize all losses
        def zero_loss():
            return torch.tensor([0.0], device=img.device)

        losses = {
            'sdf': zero_loss(),
            'gaus_l1': zero_loss(),
            'gaus_ssim': zero_loss(),
            'gaus_mask': zero_loss(),
            'gaus_orient': zero_loss(),
            'gaus_depth': zero_loss(),
        }
        
        # Gaussian feature loss with single randomly sampled view
        tb_writer = None

        # Extract hair geometry info for logging
        n_strands = pred_points_vis.shape[1]
        n_pts = pred_points_vis.shape[2]
        total_points = n_strands * n_pts
        
        # Get total number of available views
        # feats shape: [batch, n_views, C, H, W]
        # cam shape: [batch, n_views, 3, 4]
        n_views = feats.shape[1]
        
        # Randomly sample ONE view for this iteration
        # View 0 (frontal) has 3x higher chance of being selected than other views
        # Weights: view_0 = 3, view_1 = 1, view_2 = 1, ...
        view_weights = [2.0] + [1.0] * (n_views - 1)
        sampled_view_idx = random.choices(range(n_views), weights=view_weights, k=1)[0]
        self._sampled_view_idx = sampled_view_idx  # Store for logging
        
        # Extract the sampled view
        # Keep batch dimension but select only one view
        feats_sampled = feats[:, sampled_view_idx:sampled_view_idx+1, ...]  # [batch, 1, C, H, W]
        cam_sampled = cam[:, sampled_view_idx:sampled_view_idx+1, ...]      # [batch, 1, 3, 4]
        
        # Pass the sampled view to GaussianTrainerMV
        l1, ssim, mask, orient, depth, view_loss = self.gaus_trainer.step(
            pred_points_vis, cam_sampled, feats_sampled,
            scaling_factor=self.real_set_train.scale_camera_factor,
            iteration=self.step, tb_writer=tb_writer,
            mode=mode, flip=flip, appearance=self.appearance, 
            cam_idxes=cam_idxes
        )
        
        losses.update({
            'gaus_l1': l1,
            'gaus_ssim': ssim,
            'gaus_mask': mask,
            'gaus_orient': orient,
            'gaus_depth': depth
        })
        
        # Store sampled view loss for logging (now just one value)
        self._sampled_view_loss = view_loss[0] if isinstance(view_loss, list) else view_loss
            
        # SDF loss
        if self.sdf_penalty > 0:
            interested_bs, interested_idxes = torch.where(baldness_mask.reshape(img.shape[0], -1) != 0)
            dists = self.sdf.forward(pred_points_vis[interested_bs, interested_idxes].reshape(-1, 3), scale_pts=True).reshape(-1)
            losses['sdf'] = torch.relu(-dists).abs().mean()

        # Apply view weight: frontal view (view_000) has weight 1.0, other views have weight 0.5
        view_weight = 1.0 if sampled_view_idx == 0 else 0.5
        self._view_weight = view_weight  # Store for logging

        loss = view_weight * (
            self.sdf_penalty * losses['sdf'] +
            self.gaus_l1 * losses['gaus_l1'] +
            self.gaus_ssim * losses['gaus_ssim'] +
            self.gaus_mask * losses['gaus_mask'] +
            self.gaus_orient * losses['gaus_orient'] +
            self.gaus_depth * losses['gaus_depth'] 
        )

        # Save point clouds
        if self.step % self.pc_freq == 0 and rank == 0:
            save_path = os.path.join(self.savedir, f"pointclouds_{mode}", f'pred_{self.step:06d}.ply')
            # Detach and check for NaN before saving
            strands_to_save = pred_points_vis[0].detach().clone()
            if torch.isnan(strands_to_save).any():
                print(f'\n[WARNING] NaN detected in strands at step {self.step}, skipping save')
            else:
                save_strands(strands_to_save, save_path, num_points=self.num_points, cols=self.colors_save)

        # Get view name for logging
        view_name = self.real_set_train.view_list[sampled_view_idx] if hasattr(self.real_set_train, 'view_list') else f'v{sampled_view_idx}'
        sampled_loss = self._sampled_view_loss.cpu().numpy() if hasattr(self._sampled_view_loss, 'cpu') else self._sampled_view_loss

        # Log losses with detailed info
        logs = {
            f'full_loss_{mode}': loss.detach().cpu().numpy(),
            f'sampled_view_idx_{mode}': sampled_view_idx,
            f'sampled_view_name_{mode}': view_name,
            f'sampled_view_loss_{mode}': sampled_loss,
            f'view_weight_{mode}': view_weight,
            f'n_views_total_{mode}': n_views,
            f'n_strands_{mode}': n_strands,
            f'n_points_per_strand_{mode}': n_pts,
            f'total_points_{mode}': total_points,
        }
        for k, v in losses.items():
            logs[f'loss_{k}_{mode}'] = v.detach().cpu().numpy()
        
        return loss, logs
   

    def update_hairstyle(self, batch, world_size, rank, device, global_rank):
        """
        Generate hair strands from the frontal view image.
        
        The frontal view (image) is used for encoder prediction.
        This is the same as single-view - only the encoder input matters here.
        """
        n_unfreeze_comp = max(5, min(self.step // self.unfreeze_time_for_pca, self.num_components)) if self.unfreeze_time_for_pca > -1 else self.num_components
    
        if self.step <= self.num_steps_coarse and self.unfreeze_time_for_pca == -1:
            n_unfreeze_comp = 10
 
        img, baldness_mask, feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal, gaus_cam_frontal = batch

        # Move to device
        img, baldness_mask, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal, gaus_cam_frontal = (
            img.to(self.device), baldness_mask.to(self.device), cam.to(self.device), 
            flip.to(self.device), transformer_mask.to(self.device), cam_idxes.to(self.device), 
            gaus_feats_frontal.to(self.device), gaus_cam_frontal.to(self.device)
        )
        
        img, baldness_mask, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal, gaus_cam_frontal = (
            img.to(rank), baldness_mask.to(rank), cam.to(rank), flip.to(rank), 
            transformer_mask.to(rank), cam_idxes.to(rank), 
            gaus_feats_frontal.to(rank), gaus_cam_frontal.to(rank)
        )
        
        feats = feats.to(self.device)
        feats = feats.to(rank)

        # The input image is from the FRONTAL VIEW only
        model_input = img

        # Obtain mask for masking attention layers
        transformer_mask_cond = -100 * (1 - (F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size)).reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0).float())

        transformer_mask = F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size), mode='bilinear', align_corners=False)

        if self.dilate_mask:
            transformer_mask = F.max_pool2d(transformer_mask, kernel_size=3, stride=1, padding=1).squeeze(1)

        transformer_mask_cond = -100 * (1 - transformer_mask.reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0)

        # Obtain hairstyle prediction from FRONTAL VIEW
        batched_pred_strand_dirs, batched_pred_scaling_factor = self.lp_enc(model_input, transformer_mask=transformer_mask_cond, elo=self.lp_enc_elow)

        batched_pred_strand_dirs = batched_pred_strand_dirs[:, :n_unfreeze_comp]
        
        if self.scale_output:
            batched_pred_strand_dirs = batched_pred_strand_dirs * self.scale_stats_std[:n_unfreeze_comp].reshape(1, -1, 1, 1) + self.scale_stats_mean[:n_unfreeze_comp].reshape(1, -1, 1, 1)            
            
        if batched_pred_scaling_factor.shape[1] > 1:
            batched_pred_scaling_factor, batched_pred_baldness_mask = torch.split(batched_pred_scaling_factor, 1, dim=1)
        
        if self.use_scale is False:
            batched_pred_scaling_factor = torch.ones_like(batched_pred_scaling_factor)
            
        bs = img.shape[0]
                
        if self.edited_uvmap is None:
            gt_hair_mask_for_scalp = (((gaus_feats_frontal[0][0][3] > 0) * 255).detach().cpu().numpy()).astype(np.uint8)
            kernel = np.ones((10, 10), np.uint8)
            dilated_mask = cv2.dilate(gt_hair_mask_for_scalp, kernel, iterations=1)
            gt_hair_mask_dilated = ((1 - torch.tensor(dilated_mask, device=device) / 255.) > 0).bool()

            self.scalp_render_map = self.scalp_render(gaus_cam_frontal[0][0], gt_hair_mask_dilated)[None][None]

        if self.scalp_mask_prediction is None:
            self.scalp_mask_prediction = batched_pred_baldness_mask.detach()

        baldness_mask *= self.scalp_mask_prediction

        pred_strands_dirs = batched_pred_strand_dirs.permute(0, 2, 3, 1).reshape(-1, n_unfreeze_comp)
        pred_scaling_factor = batched_pred_scaling_factor.permute(0, 2, 3, 1).reshape(-1, 1)
        
        pred_pc = decode_pca(pred_strands_dirs, self.mean_shape, self.blend_shapes, n_components=n_unfreeze_comp, num_points=self.num_points) * pred_scaling_factor.view(-1, 1, 1)

        roots = self.roots_origins.repeat(bs, 1, 1, 1).reshape(-1, 1, 3)
        
        strands_number = self.texture_size ** 2
        
        pred_points_vis = torch.cat((roots, pred_pc + roots), 1).reshape(bs, strands_number, -1, 3)

        if self.upsample_hairstyle:
            bs, hw, pts, ch = pred_points_vis.shape
            strand_texture = pred_points_vis.permute(0, 2, 3, 1).reshape(bs, pts*ch, self.texture_size, self.texture_size)
           
            pred_points_vis_local = pred_points_vis - pred_points_vis[:, :, :1]
            strand_texture = pred_points_vis_local[:, :, 1:].permute(0, 2, 3, 1).reshape(bs, -1, self.texture_size, self.texture_size)
            
            bil = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0]
            near = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='nearest')[0] 

            nonzerox, nonzeroy = torch.where(baldness_mask[0][0] != 0)
            
            patch_world_displ = torch.zeros(self.texture_size, self.texture_size, self.num_points-1, 3, device=self.device)
            patch_world_displ[[nonzerox, nonzeroy]] = pred_points_vis_local.reshape(self.texture_size, self.texture_size, self.num_points, 3)[nonzerox, nonzeroy][:, 1:] - pred_points_vis_local.reshape(self.texture_size, self.texture_size, self.num_points, 3)[nonzerox, nonzeroy][:, :-1]
            strands_sim = calc_strands_similarity(patch_world_displ)
            strands_sim_hr = F.interpolate(strands_sim[None][None], size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0][0]

            latents_interp = self.blend_func(strands_sim_hr)[None] * near + (1 - self.blend_func(strands_sim_hr)[None]) * bil
            
            pres = latents_interp.reshape(-1, 3, self.resolution_upsample, self.resolution_upsample).permute(2, 3, 0, 1).reshape(1, self.resolution_upsample * self.resolution_upsample, -1, 3)
            
            upsampled_texture = torch.cat((self.roots_origins_up.reshape(1, -1, 1, 3), self.roots_origins_up.reshape(1, -1, 1, 3)+pres), -2)

            edit_uvmap = ((1 - torch.nn.functional.interpolate(self.scalp_render_map, (self.resolution_upsample, self.resolution_upsample), mode='bilinear')) > 0)[0][0]           
            upsampled_baldness_mask = F.interpolate(baldness_mask, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear', align_corners=False)
                                                    
        nonzero_idxs = torch.where(baldness_mask.reshape(-1) > 0)[0]
        self.appearance = None

        if self.upsample_hairstyle:
            interested_idxes_up = torch.where(upsampled_baldness_mask[0].reshape(-1) >= 0.99)[0]

            if self.optimize_appearance:
                param_size = (self.resolution_upsample*self.resolution_upsample, self.num_points-1, 48)
                self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[interested_idxes_up].detach().contiguous().clone(), requires_grad=True)
       
        else:
            param_size = (self.resolution_upsample * self.resolution_upsample, self.num_points-1, 48)
            self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[nonzero_idxs].detach().contiguous().clone(), requires_grad=True)

        if self.upsample_hairstyle:
            selected_strands = upsampled_texture[0][interested_idxes_up]
        else:
            selected_strands = pred_points_vis[0][nonzero_idxs]
        
        return selected_strands
    

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

          
    @torch.no_grad()
    def load_model(self, ckpt_path, rank):
        print('Loading model on GPU')

        map_location = {f'cuda:0': f'cuda:{rank}'} if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        print(f'Loaded checkpoint: {ckpt_path}')

        from collections import OrderedDict

        def _strip_module(sd):
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_sd[k.replace('module.', '')] = v
            return new_sd

        # Handle lp_enc (strip potential DDP prefixes)
        lp_enc_sd = checkpoint.get('lp_enc', {})
        lp_enc_sd = _strip_module(lp_enc_sd)
        self.lp_enc.load_state_dict(lp_enc_sd, strict=False)

        # Handle lp_enc_elow if present
        try:
            lp_enc_elow_sd = checkpoint.get('lp_enc_elow', {})
            lp_enc_elow_sd = _strip_module(lp_enc_elow_sd)
            self.lp_enc_elow.load_state_dict(lp_enc_elow_sd, strict=False)
            print('Loaded lp_enc_elow successfully')
        except Exception as e:
            print(f'Failed to load lp_enc_elow: {e}')

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded optimizer state')
        except Exception as e:
            print(f'Failed to load optimizer state: {e}')

        
    @torch.no_grad()
    def save_model(self):
        print('Saving model on GPU')

        checkpoint = {
            'lp_enc': self.lp_enc.state_dict(),
            'lp_enc_elow': self.lp_enc_elow.state_dict(),
            'step': self.step,
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        checkpoint_dir = os.path.join(self.savedir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{self.step:06d}.pth')
        torch.save(checkpoint, checkpoint_path)
    

    def train(self, world_size, rank, device, global_rank):
        # Print training configuration summary at start
        if rank == 0:
            print('\n' + '='*60)
            print('STARTING MULTI-VIEW OPTIMIZATION')
            print('='*60)
            print(f'  Subject:         {self.subject}')
            print(f'  Num views:       {self.n_views}')
            print(f'  Num strands:     {self.texture_size}x{self.texture_size} = {self.texture_size**2}')
            print(f'  Points/strand:   {self.num_points}')
            print(f'  Total points:    {self.texture_size**2 * self.num_points}')
            print(f'  Num epochs:      {self.num_epochs}')
            print(f'  Learning rate:   {self.learning_rate}')
            print('-'*60)
            print('Loss weights:')
            print(f'  L_orient: {self.gaus_orient}  |  L_mask: {self.gaus_mask}  |  L_depth: {self.gaus_depth}  |  L_sdf: {self.sdf_penalty}')
            print('='*60 + '\n')
        
        # Calculate total steps for progress bar
        steps_per_epoch = len(self.train_dl)
        total_steps = self.num_epochs * steps_per_epoch
        
        import time
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                for batch in self.train_dl:
                    loss, logs = self.training_step(batch, self.step, world_size, rank, device, global_rank)
                    if self.accumulate_gradients > 1:
                        loss /= self.accumulate_gradients
                        
                    loss.backward()  
                    
                    if (self.step + 1) % self.accumulate_gradients == 0:
                        # Gradient clipping to prevent NaN
                        torch.nn.utils.clip_grad_norm_(self.lp_enc.parameters(), max_norm=1.0)
                        if self.finetune_coarse_model:
                            torch.nn.utils.clip_grad_norm_(self.lp_enc_elow.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
          
                    # Manual inline progress display
                    if rank == 0:
                        total_loss = logs['full_loss_train']
                        sampled_idx = logs['sampled_view_idx_train']
                        sampled_loss = logs['sampled_view_loss_train']
                        view_weight = logs['view_weight_train']
                        
                        # Calculate progress
                        progress = (self.step + 1) / total_steps
                        elapsed = time.time() - start_time
                        eta = elapsed / (self.step + 1) * (total_steps - self.step - 1) if self.step > 0 else 0
                        
                        # Build progress bar string
                        bar_width = 30
                        filled = int(bar_width * progress)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        
                        # Format time
                        def fmt_time(s):
                            m, s = divmod(int(s), 60)
                            h, m = divmod(m, 60)
                            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                        
                        # Print inline progress (overwrite same line)
                        progress_str = f"\rMV-Opt ({self.n_views} views): {bar} {self.step+1}/{total_steps} [{fmt_time(elapsed)}<{fmt_time(eta)}] loss={total_loss:.4f} (view_{sampled_idx}={sampled_loss:.4f})"
                        print(progress_str, end='', flush=True)
                    
                    if self.step % self.save_freq == 0 and rank == 0:
                        print()  # New line before save message
                        print(f'[Step {self.step}] Saving checkpoint...')
                        self.save_model()
                        
                    self.step += 1

                self.epoch = epoch + 1

        except KeyboardInterrupt:
            if rank == 0:
                print('\nTraining interrupted by user.')
        
        finally:
            if rank == 0:
                # Save final point cloud
                print()  # New line
                print(f'[Step {self.step}] Saving final point cloud...')
                save_path = os.path.join(self.savedir, f"pointclouds_train", f'pred_{self.step:06d}.ply')
                # Need to get the last pred_points_vis - run one more forward pass
                try:
                    with torch.no_grad():  # No gradients needed for final save
                        for batch in self.train_dl:
                            pred_points_vis = self.update_hairstyle(batch, world_size, rank, device, global_rank)
                            # Check for NaN before saving
                            if torch.isnan(pred_points_vis).any():
                                print(f'  [WARNING] NaN detected in final strands, skipping save')
                            else:
                                save_strands(pred_points_vis.detach(), save_path, num_points=self.num_points, cols=self.colors_save)
                                print(f'  Saved: {save_path}')
                            break  # Only need one batch
                except Exception as e:
                    print(f'  Failed to save final point cloud: {e}')
                
                print('\n' + '='*60)
                print(f'OPTIMIZATION COMPLETE - Final step: {self.step}')
                print('='*60 + '\n')
