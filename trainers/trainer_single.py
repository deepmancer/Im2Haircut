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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import yaml
from pyhocon import ConfigFactory

# === Project-Specific Modules ===
# Append project paths
import sys
sys.path.append("./submodules/external/GaussianHaircut")
from src.utils.general_utils import safe_state


from src.model_utils.geometry import decode_pca
from src.datasets.real_dataset import  HairstyleRealDataset


# Losses and model utils
from src.loss_utils.head_sdf_prior import SDFHeadPrior
from src.loss_utils.scalp_renderer import ScalpRenderer

from src.gaussian_utils.GaussianTrainer import GaussianTrainer

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

             

class BaseTrainer(nn.Module): 
          
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
        self.scalp_render = ScalpRenderer(size=(512, 512)) #todo fix it
        self.accumulate_gradients = accumulate_gradients
        self.unfreeze_time_for_pca = unfreeze_time_for_pca

        self.all_steps = config['visuals_config'].get('num_epochs', 1)

        self.path_to_coords_for_each_origin = f'./data/coords_for_each_origin_64x64.pth'
#         pca map
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
            
#         for penetration loss
        if self.sdf_penalty > 0:
            ckpt = './pretrained_models/neus.pth'
            self.sdf = SDFHeadPrior(ckpt, apply_relu=False, device=self.device)
                 
        self.colors_save = None
        self.edited_uvmap = None

        
        
    def _init_dataset(self, config, scene, folder_name, world_size, rank, num_workers):
        data_path = config['dataset'].get('data_path', '')
        print('Dataset path:', data_path, '| config:', config['dataset'], '| scene:', scene)
        
        self.data_path = data_path
        self.scene = scene
        
        self.real_set_train = HairstyleRealDataset(
            **config['dataset_real'],
            infer_path=f'{data_path}/{folder_name}',
            scene=scene
        )

        self.num_workers = num_workers

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.real_set_train, num_replicas=world_size, rank=rank
        )
        self.train_dl = data.DataLoader(
            self.real_set_train,
            config['optconfig']['batch_size'],
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        
    def _init_roots_and_blend_shapes(self):
        
        self.roots_origins = torch.load(self.path_to_coords_for_each_origin)[None].float().to(self.device)
        global_mean_shape = torch.tensor(np.load(self.global_mean_path), device=self.device).float()
        mean_shape_local = torch.tensor(np.load( self.mean_shape_path), device=self.device).float()
        self.mean_shape =  global_mean_shape + mean_shape_local
        self.blend_shapes = torch.tensor(np.load(self.blend_shape_path), device=self.device).float()
        

    def _init_encoders(self, config, device, rank):

        self.projector_type = config['projector'].get('projector_type', '')
        self.lp_enc = create_projector_backbone(self.projector_type, config)
        self.lp_enc = nn.parallel.DistributedDataParallel(self.lp_enc.to(device), device_ids=[rank], find_unused_parameters=True)

        projector_type_elow = config['projector_type_elow'].get('projector_type', '')
        ckpt_path_elow = config['lp_encoder_fine']['ckpt_path_elow']
        lp_enc_elow = self.create_coarse_model(projector_type_elow, config, ckpt_path_elow, device, finetune_coarse_model=self.finetune_coarse_model)

        if self.finetune_coarse_model:
            self.lp_enc_elow = nn.parallel.DistributedDataParallel(lp_enc_elow.to(device), device_ids=[rank], find_unused_parameters=True)
        else:
            self.lp_enc_elow = lp_enc_elow

            
    def _init_gaussian_trainer(self, dataset, opt,  pipe, pointcloud_path_head, ip, port, rank, config):
        self.gaus_trainer = GaussianTrainer(
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
                new_key = k.replace('module.', '')  # Remove `module.` prefix
                new_state_dict[new_key] = v


            elow.load_state_dict(new_state_dict)
            elow.to(device)

            if finetune_coarse_model:
                print('finetune coarse model as well')
                elow.train()
            else:
                elow.eval()

            params_number =  sum(param.numel() for param in elow.parameters())
            print(f'load ckpt {ckpt_path_elow} in coarse model with {params_number}')

            return elow     

        
    def _setup_dirs_and_writer(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        for mode in ['train']:
#             os.makedirs(os.path.join(savedir, f'images_{mode}'), exist_ok=True)
            os.makedirs(os.path.join(savedir, f'pointclouds_{mode}'), exist_ok=True)
#         os.makedirs(os.path.join(savedir, 'checkpoints'), exist_ok=True)

        self.savedir = savedir
        self.writer = SummaryWriter(log_dir=os.path.join(savedir, 'logs'))
        self.step = 0
        self.epoch = 0
       
    
    def single_step(self, pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):
                         
        # Unpack batch
        img, baldness_mask, feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal = batch
        
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
        
        
        # Gaussian feature loss

        tb_writer = self.writer if self.step % self.pc_freq == 0 and rank == 0 else None

        n_strands = pred_points_vis.shape[1]
        n_pts = pred_points_vis.shape[2]

        l1, ssim, mask, orient, depth = self.gaus_trainer.step(
            pred_points_vis, cam, feats,
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
            
        # SDF loss
        if self.sdf_penalty > 0:
            interested_bs, interested_idxes = torch.where(baldness_mask.reshape(img.shape[0], -1) !=  0)
            dists = self.sdf.forward(pred_points_vis[interested_bs, interested_idxes].reshape(-1, 3), scale_pts=True).reshape(-1)
            losses['sdf'] = torch.relu(-dists).abs().mean()

        
        loss = (
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
            save_strands(pred_points_vis[0], save_path, num_points=self.num_points, cols=self.colors_save)

        # Log losses
        logs = {f'bs_{mode}': 1, f'full_loss_{mode}': loss.detach().cpu().numpy()}
        for k, v in losses.items():
            logs[f'loss_{k}_{mode}'] = v.detach().cpu().numpy()
        
        return loss, logs
   

    def update_hairstyle(self, batch,  world_size, rank, device, global_rank):
       
        n_unfreeze_comp = max(5, min(self.step // self.unfreeze_time_for_pca, self.num_components)) if self.unfreeze_time_for_pca > -1 else self.num_components
    
        if self.step <= self.num_steps_coarse and self.unfreeze_time_for_pca==-1:
            n_unfreeze_comp = 10
 
        img, baldness_mask,  feats, cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal = batch

        
        img, baldness_mask,  cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal = img.to(self.device), baldness_mask.to(self.device), cam.to(self.device), flip.to(self.device), transformer_mask.to(self.device), cam_idxes.to(self.device), gaus_feats_frontal.to(self.device),  gaus_cam_frontal.to(self.device)
        
        img, baldness_mask,  cam, flip, transformer_mask, cam_idxes, gaus_feats_frontal,  gaus_cam_frontal = img.to(rank), baldness_mask.to(rank),  cam.to(rank), flip.to(rank), transformer_mask.to(rank), cam_idxes.to(rank), gaus_feats_frontal.to(rank), gaus_cam_frontal.to(rank)
        
        feats = feats.to(self.device)
        feats = feats.to(rank)

        model_input = img

#         obtain mask for masking attention layers to interested region
        transformer_mask_cond = -100 * (1 - (F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size)).reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0).float())
    

        transformer_mask =  F.interpolate(transformer_mask.unsqueeze(1), (self.transformer_mask_size, self.transformer_mask_size), mode='bilinear', align_corners=False)

        if self.dilate_mask:

            transformer_mask = F.max_pool2d(transformer_mask, kernel_size=3, stride=1, padding=1).squeeze(1)

        transformer_mask_cond = -100 * (1 - transformer_mask.reshape(model_input.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0)

#         obtain hairstyle
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
            kernel = np.ones((10, 10), np.uint8)  # Size of kernel (5x5 in this case)
            dilated_mask = cv2.dilate(gt_hair_mask_for_scalp, kernel, iterations=1)
            gt_hair_mask_dilated = ((1 - torch.tensor(dilated_mask, device=device) / 255.) > 0).bool()

            self.scalp_render_map = self.scalp_render(gaus_cam_frontal[0][0],  gt_hair_mask_dilated)[None][None]

            
        if self.scalp_mask_prediction is None:
            self.scalp_mask_prediction = batched_pred_baldness_mask.detach()

        baldness_mask *= self.scalp_mask_prediction
                

        pred_strands_dirs = batched_pred_strand_dirs.permute(0, 2, 3, 1).reshape(-1, n_unfreeze_comp)
        pred_scaling_factor = batched_pred_scaling_factor.permute(0, 2, 3, 1).reshape(-1, 1)
        
            
        pred_pc = decode_pca(pred_strands_dirs, self.mean_shape,  self.blend_shapes, n_components=n_unfreeze_comp, num_points=self.num_points) * pred_scaling_factor.view(-1, 1, 1)

        roots = self.roots_origins.repeat(bs, 1, 1, 1).reshape(-1, 1, 3)
        
        strands_number =  self.texture_size ** 2
        
        pred_points_vis = torch.cat((roots, pred_pc  + roots), 1).reshape(bs,  strands_number, -1, 3) #[bs,  HW, num_pts,3] 
        
        if self.upsample_hairstyle:

#             use combination of nearest and bilinear similar to https://github.com/Vanessik/HAAR
            bs, hw, pts, ch = pred_points_vis.shape
            # Permute to bring spatial dimensions to PyTorch's expected order: (batch, channels, height, width)
            strand_texture = pred_points_vis.permute(0, 2, 3, 1).reshape(bs, pts*ch, self.texture_size, self.texture_size)
           
            # Shape: (bs, num_points, 3, h, w)
            # Upsample using haar interpolation
            pred_points_vis_local = pred_points_vis - pred_points_vis[:, :, :1]
            strand_texture = pred_points_vis_local[:, :, 1:].permute(0, 2, 3, 1).reshape(bs, -1, self.texture_size, self.texture_size) #597, 64, 64
            
            bil = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0]
            near = F.interpolate(strand_texture, size=(self.resolution_upsample, self.resolution_upsample), mode='nearest')[0] 

            nonzerox, nonzeroy = torch.where(baldness_mask[0][0] != 0)
            
            patch_world_displ = torch.zeros(self.texture_size, self.texture_size, self.num_points-1, 3, device=self.device)
            patch_world_displ[[nonzerox, nonzeroy]] = pred_points_vis_local.reshape(self.texture_size, self.texture_size, self.num_points, 3)[nonzerox, nonzeroy][:, 1:] - pred_points_vis_local.reshape(self.texture_size, self.texture_size, self.num_points, 3)[nonzerox, nonzeroy][:, :-1]
            strands_sim = calc_strands_similarity(patch_world_displ)
            strands_sim_hr = F.interpolate(strands_sim[None][None],size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear')[0][0]

            latents_interp = self.blend_func(strands_sim_hr)[None] * near + (1 - self.blend_func(strands_sim_hr)[None]) * bil
            
            pres = latents_interp.reshape(-1, 3, self.resolution_upsample, self.resolution_upsample).permute(2, 3, 0, 1).reshape(1, self.resolution_upsample * self.resolution_upsample, -1, 3)
            
            upsampled_texture = torch.cat((self.roots_origins_up.reshape(1, -1, 1, 3), self.roots_origins_up.reshape(1, -1, 1, 3)+pres), -2)

            edit_uvmap = ((1 - torch.nn.functional.interpolate(self.scalp_render_map, (self.resolution_upsample, self.resolution_upsample),  mode='bilinear')) > 0)[0][0]           
            upsampled_baldness_mask = F.interpolate(baldness_mask, size=(self.resolution_upsample, self.resolution_upsample), mode='bilinear', align_corners=False)# * edit_uvmap[None][None] 
                                                    
        nonzero_idxs = torch.where(baldness_mask.reshape(-1) > 0)[0]
        self.appearance = None

        if self.upsample_hairstyle:
            interested_idxes_up = torch.where(upsampled_baldness_mask[0].reshape(-1) >= 0.99)[0]

            if self.optimize_appearance:
                param_size = (self.resolution_upsample*self.resolution_upsample, self.num_points-1, 48)
                self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[interested_idxes_up].detach().contiguous().clone(), requires_grad=True)
       
        else:
            param_size = (self.resolution_upsample * self.resolution_upsample, self.num_points-1, 48)
            self.appearance = nn.Parameter(torch.ones(param_size, device=self.device)[interested_idxes].detach().contiguous().clone(), requires_grad=True)

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
        local_rank = dist.get_local_rank()
        print(f'Loading model on GPU {local_rank}')

        # Map the checkpoint to the current device
        map_location = {f'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        print(f'Loaded checkpoint: {ckpt_path}')

        # Load encoder weights
        self.lp_enc.load_state_dict(checkpoint['lp_enc'])

        # Attempt to load optional components
        try:
            self.lp_enc_elow.load_state_dict(checkpoint['lp_enc_elow'])
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
        local_rank = dist.get_local_rank()
        print(f'Saving model on GPU {local_rank}')

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

        try:

            while True:
                for batch in tqdm(self.train_dl):
                    
                    loss, logs = self.training_step(batch, self.step, world_size,rank, device, global_rank)
                    if self.accumulate_gradients > 1:
                        loss /= self.accumulate_gradients
                        
                    loss.backward()  
                    
                    if (self.step + 1) % self.accumulate_gradients == 0:

                        self.optimizer.step()
                        self.optimizer.zero_grad()
          
                    if self.step % self.logging_freq == 0 and rank == 0:
                        print('start logging')
                        for key in logs:
                            self.writer.add_scalar(f'{key}', logs[key], self.step)
                    
                    if self.step % self.save_freq == 0 and rank == 0:
                        print('start saving')
                        self.save_model()
                        
                    
                    self.step += 1
                self.epoch += 1

                if self.epoch > self.all_steps:
                    break 
                    
        except KeyboardInterrupt:
            pass
