# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import sys
sys.path.append("./submodules/external/GaussianHaircut")
from src.utils.general_utils import inverse_sigmoid
from torch import nn
import torch.nn.functional as F
import os
from src.utils.general_utils import strip_symmetric, get_expon_lr_func, build_scaling_rotation, parallel_transport
import math





class GaussianModelCurves:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, return_full_covariance=False):
            M = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = M.transpose(1, 2) @ M
            if return_full_covariance:
                return actual_covariance
            else:
                symm = strip_symmetric(actual_covariance)
                return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.label_activation = torch.sigmoid
        self.inverse_label_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.orient_conf_activation = torch.exp
        self.orient_conf_inverse_activation = torch.log
        
        
        

    def __init__(self, sh_degree, scale=8e-3, device='cuda'):

        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._orient_conf = torch.empty(0)
        self._label = torch.empty(0)
       
        self.setup_functions()
        
        self.scale = scale
        self.device = device



    def capture(self):
        return (
            self._pts,
            self.active_sh_degree,
        )
    
    def restore(self, model_args):
        (
            self._pts,
            self.active_sh_degree, 
        ) = model_args

        self.pts_origins = self._pts[:, :1]
        self._dirs = self._pts[:, 1:] - self._pts[:, :-1]
        self._orient_conf = torch.ones_like(self._features_dc[:, :1, 0])


    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.ones_like(self.get_xyz[:, :1])

    @property
    def get_label(self):
        return torch.ones_like(self.get_xyz[:, :1])
    
    @property
    def get_orient_conf(self):
        return self.orient_conf_activation(self._orient_conf)
    

    @torch.no_grad()
    def filter_points(self, viewpoint_camera):
      
        mean = self.get_xyz
        viewmatrix = viewpoint_camera.world_view_transform
        p_view = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        
        mask = p_view[:, [2]] > 0.2

        mask = torch.logical_and(mask, self.det != 0)


        mid = 0.5 * (self.cov[:, [0]] + self.cov[:, [2]])
        sqrtD = (torch.clamp(mid**2 - self.det, min=0.1))**0.5
        lambda1 = mid + sqrtD
        lambda2 = mid - sqrtD
        my_radius = torch.ceil(3 * (torch.maximum(lambda1, lambda2))**0.5)

        point_image_x = ((self.p_proj[:, [0]] + 1) * viewpoint_camera.image_width - 1.0) * 0.5
        point_image_y = ((self.p_proj[:, [1]] + 1) * viewpoint_camera.image_height - 1.0) * 0.5


        BLOCK_X = 16
        BLOCK_Y = 16

        grid_x = (viewpoint_camera.image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (viewpoint_camera.image_height + BLOCK_Y - 1) // BLOCK_Y

        
        rect_min_x = torch.clamp(((point_image_x - my_radius) / BLOCK_X).int(), min=0, max=grid_x)
        rect_min_y = torch.clamp(((point_image_y - my_radius) / BLOCK_Y).int(), min=0, max=grid_y)

        rect_max_x = torch.clamp(((point_image_x + my_radius + BLOCK_X - 1) / BLOCK_X).int(), min=0, max=grid_x)
        rect_max_y = torch.clamp(((point_image_y + my_radius + BLOCK_Y - 1) / BLOCK_Y).int(), min=0, max=grid_y)


        
        self.points_mask = torch.logical_and(mask, (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) != 0).squeeze()

        return self.points_mask

    def get_covariance(self, scaling_modifier = 1, return_full_covariance = False):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, return_full_covariance)

    def get_covariance_2d(self, viewpoint_camera, scaling_modifier = 1):
        mean = self.get_xyz

#         print('mean', mean.min(), mean.max())
        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
#         print('tz', tz.min(), tz.max())
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        J = torch.stack(
            [
                torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st column
                torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd column
                torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd column
            ],
            dim=-1 # stack columns into rows
        )

        W = viewmatrix[None, :3, :3]

        T = W @ J

        Vrk = self.get_covariance(scaling_modifier, return_full_covariance=True)
#         print('vrk', Vrk.min(), Vrk.max(),scaling_modifier)

        cov = T.transpose(1, 2) @ Vrk.transpose(1, 2) @ T

        cov[:, 0, 0] += 0.3
        cov[:, 1, 1] += 0.3

        return torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]], dim=-1)

    def get_conic(self, viewpoint_camera, scaling_modifier = 1):
        self.cov = self.get_covariance_2d(viewpoint_camera, scaling_modifier)
        
        

        self.det = self.cov[:, [0]] * self.cov[:, [2]] - self.cov[:, [1]]**2
        det_inv = 1. / (self.det + 0.0000001)
        
#         print('conic', self.cov.max(), self.cov.min(), self.det.shape, self.det.max(), self.det.min(), det_inv.min(), det_inv.max())
        conic = torch.stack([self.cov[:, 2], -self.cov[:, 1], self.cov[:, 0]], dim=-1) * det_inv


        return conic

    def get_mean_2d(self, viewpoint_camera):

        projmatrix = viewpoint_camera.full_proj_transform
        p_hom = (self.get_xyz[:, None, :] @ projmatrix[None, :3, :] + projmatrix[None, [3]])[:, 0]
        p_w = 1.0 / (p_hom[:, [3]] + 0.0000001)
        self.p_proj = p_hom[:, :3] * p_w

        return self.p_proj

    def get_depths(self, viewpoint_camera):
        viewmatrix = viewpoint_camera.world_view_transform
        p_view = (self.get_xyz[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        return p_view[:, -1:]

    def get_direction_2d(self, viewpoint_camera):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        J = torch.stack(
            [
                torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st column
                torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd column
                torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd column
            ],
            dim=-1 # stack columns into rows
        )

        W = viewmatrix[None, :3, :3]

        T = W @ J

        dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (dir3D[:, None, :] @ T)[:, 0]

        return dir2D


    def update_gaussians_hair(self, pts, features_dc=None, features_rest=None):

        # define color, predict in the future
        num_strands, num_points = pts.shape[0], pts.shape[1]

        if features_dc is None:
            features_dc =  torch.ones(num_strands, num_points - 1, 3).to(self.device) 
            features_rest  = torch.ones(num_strands, num_points - 1, 45).to(self.device)


            self._features_dc = nn.Parameter(features_dc.reshape(-1, 1, 3).contiguous().clone().requires_grad_(True))
            self._features_rest = nn.Parameter(features_rest.reshape(-1, (self.max_sh_degree + 1) ** 2 - 1, 3).contiguous().clone().requires_grad_(True))
        
        else:
            self._features_dc = features_dc.reshape(-1, 1, 3)
            self._features_rest = features_rest.reshape(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
            
        orient_conf =  torch.ones(num_strands, num_points - 1, 1).to(self.device)
        self._orient_conf = nn.Parameter(orient_conf[:, None, :].reshape(-1, 1).contiguous().clone().requires_grad_(True)) 
    
        # pts from strands
        self._pts = pts
        self._dir = (pts[:, 1:]-pts[:, :-1]).reshape(-1, 3)
        self._xyz = (self._pts[:, 1:] + self._pts[:, :-1]).view(-1, 3) * 0.5
        self._dirs = (pts[:, 1:]-pts[:, :-1]).reshape(-1, 3)
        
        
        
        # Assign geometric features        
        self._rotation = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(self._xyz[:, :1]),
                    torch.zeros_like(self._xyz[:, :2])
                ],
                dim=-1
            ),
            b=self._dir
        ).view(-1, 4) # rotation parameters that align x-axis with the segment direction

        self._scaling = torch.ones_like(self.get_xyz)
        self._scaling[:, 0] = (self._dir.norm(dim=-1) * 0.5).clip(1e-7, 100)
        self._scaling[:, 1:] = self.scale
        
#         print('IMPORTANT SCALING', self._dir.shape, self._scaling.max(), self._scaling.min())

        
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1     