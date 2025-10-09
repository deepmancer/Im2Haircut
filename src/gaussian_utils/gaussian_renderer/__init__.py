#
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
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from src.gaussian_utils.scene.gaussian_model import GaussianModel
from src.gaussian_utils.scene.gaussian_model_strands import GaussianModelCurves

import sys
sys.path.append("./submodules/external/GaussianHaircut")
from src.utils.sh_utils import eval_sh
from src.utils.general_utils import build_rotation



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, render_direction = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    PRECOMP_CONIC = True
    conic_precomp = pc.get_conic(viewpoint_camera, scaling_modifier) if PRECOMP_CONIC else None
    screenspace_points = pc.get_mean_2d(viewpoint_camera)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=torch.tan(viewpoint_camera.FoVx * 0.5).item(),
        tanfovy=torch.tan(viewpoint_camera.FoVy * 0.5).item(),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D_precomp = screenspace_points
    opacity = pc.get_opacity

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    rgb_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    if PRECOMP_CONIC:
        cov3D_precomp = pc.cov
        cov2D = pc.get_direction_2d(viewpoint_camera) if render_direction else pc.cov2d
    else:
        if render_direction:
            cov3D_precomp = pc.get_covariance()
            cov2D = pc.get_direction_2d(viewpoint_camera)
        else:
            cov2D = pc.get_covariance_2d(viewpoint_camera)
            cov3D_precomp = pc.cov

    colors_precomp = torch.cat(
        [
            rgb_precomp, 
            pc.get_label, 
            torch.ones_like(pc.get_label), # foreground mask
            cov2D, 
            pc.get_orient_conf, 
            pc.get_depths(viewpoint_camera)
        ], 
        dim=-1
    )

    points_mask = pc.filter_points(viewpoint_camera)

    means3D = means3D[points_mask]
    means2D_precomp = means2D_precomp[points_mask]
    colors_precomp = colors_precomp[points_mask]
    opacity = opacity[points_mask]
    cov3D_precomp = cov3D_precomp[points_mask]
    conic_precomp = conic_precomp[points_mask] if PRECOMP_CONIC else None

    radii = torch.zeros_like(pc.get_xyz[:, 0]).int()

    renders, _radii = rasterizer(
        means3D = means3D,
        means2D = means2D_precomp,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = None,
        rotations = None,
        cov3D_precomp = cov3D_precomp,
        conic_precomp = conic_precomp)

    radii[points_mask] = _radii

    rendered_image, rendered_mask, rendered_cov2D, rendered_orient_conf, rendered_depth = renders.split([3, 2, 3, 1, 1], dim=0)

    if render_direction:
        rendered_dir2D = F.normalize(rendered_cov2D[:2], dim=0)
        to_mirror = torch.ones_like(rendered_dir2D[[0]])
        to_mirror[rendered_dir2D[[0]] < 0] *= -1
        rendered_orient_angle = torch.acos(rendered_dir2D[[1]].clamp(-1 + 1e-3, 1 - 1e-3) * to_mirror) / math.pi
    else:
        mid = 0.5 * (rendered_cov2D[[0]] + rendered_cov2D[[2]])
        det = rendered_cov2D[[0]] * rendered_cov2D[[2]] - rendered_cov2D[[1]]**2
        sqrtD = (torch.clamp(mid**2 - det, min=0.1))**0.5
        lambda1 = mid + sqrtD
        lambda2 = mid - sqrtD
        rendered_orient_conf = rendered_orient_conf * (1 - lambda2 / lambda1)
        rendered_orient_angle = 1 - (torch.atan2(rendered_cov2D[[1]], rendered_cov2D[[0]]) / math.pi + 0.5)

    return {"render": rendered_image,
            "mask": rendered_mask,
            "orient_angle": rendered_orient_angle,
            "orient_conf": rendered_orient_conf,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    
    
    

def render_hair(viewpoint_camera, pc : GaussianModel, pc_hair: GaussianModelCurves, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, render_direction = False, use_directed_loss=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    conic_precomp = torch.cat([
        pc.get_conic(viewpoint_camera, scaling_modifier)[pc.mask_precomp], 
        pc_hair.get_conic(viewpoint_camera, scaling_modifier)]
    )
    screenspace_points = torch.cat([pc.get_mean_2d(viewpoint_camera)[pc.mask_precomp].detach(), pc_hair.get_mean_2d(viewpoint_camera)], dim=0)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc_hair.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    means3D = torch.cat([pc.xyz_precomp, pc_hair.get_xyz])
    means2D_precomp = screenspace_points
    opacity = torch.cat([pc.opacity_precomp, pc_hair.get_opacity])

    points_mask = torch.cat([
        pc.filter_points(viewpoint_camera)[pc.mask_precomp], 
        pc_hair.filter_points(viewpoint_camera)]
    )
    scales = torch.cat([pc.scaling_precomp, pc_hair.get_scaling])
    rotations = torch.cat([pc.rotation_precomp, pc_hair.get_rotation])

    shs_view = torch.cat([pc.shs_view, pc_hair.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)])
    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc_hair.active_sh_degree, shs_view, dir_pp_normalized)
    rgb_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    label_precomp = torch.cat([torch.zeros_like(pc.xyz_precomp[:, :1]), pc_hair.get_label])
    if render_direction:

        cov2D = torch.cat([
            torch.zeros_like(pc.xyz_precomp), 
            pc_hair.get_direction_2d(viewpoint_camera)])
    else:

        cov2D = torch.cat([
            torch.zeros_like(pc.xyz_precomp), 
            pc_hair.cov]) if hasattr(pc_hair, 'cov') else torch.cat([
                torch.zeros_like(pc.xyz_precomp), 
                pc_hair.get_covariance_2d(viewpoint_camera, scaling_modifier)])
    orient_conf = torch.cat([torch.zeros_like(pc.xyz_precomp[:, :1]), pc_hair.get_orient_conf])
    depth = torch.cat([pc.get_depths(viewpoint_camera)[pc.mask_precomp], pc_hair.get_depths(viewpoint_camera)])
    colors_precomp = torch.cat([rgb_precomp, label_precomp, torch.ones_like(label_precomp), cov2D, orient_conf, depth], dim=-1)

    radii = torch.zeros_like(means3D[:, 0]).int()

    means3D = means3D[points_mask]
    means2D_precomp = means2D_precomp[points_mask]
    colors_precomp = colors_precomp[points_mask]
    opacity = opacity[points_mask]
    scales = scales[points_mask]
    rotations = rotations[points_mask]
    conic_precomp = conic_precomp[points_mask]

    renders, _radii = rasterizer(
        means3D = means3D,
        means2D = means2D_precomp,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        conic_precomp = conic_precomp)

    radii[points_mask] = _radii
    
    rendered_image, rendered_mask, rendered_cov2D, rendered_orient_conf, rendered_depth = renders.split([3, 2, 3, 1, 1], dim=0)
    
    if render_direction:

        if use_directed_loss:
            
            epsilon = 1e-6

            # Compute the magnitude of the vector
            magnitude = rendered_cov2D[:2].pow(2).sum(dim=0).sqrt()
            # Create a mask for zero vectors
            non_zero_mask = magnitude > epsilon

            # Default direction for zero vectors
            default_dir = torch.tensor([1.0, 0.0], device=rendered_cov2D.device).view(2, 1, 1)

            # Normalize non-zero vectors and replace zero vectors with the default direction
            rendered_dir2D = torch.where(
                non_zero_mask,
                F.normalize(rendered_cov2D[:2], dim=0),
                default_dir.expand_as(rendered_cov2D[:2])
            )

            angle_radians = torch.atan2(rendered_dir2D[1], rendered_dir2D[0])
            rendered_orient_angle = (angle_radians + math.pi) / (2 * math.pi)
        else:
            rendered_dir2D = F.normalize(rendered_cov2D[:2], dim=0)
            to_mirror = torch.ones_like(rendered_dir2D[[0]])
            to_mirror[rendered_dir2D[[0]] < 0] *= -1
            rendered_orient_angle = torch.acos(rendered_dir2D[[1]].clamp(-1 + 1e-3, 1 - 1e-3) * to_mirror) / math.pi
        
    else:
        rendered_orient_angle = 1 - (torch.atan2(rendered_cov2D[[1]], rendered_cov2D[[0]]) / math.pi + 0.5)

    return {"render": rendered_image,
            "mask": rendered_mask,
            "orient_angle": rendered_orient_angle,
            "orient_conf": rendered_orient_conf,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}