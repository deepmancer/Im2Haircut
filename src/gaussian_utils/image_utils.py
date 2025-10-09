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
import math

def vis_directed_orient(angle_map, r_channel=None):
    """
    Generates an RGB strand map from angle and magnitude.
    
    Args:
        angle_map: (H, W) array of orientation angles (in radians).
        magnitude: (H, W) array of magnitudes (e.g., strength or intensity).
    
    Returns:
        RGB image: (H, W, 3) array.
    """
    angle_map = angle_map * 2 * torch.pi + torch.pi/2
    # Compute 2D orientation unit vectors
    cos_theta = torch.cos(angle_map) # x-component
    sin_theta = torch.sin(angle_map)  # y-component
    
    # Normalize orientation vectors to [0, 1]
    g_channel = (cos_theta / 2) + 0.5  # Green (x-component)
    b_channel = (sin_theta / 2) + 0.5  # Blue (y-component)

    # Combine into an RGB image
    if r_channel is not None:
        rgb_image = torch.cat([r_channel, g_channel, b_channel], dim=1)
        return rgb_image
    else:
        return torch.cat([g_channel, b_channel], dim=1)


def vis_depth(depth):
    depth_vis = (depth + 1).log()
    depth_vis = (depth_vis - depth_vis.amin()) / (depth_vis.amax() / depth_vis.amin())
    return depth_vis