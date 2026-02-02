"""
Multi-view Gaussian Trainer for hair reconstruction.

This module extends GaussianTrainer to support per-view loss computation
and reporting for multi-view 3DGS optimization.

Optimization strategy:
- At each iteration, a SINGLE view is randomly sampled for supervision
- The 3D hair strands are ALWAYS generated from the frontal view (view_000)
- This allows the model to optimize against diverse viewpoints over training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./submodules/external/GaussianHaircut')
from src.utils.loss_utils import l1_loss, or_loss, ssim

from src.loss_utils.losses import or_loss_directed
from src.gaussian_utils.GaussianTrainer import GaussianTrainer, normalize_depth, flip_hairstyle, obtain_camera
from src.gaussian_utils.image_utils import vis_directed_orient


class GaussianTrainerMV(GaussianTrainer):
    """
    Multi-view Gaussian Trainer that computes losses for sampled views.
    
    Key design:
    - Receives a single randomly sampled view per iteration
    - Computes and returns loss for that view
    - Supports full multi-view computation when multiple views are passed
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_losses_for_view(self, gt, pred):
        """
        Compute losses for a single view (or batch of views).
        
        Args:
            gt: Ground truth dictionary with tensors of shape [n, C, H, W]
            pred: Prediction dictionary with tensors of shape [n, C, H, W]
            
        Returns:
            dict: Contains individual loss components
        """
        L1 = l1_loss(pred["image"], gt["image"])
        SSIM_loss = 1.0 - ssim(pred["image"], gt["image"])
        Lmask = l1_loss(pred["mask"], gt["mask"])
        Ldepth = l1_loss(pred["depth"], gt["depth"])
        
        orient_weight = torch.ones_like(gt["mask"][:, :1])
        if self.use_conf:
            orient_weight *= gt["orient_conf"]
        
        if self.use_directed_loss and gt["directed_map"] is not None:
            Lorient = or_loss_directed(
                vis_directed_orient(pred["orient_angle"]),
                gt["directed_map"][:, 1:],
                pred["orient_conf"] if self.use_conf else None,
                weight=orient_weight,
                mask=gt["mask"][:, :1],
                type=self.loss_type
            )
        else:
            Lorient = or_loss(
                pred["orient_angle"],
                gt["orient_angle"],
                pred["orient_conf"] if self.use_conf else None,
                weight=orient_weight,
                mask=gt["mask"][:, :1]
            )
        
        return {
            "l1": L1,
            "ssim": SSIM_loss,
            "mask": Lmask,
            "orient": Lorient,
            "depth": Ldepth
        }
    
    def step(self, strands, gt_cam, feats, scaling_factor, idx=0, iteration=0, 
             tb_writer=None, mode='train', flip=None, appearance=None, cam_idxes=None):
        """
        Perform a training step with the given view(s).
        
        In the random sampling strategy:
        - gt_cam and feats contain a SINGLE sampled view [batch, 1, ...]
        - Computes loss for that view
        
        Args:
            strands: Hair strands [batch, n_strands, n_points, 3]
            gt_cam: Camera matrices [batch, n_views, 3, 4] (typically n_views=1 after sampling)
            feats: Ground truth features [batch, n_views, C, H, W]
            scaling_factor: Camera scaling factor
            idx: Index (unused)
            iteration: Current iteration number
            tb_writer: TensorBoard writer (optional)
            mode: 'train' or 'eval'
            flip: Flip flag
            appearance: Appearance features (optional)
            cam_idxes: Camera indices (optional)
            
        Returns:
            tuple: (l1, ssim, mask, orient, depth, view_total_loss)
                   where view_total_loss is the combined loss for the sampled view
        """
        # Render and parse features
        parsed = self.render_and_parse_feats(strands, gt_cam, feats, scaling_factor, flip, appearance)
        gt, pred = parsed['gt'], parsed['pred']
        
        # Compute losses
        losses = self.compute_losses_for_view(gt, pred)
        
        # Compute total view loss (unweighted sum - weighting done in trainer)
        view_total = (losses["orient"] + losses["mask"] + losses["depth"]).detach()
        
        if tb_writer is not None:
            self.log_to_tensorboard(tb_writer, gt, pred, iteration, mode)
        
        # Return losses and the total view loss as a list for compatibility
        return (losses["l1"], losses["ssim"], losses["mask"], 
                losses["orient"], losses["depth"], [view_total])
