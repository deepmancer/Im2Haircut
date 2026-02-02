"""
Multi-view dataset for Im2Haircut.

This dataset loads multiple views of the same subject for multi-view 3DGS optimization.
The frontal view (view_000) is always used as the input to the encoder network.
"""

import pickle
from torch.nn import functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import torch.nn.functional as F
from skimage.transform import resize
from src.model_utils.preprocessing import erode_mask, normalized_depth_quantile, PILtoTorch


class HairstyleRealDatasetMultiView(Dataset):
    """
    Multi-view dataset for hair reconstruction.
    
    Data structure expected:
    multiview_data/
    ├── subject_name/
    │   ├── resized_img_aligned/
    │   │   ├── view_000.png  (frontal - used for encoder)
    │   │   ├── view_001.png
    │   │   └── ...
    │   ├── proj_matx_inv_aligned/
    │   │   ├── view_000.txt
    │   │   ├── view_001.txt
    │   │   └── ...
    │   └── ... (other preprocessed folders)
    """
    
    def __init__(self,
                 device='cpu',
                 path_to_meshgrid_data='',
                 image_size=512,
                 gs_scale_path="",
                 infer_path='',
                 convert_gabor_to_strand_map=False,
                 subject='',
                 use_align=True,
                 frontal_view='view_000'):
        """
        Args:
            device: Device to load tensors to
            path_to_meshgrid_data: Path to meshgrid data files
            image_size: Image resolution (assumed square)
            gs_scale_path: Path to Gaussian scale pickle file
            infer_path: Root path to the subject data (e.g., data/multiview_data/subject_name)
            convert_gabor_to_strand_map: Whether to convert gabor to strand map
            subject: Subject name (for logging)
            use_align: Whether to use aligned data
            frontal_view: Name of the frontal view file (without extension)
        """
        self.device = device
        self.image_size = image_size
        self.gs_scale_path = gs_scale_path
        self.convert_gabor_to_strand_map = convert_gabor_to_strand_map
        self.use_align = use_align
        self.root_path = infer_path
        self.scale_camera_factor = 1
        self.subject = subject
        self.frontal_view = frontal_view

        # Meshgrid and mask paths
        texture_size = 64
        self.path_to_faces_for_each_origin = os.path.join(path_to_meshgrid_data, f'faces_{texture_size}x{texture_size}.pth')
        self.path_to_coords_for_each_origin = os.path.join(path_to_meshgrid_data, f'coords_for_each_origin_{texture_size}x{texture_size}.pth')
        self.path_to_full_mask = os.path.join(path_to_meshgrid_data, f'scalp_mask_{texture_size}x{texture_size}.png')
        self.path_to_meshgrid = os.path.join(path_to_meshgrid_data, 'usc_mean_scalp_mask_07.png')

        # Load all views for this subject
        wo_align = '' if self.use_align is False else '_aligned'
        img_folder = os.path.join(self.root_path, 'resized_img' + wo_align)
        
        if os.path.exists(img_folder):
            all_views = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))])
        else:
            all_views = []
            
        self.view_list = all_views
        self.n_views = len(self.view_list)
        
        # Identify frontal view index
        self.frontal_idx = 0
        for i, view in enumerate(self.view_list):
            if view.startswith(self.frontal_view):
                self.frontal_idx = i
                break
        
        self._setup_meshgrid()

    def _setup_meshgrid(self):
        """Load and prepare meshgrid data."""
        self.faces_for_each_origin = torch.load(self.path_to_faces_for_each_origin).long().to(self.device)
        self.coords_for_each_origin = torch.load(self.path_to_coords_for_each_origin).float().to(self.device)

        full_mask_img = torch.tensor(cv2.imread(self.path_to_full_mask) / 255)[:, :, :1].squeeze(-1).to(self.device)
        meshgrid_mask_img = torch.tensor(cv2.imread(self.path_to_meshgrid) / 255)[:, :, :1].squeeze(-1).to(self.device)

        self.full_meshgrid_mask = F.interpolate(
            meshgrid_mask_img.unsqueeze(0).unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0) * full_mask_img

    def __len__(self):
        # Return 1 since we process all views of one subject together
        return 1

    def _get_view_path(self, view_name, folder):
        """Get path to a specific view's file in a folder."""
        wo_align = '' if self.use_align is False else '_aligned'
        return os.path.join(self.root_path, folder + wo_align, view_name)

    def _setup_image_for_view(self, view_name, flip=False):
        """Load and process orientation map image for a specific view."""
        img_path = self._get_view_path(view_name, 'orientation_maps')
        image = np.array(Image.open(img_path))[..., None]

        if flip:
            image = image[:, ::-1]

        image_tensor = torch.tensor(image[:, :, -1:] / 255., device=self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        return F.interpolate(image_tensor, size=self.image_size, mode="nearest")[0]

    def _use_depth_for_view(self, view_name, flip=False):
        """Load and normalize depth image for a specific view."""
        depth_filename = view_name.replace('.png', '.npz').replace('.jpg', '.npz')
        depth_path = self._get_view_path(depth_filename, 'depth_apple_pro')
        seg_path = self._get_view_path(view_name, 'seg')

        hair_mask = np.array(Image.open(seg_path)) / 255. > 0.5

        depth_img = normalized_depth_quantile(depth_path, hair_mask=hair_mask, errode=True, kernel=2)
        depth_img = resize(depth_img, (self.image_size, self.image_size), anti_aliasing=True)

        image_tensor = torch.tensor(depth_img[..., None])

        if flip:
            image_tensor = torch.flip(image_tensor, [1])

        return F.interpolate(image_tensor.float()[None].permute(0, 3, 1, 2), size=self.image_size, mode='bilinear')[0]

    def _use_silh_for_view(self, view_name, flip=False):
        """Load and blend silhouette images for a specific view."""
        seg_path = self._get_view_path(view_name, 'seg')
        body_path = self._get_view_path(view_name, 'body_img')

        silh_img = np.array(Image.open(seg_path)) / 255. > 0.5
        silh_img = silh_img[..., None] if silh_img.ndim == 2 else silh_img[:, :, :1]

        image = torch.tensor(silh_img, device=self.device).float()
        hair_silh_np = image.cpu().numpy()

        try:
            body_img = np.array(Image.open(body_path))[:, :, :1] / 255. > 0.5
        except:
            body_img = np.array(Image.open(body_path))[..., None] / 255. > 0.5

        body_tensor = torch.tensor(body_img, device=self.device).float()
        only_body = body_tensor - image
        image += 0.5 * only_body

        full_mask_np = image.cpu().numpy()

        if flip:
            image = torch.flip(image, [1])
            hair_silh_np = hair_silh_np[:, ::-1]
            full_mask_np = full_mask_np[:, ::-1]

        hair_silh_np = (hair_silh_np.squeeze(-1) * 255).astype(np.uint8)
        transformer_mask = full_mask_np.squeeze(-1).copy()

        if image.ndim < 3:
            image = image[:, :, None]

        image_tensor = F.interpolate(image[None].permute(0, 3, 1, 2), size=self.image_size, mode='nearest')[0]
        return image_tensor, transformer_mask

    def load_gaus_for_view(self, view_name, flip=False):
        """Load Gaussian features and camera for a specific view."""
        wo_align = '' if self.use_align is False else '_aligned'
        root = self.root_path

        # Paths
        seg_path = os.path.join(root, 'seg' + wo_align, view_name)
        silh_path = os.path.join(root, 'body_img' + wo_align, view_name)
        orient_path = os.path.join(root, 'orientation_maps' + wo_align, view_name)
        img_path = os.path.join(root, 'resized_img' + wo_align, view_name)
        strand_map_path = os.path.join(root, 'strand_map' + wo_align, view_name)
        cam_path = os.path.join(root, 'proj_matx_inv' + wo_align, view_name.split('.')[0] + '.txt')
        depth_path = os.path.join(root, 'depth_apple_pro' + wo_align, view_name.replace('.png', '.npz').replace('.jpg', '.npz'))

        # Load tensors
        image = PILtoTorch(Image.open(img_path), self.image_size)
        mask_body = PILtoTorch(Image.open(silh_path), self.image_size)
        mask_hair = PILtoTorch(Image.open(seg_path), self.image_size)
        orient_angle = PILtoTorch(Image.open(orient_path), self.image_size)
        orient_conf = PILtoTorch(Image.open(orient_path), self.image_size)
        strand_map = PILtoTorch(Image.open(strand_map_path), self.image_size)
        depth = torch.tensor(
            resize(normalized_depth_quantile(depth_path, hair_mask=mask_hair.numpy()[0], errode=True, kernel=2),
                   (self.image_size, self.image_size), anti_aliasing=True)[..., None]).permute(2, 0, 1)

        # Preprocess features
        gt_image = image[:3]
        gt_strand_map = strand_map[:3]
        gt_mask_body = mask_body[:1]
        gt_mask_hair = mask_hair[:1]
        gt_orient_angle = orient_angle[:1]
        gt_orient_conf = orient_conf[:1]
        gt_depth = depth[:1]

        # Combine gabor map with direction map from hairstep
        if self.convert_gabor_to_strand_map:
            strand_map_vis = gt_strand_map.permute(1, 2, 0)
            cos_theta = 2 * strand_map_vis[:, :, 1] - 1
            sin_theta = 2 * strand_map_vis[:, :, 2] - 1
            angles = torch.atan2(sin_theta, cos_theta) * 180 / torch.pi % 360

            hairstep_mask_up = ((angles > 90) & (angles <= 270)).clone()

            gabor_angle = (gt_orient_angle * 180)[0].clone()
            gabor_angle = (180 - gabor_angle) % 180
            gabor_mask_up = gabor_angle > 90

            gabor_angle += 180 * ((gabor_mask_up.float() - hairstep_mask_up.float()).abs() > 0).float()

            g_channel = torch.cos(gabor_angle / 180 * torch.pi) / 2 + 0.5
            b_channel = torch.sin(gabor_angle / 180 * torch.pi) / 2 + 0.5

            gabor_map_color = torch.stack([
                strand_map_vis[..., 0],
                g_channel * gt_mask_hair[0],
                b_channel * gt_mask_hair[0]
            ], dim=-1)

            gt_strand_map = gabor_map_color.permute(2, 0, 1)

        # Stack all features
        feats_gaus = torch.cat([
            gt_image, gt_mask_hair, gt_mask_body,
            gt_orient_angle, gt_orient_conf,
            gt_depth, gt_strand_map
        ], dim=0)

        if flip:
            feats_gaus = torch.flip(feats_gaus, dims=[-1])

        # Load camera
        scale_mat = np.eye(4, dtype=np.float32)
        with open(self.gs_scale_path, 'rb') as f:
            transform = pickle.load(f)
            scale_mat[:3, :3] *= transform['scale']
            scale_mat[:3, 3] = transform['translation']

        cam_gaus = np.loadtxt(cam_path) @ scale_mat
        return feats_gaus, cam_gaus

    def __getitem__(self, idx):
        """
        Returns data for all views of the subject.
        
        The frontal view is used for:
        1. Encoder input (image for predicting hair strands)
        2. gaus_feats_frontal and gaus_cam_frontal
        
        All views are used for:
        1. gaus_feats and gaus_cam (multi-view supervision)
        """
        flip = False
        baldness_mask = self.full_meshgrid_mask

        # Get frontal view name
        frontal_view_name = self.view_list[self.frontal_idx]

        # Load frontal view data for encoder input
        image = self._setup_image_for_view(frontal_view_name)
        silh_img, transformer_mask = self._use_silh_for_view(frontal_view_name)
        image = torch.cat((image, silh_img[:1]), 0)
        depth_img = self._use_depth_for_view(frontal_view_name)
        image = torch.cat((image, depth_img[:1]), 0)

        # Load ALL views for multi-view supervision
        gaus_feats = []
        gaus_cam = []

        for view_name in self.view_list:
            try:
                g_feats, g_cam = self.load_gaus_for_view(view_name)
                gaus_cam.append(g_cam)
                gaus_feats.append(g_feats)
            except Exception as e:
                print(f"Warning: Failed to load view {view_name}: {e}")
                continue

        gaus_feats = torch.stack(gaus_feats)  # [N_views, C, H, W]
        gaus_cam = np.stack(gaus_cam)          # [N_views, 3, 4]

        # Load frontal view for scalp rendering (used in trainer)
        gaus_cam_frontal = []
        gaus_feats_frontal = []
        g_feats_frontal, g_cam_frontal = self.load_gaus_for_view(frontal_view_name)
        gaus_cam_frontal.append(g_cam_frontal)
        gaus_feats_frontal.append(g_feats_frontal)

        gaus_feats_frontal = torch.stack(gaus_feats_frontal)
        gaus_cam_frontal = np.stack(gaus_cam_frontal)

        idx_of_name = 0

        return (
            image,                    # Frontal view for encoder [C, H, W]
            baldness_mask[None],      # Baldness mask [1, H, W]
            gaus_feats,               # All views features [N_views, C, H, W]
            gaus_cam,                 # All views cameras [N_views, 3, 4]
            torch.tensor([int(flip)]),
            transformer_mask,         # Transformer mask for frontal view
            torch.tensor([idx_of_name]),
            gaus_feats_frontal,       # Frontal view features [1, C, H, W]
            gaus_cam_frontal          # Frontal view camera [1, 3, 4]
        )
