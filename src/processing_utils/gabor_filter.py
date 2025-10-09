from PIL import Image
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage.filters import difference_of_gaussians
import math
import os
import tqdm
import cv2
import argparse
import torch
import numpy as np

from skimage.filters import gabor_kernel
from skimage.filters import difference_of_gaussians


class GaborFilter(torch.nn.Module):
    def __init__(
        self,
        sigma_x: float = 1.8,
        sigma_y: float = 2.4,
        freq: float = 0.23,
        num_filters: int = 180,
        low_sigma: float = 0.4,
        high_sigma: float = 10.0
    ):

        super().__init__()

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.freq = freq
        self.num_filters = num_filters
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

        self.weights = GaborFilter.generate_gabor_filters(sigma_x, sigma_y, freq, num_filters)
        self.weights = torch.tensor(self.weights, dtype=torch.float).unsqueeze(1)

        self.bins = torch.linspace(0, torch.pi * (num_filters - 1) / num_filters, num_filters)
        self.bins = self.bins.reshape(1, -1, 1, 1)

        self.weights = torch.nn.Parameter(self.weights)
        self.bins = torch.nn.Parameter(self.bins)

    @staticmethod
    def generate_gabor_filters(sigma_x, sigma_y, freq, num_filters) -> np.array:
        # Build Gabor filters for $[0, \pi]$ interval
        kernels = []
        for theta in np.linspace(0, np.pi * (num_filters - 1) / num_filters, num_filters):
            kernels.append(np.real(gabor_kernel(freq, theta=np.pi - theta, sigma_x=sigma_x, sigma_y=sigma_y)))

        # Largest dimension of a single kernel
        s = max(max(k.shape) for k in kernels)

        # Pad kernels with zeros to form a single tensor
        kernels = np.stack([np.pad(k, [[(s - k.shape[0]) // 2], [(s - k.shape[1]) // 2]]) for k in kernels])

        return kernels

    @staticmethod
    def rgb2gray(x: torch.Tensor) -> torch.Tensor:
        # Reshape to [batch, 1, h, w]
        return (0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]).unsqueeze(1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Returns orientation and confidence maps
        """

        gray = GaborFilter.rgb2gray(x)

        filtered = [difference_of_gaussians(t[0].cpu().numpy(), self.low_sigma, self.high_sigma) for t in gray]
        filtered = torch.tensor(np.stack(filtered)).unsqueeze(1)
        filtered = filtered.to(self.weights.device)

        response = torch.nn.functional.conv2d(
            filtered,
            self.weights,
            padding=(self.weights.shape[2] // 2, self.weights.shape[3] // 2))

        response = response.abs()

        # Orientation field
        orientation_map = response.argmax(dim=1, keepdim=True) / self.num_filters * 255
        
# #         # Confidence field

        orientation_map_rad = orientation_map / self.num_filters * torch.pi
        dists = torch.minimum(
            torch.abs(orientation_map_rad - self.bins),
            torch.minimum(
                torch.abs(orientation_map_rad - self.bins - torch.pi),
                torch.abs(orientation_map_rad - self.bins + torch.pi)
            )
        )

        F_orients_norm = response / response.sum(axis=1, keepdims=True)
        confidence_map = (dists**2 * F_orients_norm).sum(1)
        return orientation_map, confidence_map