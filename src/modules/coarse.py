import sys
import os


sys.path.append('./submodules/external/VOODOO3D-official/models')
sys.path.append('./submodules/external/VOODOO3D-official')

from additional_modules.deeplabv3.deeplabv3 import DeepLabV3
from lp3d_model import OverlapPatchEmbed, PositionalEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.model_utils.pos_enc import positional_encoding
from src.modules.decoder import Decoder, StrandPositionDecoder, StrandMaskDecoder   
from src.modules.blocks import Block

def _make_conv_layers(in_channels=32,
                      layer_out_channels=[256, 256, 256],
                      layer_kernel_sizes=[3, 3, 3], use_norm=True):
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1))
            if use_norm:
                print('use norm')
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    
def _make_conv_1x1_layers(in_channels=256,
                      layer_out_channels=[256, 256, 256],
                      layer_kernel_sizes=[1, 1, 1], use_norm=True):
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=0))
            if use_norm:
                print('use norm')
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)
    

class EHigh(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3, out_channels=96):
        super().__init__(img_size)

        self.conv1 = nn.Conv2d(img_channels + 2, 64, 7, 2, 3, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(64, 96, 3, 2, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv2d(96, 96, 3, 2, 1, bias=True)
        self.act3 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act4 = nn.LeakyReLU(0.01)

        self.conv5 = nn.Conv2d(96, out_channels, 3, 1, 1, bias=True)
        self.act5 = nn.LeakyReLU(0.01)

    def forward(self, img: torch.Tensor):
        x = self._add_positional_encoding(img)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)

        return x
    


class ELow(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3, ch=10, num_blocks=5, num_dim=1024, use_linear_head=False,  deconv_body_channels=[256, 256, 128], deconv_body_kernels=[3, 3, 3], deconv_head_channels=[128, 128, 64], deconv_head_kernels=[3, 3, 3], use_norm=True, low_ch=10, pooling='max', use_selected_averaging=False, use_pos_enc=False, pos_enc_feats=-1, init_for_posenc_random=False, fix_decoder=False):
        
        super().__init__(img_size)

        self.deeplabv3_backbone = EHigh(img_channels=img_channels + 2, out_channels=256)
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256, embed_dim=num_dim
        )
        
        self.blocks = nn.ModuleList([Block(dim=num_dim, num_heads=4, mlp_ratio=2, sr_ratio=1) for _ in range(num_blocks)])
        self.pooling = pooling
        self.use_selected_averaging = use_selected_averaging

        self.decoder = Decoder(latent_dim=num_dim, use_pos_enc=use_pos_enc, pos_enc_feats=pos_enc_feats, init_for_posenc_random=init_for_posenc_random)
        self.position_decoder = StrandPositionDecoder(ch=low_ch+1, use_pos_enc=use_pos_enc,pos_enc_feats=pos_enc_feats, init_for_posenc_random=init_for_posenc_random)
        self.mask_decoder = StrandMaskDecoder(mask=1)
        self.low_ch = low_ch
        
        if fix_decoder:
            print('i am fixing deocder network')
            self.decoder.requires_grad_(False).eval()
            self.position_decoder.requires_grad_(False).eval()
            self.mask_decoder.requires_grad_(False).eval()
            
        
        if self.pooling == 'wmean':
            self.weight_aggregator = nn.Linear(num_dim, 1)
        
    
#     @torch.no_grad()
    def forward_feats(self, img: torch.Tensor, mask=None):
        x = self._add_positional_encoding(img)
        x = self.deeplabv3_backbone(x)

        x, H, W = self.patch_embed(x)  
        
        for i in range(len(self.blocks)):
            
            x = self.blocks[i](x, H, W, mask=mask)
        
        if self.pooling == 'max':

            aggregated_vector = x.max(dim=1).values 
        elif self.pooling == 'mean':

            aggregated_vector = x.mean(dim=1)
        elif self.pooling == 'wmean':
            agg_weights = self.weight_aggregator(x) # Shape: (N, C, 1)
            agg_weights_norm = torch.softmax(agg_weights.squeeze(-1), dim=1)  # Shape: (N, C)
            aggregated_vector = torch.sum(x * agg_weights_norm.unsqueeze(-1), dim=1)  # Shape: (N, F) 
            
        decoded = self.decoder(aggregated_vector)  # Output: 512 x 32 x 32
        mask = self.mask_decoder(decoded)  # Output: 100 x 32 x 32
        position = self.position_decoder(decoded)  # Output: 300 x 32 x 32

        return aggregated_vector, x, position[:, :self.low_ch], torch.cat((position[:, self.low_ch:], mask), 1)


            
    def forward(self, img: torch.Tensor, mask=None):

        x = self._add_positional_encoding(img)

        x = self.deeplabv3_backbone(x)

        x, H, W = self.patch_embed(x)        

        for i in range(len(self.blocks)):
            
            x = self.blocks[i](x, H, W, mask=mask)

        if self.pooling == 'max':

            aggregated_vector = x.max(dim=1).values 
        elif self.pooling == 'mean':
            
            if self.use_selected_averaging:
                average_masking = (mask.reshape(mask.shape[0], -1, 1) > -100).float()
                sum_masked = (x * average_masking).sum(dim=1)

                valid_counts = average_masking.sum(dim=1)  # [bs, 1]

                # Avoid division by zero by setting invalid counts to 1 (or handle separately if needed)
                valid_counts = valid_counts.clamp_min(1)
                
                aggregated_vector = sum_masked / valid_counts 

            else:
                
                aggregated_vector = x.mean(dim=1)
                
        elif self.pooling == 'wmean':
            agg_weights = self.weight_aggregator(x) # Shape: (N, C, 1)
            agg_weights_norm = torch.softmax(agg_weights.squeeze(-1), dim=1)  # Shape: (N, C)
            aggregated_vector = torch.sum(x * agg_weights_norm.unsqueeze(-1), dim=1)  # Shape: (N, F)

        decoded = self.decoder(aggregated_vector)  # Output: 512 x 32 x 32
        mask = self.mask_decoder(decoded)  # Output: 100 x 32 x 32
        position = self.position_decoder(decoded)  # Output: 300 x 32 x 32

        return position[:, :self.low_ch], torch.cat((position[:, self.low_ch:], mask), 1)

    

class Lp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3,  predict_scale=False, low_ch=10, num_blocks=5, num_dim=1024, predict_mask=False, use_linear_head=False,  deconv_body_channels=[256, 256, 128], deconv_body_kernels=[3, 3, 3], deconv_head_channels=[128, 128, 64], deconv_head_kernels=[3, 3, 3], use_norm=True, use_mask_silh=False, pooling='max', use_selected_averaging=False, use_pos_enc=False, pos_enc_feats=-1, init_for_posenc_random=False, fix_decoder=False):
        super().__init__()
        self.img_size = img_size
        self.predict_scale = predict_scale
        self.use_mask_silh = use_mask_silh

        self.elo = ELow(img_size, img_channels, ch=low_ch+predict_scale+predict_mask, num_blocks=num_blocks, num_dim=num_dim, use_linear_head=use_linear_head, deconv_body_channels=deconv_body_channels, deconv_body_kernels=deconv_body_kernels, deconv_head_channels=deconv_head_channels, deconv_head_kernels=deconv_head_kernels,use_norm=use_norm, low_ch=low_ch, pooling=pooling, use_selected_averaging=use_selected_averaging, use_pos_enc=use_pos_enc, pos_enc_feats=pos_enc_feats, init_for_posenc_random=init_for_posenc_random, fix_decoder=fix_decoder)

        self.low_ch = low_ch
        self.use_linear_head = use_linear_head
        
    
    
    def forward_feats(self, img: torch.Tensor,  transformer_mask=None):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size
        
        if self.use_mask_silh:

            if transformer_mask is None:

                transformer_mask = -100 * (1 - F.interpolate(img[:, 1].unsqueeze(1), (32, 32)).reshape(img.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0)
    
                
        return self.elo.forward_feats(img, mask=transformer_mask)
        
        
     
    def forward(self, img: torch.Tensor,  transformer_mask=None):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        if self.use_mask_silh:

            if transformer_mask is None:

                transformer_mask = -100 * (1 - F.interpolate(img[:, 1].unsqueeze(1), (32, 32)).reshape(img.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0)
    
    
        else:
            transformer_mask = None
        
        return self.elo(img, mask=transformer_mask)
