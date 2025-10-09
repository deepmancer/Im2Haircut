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
from collections import OrderedDict
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
        x = self._add_positional_encoding(x)
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
    
    

class Lp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3,  predict_scale=False, low_ch=10, hr_ch=54, num_blocks=5, num_dim=1024,  use_mask_silh=False, pooling='max', use_selected_averaging=False, zero_init=False, use_pos_enc=False, pos_enc_feats=-1, init_for_posenc_random=False, ckpt_path_elow=''):
        super().__init__()
        self.img_size = img_size
        self.predict_scale = predict_scale
        self.use_mask_silh = use_mask_silh
        
        
        checkpoint = torch.load(ckpt_path_elow, map_location='cpu')
        state_dict = checkpoint['lp_enc']

        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            new_key = k.replace('module.elo.', '')  # Remove `module.` prefix
            new_state_dict[new_key] = v

#         elow.load_state_dict(new_state_dict)
        decoder_parameters = OrderedDict({k.replace('decoder.', ''): v for k, v in new_state_dict.items() if 'decoder' in k and 'position_decoder' not in k and 'mask_decoder' not in k})
        decoder_pos_parameters = OrderedDict({k.replace('position_decoder.', ''): v for k, v in new_state_dict.items() if 'position_decoder' in k and 'conv3' not in k})
        decoder_mask_parameters = OrderedDict({k.replace('mask_decoder.', ''): v for k, v in new_state_dict.items() if 'mask_decoder' in k})
        ehi_parameters = OrderedDict({k.replace('deeplabv3_backbone.', ''): v for k, v in new_state_dict.items() if 'deeplabv3_backbone' in k})
        print('load ehi', img_channels)
        self.ehi = EHigh(img_channels=img_channels+2, out_channels=256)
        self.ehi.load_state_dict(ehi_parameters)
        
        
        self.low_ch = low_ch
        self.hr_ch = hr_ch

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256, embed_dim=num_dim
        )

        self.pooling = pooling
        self.use_selected_averaging = use_selected_averaging


        self.decoder = Decoder(latent_dim=num_dim, use_pos_enc=use_pos_enc, pos_enc_feats=pos_enc_feats, init_for_posenc_random=init_for_posenc_random)
        self.decoder.load_state_dict(decoder_parameters)
        print('load deocer params')

        self.position_decoder = StrandPositionDecoder(ch=hr_ch+1, use_pos_enc=use_pos_enc,pos_enc_feats=pos_enc_feats, init_for_posenc_random=init_for_posenc_random, zero_init=zero_init)
        self.position_decoder.load_state_dict(decoder_pos_parameters, strict=False)
        print('load pos decoder')
        
        self.mask_decoder = StrandMaskDecoder(mask=1)
        self.mask_decoder.load_state_dict(decoder_mask_parameters)
        print('load mask decoder')
        
        
        
        if self.pooling == 'wmean':
            self.weight_aggregator = nn.Linear(num_dim, 1)


        self.conv1 = nn.Conv2d(384, 256, 3, 1, 1, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256, embed_dim=num_dim
        )

        self.blocks = nn.ModuleList([Block(dim=num_dim, num_heads=4, mlp_ratio=2, sr_ratio=1) for _ in range(num_blocks)])
        self.up = nn.PixelShuffle(upscale_factor=2)




    def forward(self, img: torch.Tensor,  transformer_mask=None, elo=None):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        if self.use_mask_silh:

            if transformer_mask is None:
                transformer_mask = -100 * (1 - (F.interpolate(img[:, 1].unsqueeze(1), (32, 32)).reshape(img.shape[0], -1).unsqueeze(1).unsqueeze(2) > 0).float())
        else:
            transformer_mask=None

        f_hi = self.ehi(img) #[8, 256, 64, 64]
        
        _, feats_lo, texture_low, coarse_mask_scale = safe_forward_feats(elo, img, transformer_mask=transformer_mask)
        
        f_lo = feats_lo.reshape(img.shape[0], 32, 32, -1).permute(0, 3, 1, 2).contiguous()

        f_lo = self.up(f_lo)
        
        x = torch.cat((f_lo, f_hi), dim=1)


        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x, H, W = self.patch_embed(x)        

        for i in range(len(self.blocks)):
            
            x = self.blocks[i](x, H, W, mask=transformer_mask)

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
        
        fine_mask_scale = torch.cat((position[:, self.hr_ch:], mask), 1)
        
        texture_hr = position[:, :self.hr_ch]    
                     
        return torch.cat((texture_low, texture_hr), 1), fine_mask_scale


# Universal function to handle DDP or regular models
def safe_forward_feats(model, *args, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.forward_feats(*args, **kwargs)
    else:
        return model.forward_feats(*args, **kwargs)