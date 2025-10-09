import sys
import os

sys.path.append('./submodules/external/VOODOO3D-official/models')
sys.path.append('./submodules/external/VOODOO3D-official')

from additional_modules.deeplabv3.deeplabv3 import DeepLabV3
from lp3d_model import OverlapPatchEmbed, Block, PositionalEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.model_utils.pos_enc import positional_encoding
from collections import OrderedDict

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, use_pos_enc=False, pos_enc_feats=3, init_for_posenc_random=False):
        super().__init__()
        
        self.use_pos_enc = use_pos_enc
        
        self.additional_ch = pos_enc_feats
        if self.use_pos_enc:
            self.m1_positional_encoding = nn.Parameter(torch.zeros(1, 256, 8, 8)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 256, 8, 8))
            self.m2_positional_encoding = nn.Parameter(torch.zeros(1, 512, 16, 16)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 512, 16, 16))
            self.m3_positional_encoding = nn.Parameter(torch.zeros(1, 512, 32, 32)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 512, 32, 32))
            self.m4_positional_encoding = nn.Parameter(torch.zeros(1, 512, 64, 64)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 512, 64, 64))


        
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 4096)

        # Separate layers from sequential for visualization
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(256*2, 512, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        


    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        
#         print('inside decoder', x.shape, self.use_pos_enc)
        
        
        x = F.relu(self.fc2(x))
        
        
        x = x.view(-1, 256, 4, 4)  # Reshape to match Conv2D input
                
        
        
        x = self.upsample1(x)
        
        
        if self.use_pos_enc:
# #             print('inside posenc', x.shape)
            x = torch.cat((x, self.m1_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
# #             print('after posenc', x.shape)
        
        x = self.conv1(x)
#         print('inside conv1', x.shape)
        
        x = self.relu1(x)
        
        x = self.upsample2(x)
# #         print('inside upsample2', x.shape)
        if self.use_pos_enc:
            x = torch.cat((x, self.m2_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
# #             print('posenc2', x.shape)
        x = self.conv2(x)
#         print('inside conv2', x.shape)
        
        x = self.relu2(x)
        
        x = self.upsample3(x)
# #         print('inside upsample3', x.shape)
        if self.use_pos_enc:
            x = torch.cat((x, self.m3_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
#             print('posenc2', x.shape)
        x = self.conv3(x)
#         print('inside conv3', x.shape)
        
        x = self.relu3(x)
        
        x = self.upsample4(x)
#         print('inside u4', x.shape)
        if self.use_pos_enc:
            x = torch.cat((x, self.m4_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
        
        
        x = self.conv4(x)
#         print('inside conv4', self.conv4.weight.sum())
        
        x = self.relu4(x)
        
        return x
    
    
class StrandPositionDecoder(nn.Module):
    def __init__(self, ch, use_pos_enc, pos_enc_feats=3, init_for_posenc_random=False, zero_init=False):
        super(StrandPositionDecoder, self).__init__()
        self.use_pos_enc = use_pos_enc
        
        self.conv1 = nn.Conv2d(512*2, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512*2, 256, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
        self.conv3 = nn.Conv2d(256*2, ch, kernel_size=1, stride=1, padding=0)
        
        if self.use_pos_enc:
            self.m1_positional_encoding = nn.Parameter(torch.zeros(1, 512, 64,64)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 512, 64, 64))
            self.m2_positional_encoding = nn.Parameter(torch.zeros(1, 512, 64, 64)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 512, 64, 64))
            self.m3_positional_encoding = nn.Parameter(torch.zeros(1, 256, 64, 64)) if init_for_posenc_random is False else nn.Parameter(torch.randn(1, 256, 64, 64))
        
        if zero_init:   
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize the weights of the last layer to zero
        if isinstance(self.conv3, nn.Conv2d):
            nn.init.zeros_(self.conv3.weight)
            if self.conv3.bias is not None:
                nn.init.zeros_(self.conv3.bias)
        



    def forward(self, x):
#         print('inside StrandPositionDecoder1', x.shape)
        if self.use_pos_enc:
            x = torch.cat((x, self.m1_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
            
#         print('inside StrandPositionDecoder', x.shape)
        x = self.conv1(x)
#         print('StrandPositionDecoder dec2', x.shape)
        x = self.relu(x)
        if self.use_pos_enc:
            x = torch.cat((x, self.m2_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
            
        x = self.conv2(x)
#         print('dec2', x.shape)
        
        x = self.tanh(x)
        if self.use_pos_enc:
#             print('StrandPositionDecoder dec3', x.shape)
            x = torch.cat((x, self.m3_positional_encoding.repeat(x.shape[0], 1, 1, 1)), 1)
            
#         print('dec3', x.shape)
        x = self.conv3(x)
#         print('dec3', x.shape)
        
        return x

class StrandMaskDecoder(nn.Module):
    def __init__(self, mask):
        super(StrandMaskDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(64, mask, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)
    
