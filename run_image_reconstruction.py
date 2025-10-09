# === Standard Library ===
import os
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

from trainers.trainer_single import BaseTrainer
import trimesh

             

class PriorTrainer(BaseTrainer):
    def __init__(self,
                 config,
                 world_size,
                 rank, 
                 device,
                 global_rank,
                 ckpt_path=None,
                 savedir='./exps/',
                 unfreeze_time_for_pca=-1,
                 ngpus=-1,
                 num_workers=0,
                 accumulate_gradients=1,
                 dataset=None,
                 opt=None,
                 pipe=None,
                 pointcloud_path_head=None, 
                 ip=None,
                 port=None,
                 folder_name='', 
                 scene='',
                 upsample_hairstyle=False, 
                 upsample_resolution=64,
                 num_steps_coarse=200,
                 optimize_appearance=False
                 ):
        
        nn.Module.__init__(self)
        
        self._init_basic_config(num_steps_coarse, device, ngpus, accumulate_gradients,
                            upsample_hairstyle, upsample_resolution, optimize_appearance, config, 
                            unfreeze_time_for_pca)

        self._init_dataset(config, scene, folder_name, world_size, rank, num_workers)

        self._init_roots_and_blend_shapes()

        self._init_config(config)
            
        self._init_encoders(config, device, rank)

        self._init_gaussian_trainer(dataset, opt, pipe, pointcloud_path_head, ip, port, rank, config)

        self._setup_dirs_and_writer(savedir)

        if ckpt_path:
            print('Loading checkpoint...')
            self.load_model(ckpt_path, rank)
       
        self.optimizer = self.configure_optimizers()

    
    def training_step(self, batch, batch_idx, world_size, rank, device, global_rank, mode='train'):

        pred_points_vis = self.update_hairstyle(batch, world_size, rank, device, global_rank)[None]

        loss, logs = self.single_step(pred_points_vis, batch, batch_idx, world_size, rank, device, global_rank)
        
        if self.step % self.save_freq == 0 and rank == 0:
            print('start saving')
            self.save_model()

        return loss, logs
        
       
    def configure_optimizers(self, coarse=False):
        params = list(self.lp_enc.parameters())
        
        if self.finetune_coarse_model:
            params += list(self.lp_enc_elow.parameters())
        

        if self.optimizer_type == 'adam':
            opt_ae = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
            
        elif self.optimizer_type == 'adamw':
            opt_ae = torch.optim.AdamW(filter(lambda p: p.requires_grad, params),
                                  lr=self.learning_rate, weight_decay=self.weight_decay)
            
        return opt_ae


    
    
def main(args, dataset, opt,  pipe, pointcloud_path_head,  ip=None, port=None):


    # Configuration
    f = open(args.conf_path)
    
    conf_text = f.read()
    f.close()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    conf = ConfigFactory.parse_string(conf_text)

    file_backup(os.path.join(args.savedir, 'recording'), args.conf_path, dir_lis=conf['general']['base_exp_dir'])
    
    dist.init()
    rank = dist.get_local_rank()
    global_rank = dist.get_global_rank()
    device = torch.device(rank)
    world_size = dist.get_world_size()
    print(f'Starting in machine {device} which is at rank {global_rank} of world size {world_size} and rank {rank}')
    dist.print0(f'\n\nDistributing across {world_size} GPUs\n\n')
       
    training = PriorTrainer(conf, world_size, rank, device, global_rank, ckpt_path=args.ckpt_path, savedir=args.savedir,    unfreeze_time_for_pca=args.unfreeze_time_for_pca, ngpus=args.ngpus, num_workers=args.num_workers, accumulate_gradients=args.accumulate_gradients, dataset=dataset, opt=opt,  pipe=pipe,  pointcloud_path_head=pointcloud_path_head, ip=ip, port=port, folder_name=args.folder_name, scene=args.scene, upsample_hairstyle=args.upsample_hairstyle,  upsample_resolution=args.upsample_resolution, optimize_appearance=args.optimize_appearance, num_steps_coarse=args.num_steps_coarse)

    
                
                
    training.train(world_size,  rank, device, global_rank)

    dist.cleanup()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--savedir', default='./experiments', type=str)
    parser.add_argument('--conf_path', default='./configs/base.conf', type=str)
    parser.add_argument('--ngpus', default=-1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--unfreeze_time_for_pca', default=-1, type=int)
    parser.add_argument('--upsample_hairstyle', default=False, type=bool)
    parser.add_argument("--folder_name", type=str, default = '')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.10")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pointcloud_path_head", type=str, default = None)
    parser.add_argument("--hair_conf_path", type=str, default = None)
    parser.add_argument("--scene", type=str, default = '')
    
    parser.add_argument('--upsample_resolution', type=int, default=64)
    parser.add_argument('--num_steps_coarse', type=int, default=20)
    parser.add_argument('--optimize_appearance', type=bool, default=False)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    # Initialize system state (RNG)
    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    

    main(args, lp.extract(args), op.extract(args),  pp.extract(args), args.pointcloud_path_head, args.ip, args.port)
 
