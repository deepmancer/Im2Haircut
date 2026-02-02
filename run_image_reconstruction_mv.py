# === Standard Library ===
import os
import random
import argparse
import pickle
import glob

# === Third-Party Libraries ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
import numpy as np
import cv2
import yaml
from pyhocon import ConfigFactory

# === Project-Specific Modules ===
import sys
sys.path.append("./submodules/external/GaussianHaircut")
from src.utils.general_utils import safe_state

from src.model_utils.geometry import decode_pca
from src.datasets.real_dataset_multiview import HairstyleRealDatasetMultiView

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

# === Environment and Torch Settings ===
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from trainers.trainer_single_mv import BaseTrainerMultiView
import trimesh


class PriorTrainerMultiView(BaseTrainerMultiView):
    """
    Multi-view trainer for hair reconstruction.
    
    Uses multiple views of the same subject for 3DGS optimization,
    while the hair strand prediction is generated from the frontal view only.
    """
    
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
                 subject='',
                 upsample_hairstyle=False, 
                 upsample_resolution=64,
                 num_steps_coarse=200,
                 optimize_appearance=False
                 ):
        
        nn.Module.__init__(self)
        
        self._init_basic_config(num_steps_coarse, device, ngpus, accumulate_gradients,
                            upsample_hairstyle, upsample_resolution, optimize_appearance, config, 
                            unfreeze_time_for_pca)

        self._init_dataset_multiview(config, subject, folder_name, world_size, rank, num_workers)

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


def main(args, dataset, opt, pipe, pointcloud_path_head, ip=None, port=None):
    # Configuration
    f = open(args.conf_path)
    conf_text = f.read()
    f.close()
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    conf = ConfigFactory.parse_string(conf_text)
    
    # Get data path from config or args
    data_path = conf['dataset'].get('data_path', './data/')
    folder_name = args.folder_name if args.folder_name else 'multiview_data'
    multiview_root = os.path.join(data_path, folder_name)
    
    # Determine list of subjects to process
    if args.subject:
        # Single subject specified via command line
        subjects = [args.subject]
    else:
        # Discover all subjects in multiview_data folder
        if not os.path.isdir(multiview_root):
            print(f"Error: Multiview data directory not found: {multiview_root}")
            return
        
        subjects = sorted([
            d for d in os.listdir(multiview_root)
            if os.path.isdir(os.path.join(multiview_root, d))
        ])
        
        if not subjects:
            print(f"No subjects found in {multiview_root}")
            return
    
    print('\n' + '='*60)
    print('IM2HAIRCUT - MULTI-VIEW BATCH PROCESSING')
    print('='*60)
    print(f'  Data root:     {multiview_root}')
    print(f'  Subjects:      {len(subjects)} found')
    for i, s in enumerate(subjects):
        print(f'    [{i+1}] {s}')
    print('='*60 + '\n')
    
    # Process each subject
    # subjects[0] = "sample_140"
    random.shuffle(subjects)
    for subj_idx, subject in enumerate(subjects):
        print('\n' + '#'*60)
        print(f'# SUBJECT {subj_idx + 1}/{len(subjects)}: {subject}')
        print('#'*60)
        
        # Check if preprocessed data exists
        subject_path = os.path.join(multiview_root, subject)
        aligned_path = os.path.join(subject_path, 'resized_img_aligned')
        
        if not os.path.isdir(aligned_path):
            print(f'  [SKIP] No preprocessed data found at: {aligned_path}')
            continue
        
        # Count available views
        view_files = glob.glob(os.path.join(aligned_path, '*.png'))
        if not view_files:
            print(f'  [SKIP] No view images found in: {aligned_path}')
            continue
        
        print(f'  Views found: {len(view_files)}')
        
        # Build save directory for this subject
        savedir = os.path.join(args.savedir_base, folder_name, subject) if args.savedir_base else args.savedir
        
        # Check if subject has already been processed
        final_output_path = os.path.join(savedir, 'pointclouds_train', 'pred_000300.ply')
        if os.path.exists(final_output_path):
            print(f'  [SKIP] Already processed - found: {final_output_path}')
            continue

        file_backup(os.path.join(savedir, 'recording'), args.conf_path, dir_lis=conf['general']['base_exp_dir'])
        
        # Generate a unique port for each subject to avoid binding conflicts
        subject_port = port + subj_idx * 10 + random.randint(0, 9)
        print(f'  Using port: {subject_port}')
        
        rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            print('\n' + '='*60)
            print('IM2HAIRCUT - MULTI-VIEW 3DGS OPTIMIZATION')
            print('='*60)
            print(f'  Subject:     {subject}')
            print(f'  Config:      {args.conf_path}')
            print(f'  Checkpoint:  {args.ckpt_path}')
            print(f'  Save dir:    {savedir}')
            print(f'  World size:  {world_size} GPU(s)')
            print('='*60 + '\n')
        
        try:
            training = PriorTrainerMultiView(
                conf, world_size, rank, device, global_rank, 
                ckpt_path=args.ckpt_path, 
                savedir=savedir,    
                unfreeze_time_for_pca=args.unfreeze_time_for_pca, 
                ngpus=args.ngpus, 
                num_workers=args.num_workers, 
                accumulate_gradients=args.accumulate_gradients, 
                dataset=dataset, 
                opt=opt,  
                pipe=pipe,  
                pointcloud_path_head=pointcloud_path_head, 
                ip=ip, 
                port=subject_port, 
                folder_name=folder_name, 
                subject=subject, 
                upsample_hairstyle=args.upsample_hairstyle,  
                upsample_resolution=args.upsample_resolution, 
                optimize_appearance=args.optimize_appearance, 
                num_steps_coarse=args.num_steps_coarse
            )

            training.train(world_size, rank, device, global_rank)
            
        except Exception as e:
            print(f'  [ERROR] Failed to process {subject}: {e}')
            import traceback
            traceback.print_exc()
        
        finally:
            # Clear CUDA cache between subjects
            torch.cuda.empty_cache()
    
    print('\n' + '='*60)
    print('ALL SUBJECTS PROCESSED')
    print('='*60 + '\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--ckpt_path', default='./pretrained_models/fine.pth', type=str)
    parser.add_argument('--savedir', default='./experiments', type=str, help='Save directory for single subject mode')
    parser.add_argument('--savedir_base', default='./exps_inverse_stage/multiview_batch', type=str, 
                        help='Base save directory for batch mode (subjects saved as savedir_base/folder_name/subject/)')
    parser.add_argument('--conf_path', default='./configs/static.conf', type=str)
    parser.add_argument('--ngpus', default=-1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--unfreeze_time_for_pca', default=-1, type=int)
    parser.add_argument('--upsample_hairstyle', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument("--folder_name", type=str, default='multiview_data', 
                        help='Folder name containing multiview subjects (relative to data_path in config)')
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pointcloud_path_head", type=str, default='./data/pointcloud.ply')
    parser.add_argument("--hair_conf_path", type=str, default=None)
    
    # Multi-view specific arguments
    parser.add_argument("--subject", type=str, default='', 
                        help="Subject name to process. If empty, all subjects in folder_name will be processed.")
    parser.add_argument("--multiview", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, 
                        help="Enable multi-view optimization")
    
    parser.add_argument('--upsample_resolution', type=int, default=256)
    parser.add_argument('--num_steps_coarse', type=int, default=20)
    parser.add_argument('--optimize_appearance', type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    main(args, lp.extract(args), op.extract(args), pp.extract(args), args.pointcloud_path_head, args.ip, args.port)
