import open3d as o3d
import torch
import os
import numpy as np
import trimesh
from tqdm import tqdm
import argparse



def main(args,device='cuda'):
    head_main =args.head_main

    root_path = args.root_path
    postfix = args.save_postfix

    path_original = f'{root_path}/bfm_meshes' + postfix
    cam_path = f'{root_path}/bfm_cameras' + postfix

    save_path = f'{root_path}/bfm_meshes_space' + postfix
    save_mesh_path = f'{root_path}/bfm_meshes_our_space' + postfix

    save_path_proj_matx = f'{root_path}/proj_matx' + postfix
    save_path_proj_matx_inv = f'{root_path}/proj_matx_inv' + postfix

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_mesh_path, exist_ok=True)
    os.makedirs(save_path_proj_matx, exist_ok=True)
    os.makedirs(save_path_proj_matx_inv, exist_ok=True)

    for idx in tqdm(range(len(os.listdir(path_original)))):
        try:
            
            name = os.listdir(path_original)[idx]    
            if os.path.exists(os.path.join(save_path_proj_matx_inv, name.replace('obj', 'txt'))):
                continue
            else: 

                path = os.path.join(path_original, name)
                head_base = np.array(trimesh.load_mesh(path).vertices)

                mesh = trimesh.load_mesh(path)

                transf_matx_walk = np.array([[0.13, -0.02, -0.03],
                                        [-0.02, -0.13, -0.02],
                                        [-0.02, 0.03, -0.13]]).T

                t_matx_walk = np.array([[[0.26, 1.94, 1.39]]])


                full_matx = np.eye(4)
                full_matx[:3, :3] = transf_matx_walk.T
                full_matx[:3, 3] = t_matx_walk


                transf_strands = np.matmul(head_base, transf_matx_walk)
                head_keypoints = transf_strands + t_matx_walk[0]

                reference_keypoints = np.array(trimesh.load(head_main).vertices) 

                head_keypoints_cloud = o3d.geometry.PointCloud()
                head_keypoints_cloud.points = o3d.utility.Vector3dVector(head_keypoints)

                reference_keypoints_cloud = o3d.geometry.PointCloud()
                reference_keypoints_cloud.points = o3d.utility.Vector3dVector(reference_keypoints)

                # Perform point-to-point registration
                transformation = o3d.pipelines.registration.registration_icp(
                    head_keypoints_cloud, reference_keypoints_cloud, 1, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))



                R = torch.tensor(transformation.transformation[:3, :3]).T
                T = torch.tensor(transformation.transformation[:3, 3])

                transf_head = np.matmul(head_keypoints, R)
                head_changed = np.array(transf_head+T[None])

                final_proj_matx = np.eye(4)
                final_proj_matx[:3, :3] = R.T
                final_proj_matx[:3, 3] = T

                P =  final_proj_matx @ full_matx 

                # Example: Transform head_base points using the projection matrix
                head_base_hom = np.hstack([head_base, np.ones((head_base.shape[0], 1))])  # Add 1 for homogeneous coords
                transformed_points_hom = head_base_hom @ P.T  # Apply projection matrix
                transformed_points = transformed_points_hom[:, :3]
                mesh.vertices = transformed_points

                _ = trimesh.PointCloud(head_changed).export(os.path.join(save_path, name))
                mesh.export(os.path.join(save_mesh_path, name))

                np.savetxt( os.path.join(save_path_proj_matx, name.replace('obj', 'txt')), P)

                final_projector = np.loadtxt(os.path.join(cam_path, name.replace('obj', 'txt'))) @ np.linalg.inv(P)
                np.savetxt( os.path.join(save_path_proj_matx_inv, name.replace('obj', 'txt')), final_projector)
        except Exception as e:
            print('continue')
            continue
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--root_path', default= './', type=str)
    parser.add_argument('--head_main', default= f'/home/vsklyarova/Projects/VAE_channel_reduction/Baselines/data/head_prior.obj', type=str) 
    parser.add_argument('--save_postfix', default= '', type=str)


    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)  