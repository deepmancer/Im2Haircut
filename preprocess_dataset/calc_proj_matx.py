import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import trimesh
from tqdm import tqdm
import argparse
from PIL import Image
import sys
import cv2

# Add the preprocess_dataset directory to path for importing FacialLandmarkDetector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facial_landmark_detector import FacialLandmarkDetector


class PerspectiveCamera(nn.Module):
    """
    Differentiable perspective camera model for optimization.
    
    Uses a simple parameterization:
    - Translation (tx, ty, tz): camera position offset
    - Rotation (rx, ry, rz): Euler angles in radians
    - Focal length (log scale for stability)
    
    The 3D landmarks are in a coordinate system where:
    - The face looks towards -Z (nose points to -Z)
    - +Y is up
    - +X is to the face's left (viewer's right)
    
    The 2D image has:
    - Origin at top-left
    - +X to the right
    - +Y downward
    """
    
    def __init__(self, focal_length_mm=75.0, sensor_width_mm=36.0, image_width=512, image_height=512,
                 init_translation=None, device='cuda'):
        super().__init__()
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        
        # Convert focal length from mm to pixels
        focal_pixels = focal_length_mm * image_width / sensor_width_mm
        
        # Optimizable parameters
        self.log_focal = nn.Parameter(torch.tensor(np.log(focal_pixels), dtype=torch.float32, device=device))
        
        # Principal point (fixed at image center)
        self.register_buffer('cx', torch.tensor(image_width / 2.0, dtype=torch.float32, device=device))
        self.register_buffer('cy', torch.tensor(image_height / 2.0, dtype=torch.float32, device=device))
        
        # Translation parameters (will be optimized)
        if init_translation is None:
            init_translation = [0.0, 0.0, 1.0]  # Start 1 unit away in Z
        self.translation = nn.Parameter(torch.tensor(init_translation, dtype=torch.float32, device=device))
        
        # Rotation parameters (Euler angles: rx, ry, rz)
        # Initialize to identity rotation
        self.rotation_euler = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        
        # Scale parameter for the 3D model (helps with matching)
        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    
    def euler_to_rotation_matrix(self, euler):
        """Convert Euler angles (rx, ry, rz) to rotation matrix.
        
        Uses ZYX convention (yaw-pitch-roll): R = Rz @ Ry @ Rx
        All operations are differentiable.
        """
        rx, ry, rz = euler[0], euler[1], euler[2]
        
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        
        # Combined rotation matrix elements (Rz @ Ry @ Rx)
        # More numerically stable to compute directly
        one = torch.ones(1, device=self.device, dtype=torch.float32)
        zero = torch.zeros(1, device=self.device, dtype=torch.float32)
        
        r00 = cy * cz
        r01 = cz * sx * sy - cx * sz
        r02 = cx * cz * sy + sx * sz
        
        r10 = cy * sz
        r11 = cx * cz + sx * sy * sz
        r12 = cx * sy * sz - cz * sx
        
        r20 = -sy
        r21 = cy * sx
        r22 = cx * cy
        
        # Build rotation matrix by stacking
        row0 = torch.stack([r00, r01, r02])
        row1 = torch.stack([r10, r11, r12])
        row2 = torch.stack([r20, r21, r22])
        R = torch.stack([row0, row1, row2])
        
        return R
    
    def get_focal_length(self):
        """Get focal length in pixels."""
        return torch.exp(self.log_focal)
    
    def get_scale(self):
        """Get scale factor."""
        return torch.exp(self.log_scale)
    
    def project(self, points_3d):
        """
        Project 3D points to 2D image coordinates.
        
        The projection pipeline:
        1. Apply scale
        2. Apply rotation (includes coordinate system conversion)
        3. Apply translation (move in front of camera)
        4. Perspective divide
        5. Apply intrinsics (focal length, principal point)
        
        Coordinate systems:
        - 3D world (head prior): +X right, +Y up, +Z towards viewer (face looks towards +Z)
        - Camera (OpenCV convention): +X right, +Y down, +Z into scene
        - 2D image: Origin top-left, +X right, +Y down
        
        The Y-flip is handled by the rotation matrix, not in the intrinsics.
        This ensures compatibility with cv2.decomposeProjectionMatrix.
        
        Args:
            points_3d: (N, 3) tensor of 3D points in head prior space
            
        Returns:
            points_2d: (N, 2) tensor of 2D points in image coordinates
        """
        # Get parameters
        f = self.get_focal_length()
        s = self.get_scale()
        R_model = self.euler_to_rotation_matrix(self.rotation_euler)
        t = self.translation
        
        # Flip matrix to convert from world (+Y up) to camera (+Y down) convention
        # This flips Y and Z: [1, 0, 0; 0, -1, 0; 0, 0, -1]
        flip_yz = torch.tensor([[1., 0., 0.],
                                 [0., -1., 0.],
                                 [0., 0., -1.]], device=self.device, dtype=torch.float32)
        
        # Combined rotation: first apply model rotation, then flip
        R = torch.mm(flip_yz, R_model)
        
        # Apply scale
        points_scaled = points_3d * s
        
        # Apply rotation: rotated = R @ points.T -> (3, N), then transpose back
        points_rotated = torch.mm(R, points_scaled.T).T  # (N, 3)
        
        # Apply translation (t should place points in front of camera with positive Z)
        points_cam = points_rotated + t.unsqueeze(0)
        
        # Extract coordinates in camera space
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Ensure positive depth (points must be in front of camera)
        z_safe = torch.clamp(z, min=0.01)
        
        # Perspective projection
        x_proj = x / z_safe
        y_proj = y / z_safe
        
        # Apply intrinsics (standard convention: positive focal lengths)
        x_img = f * x_proj + self.cx
        y_img = f * y_proj + self.cy  # No flip here - handled by rotation
        
        points_2d = torch.stack([x_img, y_img], dim=1)
        return points_2d


def optimize_camera(camera, landmarks_3d, landmarks_2d, num_steps=1500, min_steps=500, 
                    lr=0.01, patience=100, verbose=False):
    """
    Optimize camera parameters to minimize reprojection error.
    
    Args:
        camera: PerspectiveCamera instance
        landmarks_3d: (N, 3) tensor of 3D landmarks in head prior space
        landmarks_2d: (N, 2) tensor of 2D landmarks (ground truth)
        num_steps: maximum number of optimization steps
        min_steps: minimum number of steps before early stopping
        lr: learning rate
        patience: early stopping patience
        verbose: print optimization progress
        
    Returns:
        best_loss: final reprojection error
    """
    optimizer = optim.Adam(camera.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                       patience=50, min_lr=1e-6)
    
    best_loss = float('inf')
    best_state = None
    no_improvement_count = 0
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Project 3D landmarks to 2D (no centering - direct head prior space)
        projected_2d = camera.project(landmarks_3d)
        
        # Compute reprojection loss (mean L1 distance per landmark)
        loss = torch.mean(torch.sum(torch.abs(projected_2d - landmarks_2d), dim=1))

        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        current_loss = loss.item()
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.clone() for k, v in camera.state_dict().items()}
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if verbose and step % 100 == 0:
            print(f"Step {step}: Loss = {current_loss:.4f}, Best = {best_loss:.4f}")
        
        # Early stopping (only after min_steps)
        if step >= min_steps and no_improvement_count >= patience:
            if verbose:
                print(f"Early stopping at step {step}")
            break
    
    # Restore best state
    if best_state is not None:
        camera.load_state_dict(best_state)
    
    return best_loss


def compute_projection_matrices_from_camera(camera, image_width, image_height):
    """
    Compute projection matrices P and final_projector from optimized camera.
    
    This maintains compatibility with the original code's conventions:
    - P: 4x4 transformation matrix (head prior space to camera space)
    - final_projector: 3x4 projection matrix (head prior space to pixel space)
    
    The projection pipeline matches what project() does:
    1. Scale points: p_scaled = s * p
    2. Apply rotation with flip: p_rotated = R_final @ p_scaled where R_final = flip_yz @ R_model
    3. Add translation: p_cam = p_rotated + t
    4. Project: p_img = K @ p_cam (with positive focal lengths)
    
    The flip is embedded in R (not K) so cv2.decomposeProjectionMatrix works correctly.
    
    Args:
        camera: optimized PerspectiveCamera
        image_width, image_height: image dimensions
    
    Returns:
        P: 4x4 numpy array (head prior to camera transform)
        K_3x4: 3x4 projection matrix (head prior to pixels)
    """
    with torch.no_grad():
        f = camera.get_focal_length().item()
        s = camera.get_scale().item()
        R_model = camera.euler_to_rotation_matrix(camera.rotation_euler).cpu().numpy()
        t = camera.translation.cpu().numpy()
        cx = camera.cx.item()
        cy = camera.cy.item()
        
        # Flip matrix: converts from world (+Y up, +Z towards viewer) 
        # to camera (+Y down, +Z into scene) convention
        # This is embedded in R so cv2.decomposeProjectionMatrix gives positive focal lengths
        flip_yz = np.array([[1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., -1.]])
        
        # Combined rotation with flip embedded
        R_final = flip_yz @ R_model
        
        # Build the full 4x4 transformation matrix
        # The transformation is: p_cam = R_final @ (s * p) + t
        # In homogeneous form: [s*R_final | t]
        P = np.eye(4)
        P[:3, :3] = s * R_final
        P[:3, 3] = t
        
        # Build intrinsic matrix K (3x3) with POSITIVE focal lengths
        # The Y-flip is handled by R_final, not K
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        # Full projection matrix: K @ [R | t] = K @ P[:3, :]
        K_3x4 = K @ P[:3, :]
        
    return P, K_3x4


def save_landmark_visualization(img_array, landmarks_2d_gt, landmarks_2d_proj, save_path):
    """
    Save visualization of ground truth and projected landmarks overlaid on image.
    
    Args:
        img_array: numpy array of image (H, W, 3) in RGB
        landmarks_2d_gt: (N, 2) ground truth 2D landmarks (green)
        landmarks_2d_proj: (N, 2) projected 2D landmarks (red)
        save_path: path to save the visualization
    """
    # Create a copy of the image
    vis_img = img_array.copy()
    
    # Draw lines connecting corresponding landmarks (yellow)
    for i in range(len(landmarks_2d_gt)):
        gt_pt = (int(round(landmarks_2d_gt[i, 0])), int(round(landmarks_2d_gt[i, 1])))
        proj_pt = (int(round(landmarks_2d_proj[i, 0])), int(round(landmarks_2d_proj[i, 1])))
        cv2.line(vis_img, gt_pt, proj_pt, (255, 255, 0), 1)
    
    # Draw ground truth landmarks in green (larger)
    for i, (x, y) in enumerate(landmarks_2d_gt):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
            cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
    
    # Draw projected landmarks in red (smaller, on top)
    for i, (x, y) in enumerate(landmarks_2d_proj):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
            cv2.circle(vis_img, (x, y), 2, (255, 0, 0), -1)
    
    # Save the visualization
    Image.fromarray(vis_img).save(save_path)


def main(args, device='cuda'):
    head_main = args.head_main
    root_path = args.root_path
    postfix = args.save_postfix
    
    # Input paths
    img_path = f'{root_path}/resized_img' + postfix.replace('_aligned', '') if '_aligned' in postfix else f'{root_path}/resized_img'
    if '_aligned' in postfix:
        img_path = f'{root_path}/resized_img_aligned'
    
    # Output paths (maintaining original naming convention)
    save_path = f'{root_path}/bfm_meshes_space' + postfix
    save_mesh_path = f'{root_path}/bfm_meshes_our_space' + postfix
    save_path_proj_matx = f'{root_path}/proj_matx' + postfix
    save_path_proj_matx_inv = f'{root_path}/proj_matx_inv' + postfix
    save_path_vis = f'{root_path}/landmark_vis' + postfix
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_mesh_path, exist_ok=True)
    os.makedirs(save_path_proj_matx, exist_ok=True)
    os.makedirs(save_path_proj_matx_inv, exist_ok=True)
    os.makedirs(save_path_vis, exist_ok=True)
    
    # Load head prior mesh
    head_prior_mesh = trimesh.load_mesh(head_main)
    head_prior_vertices = np.array(head_prior_mesh.vertices)
    
    # Load 478 landmark indices for head_prior.obj
    lmk_indices_path = args.lmk_indices_path
    lmk_indices_478 = np.load(lmk_indices_path)
    
    # Extract 3D landmarks (478 points) from head prior
    landmarks_3d_478 = head_prior_vertices[lmk_indices_478]
    
    # Initialize facial landmark detector
    detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        enable_fallback=True,
        enable_preprocessing=True
    )
    
    # Convert 478 landmarks to 68 landmarks using the detector's correspondence
    landmarks_3d_68 = detector.convert_mediapipe_to_dlib68(landmarks_3d_478)
    landmarks_3d_68_tensor = torch.tensor(landmarks_3d_68, dtype=torch.float32, device=device)
    
    print(f"Loaded head prior with {len(head_prior_vertices)} vertices")
    print(f"Using {len(landmarks_3d_68)} 3D landmarks for camera optimization")
    print(f"3D landmarks range: X=[{landmarks_3d_68[:, 0].min():.4f}, {landmarks_3d_68[:, 0].max():.4f}], "
          f"Y=[{landmarks_3d_68[:, 1].min():.4f}, {landmarks_3d_68[:, 1].max():.4f}], "
          f"Z=[{landmarks_3d_68[:, 2].min():.4f}, {landmarks_3d_68[:, 2].max():.4f}]")
    
    # Get list of images to process
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    if os.path.exists(img_path):
        for f in os.listdir(img_path):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_files.append(f)
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images to process in {img_path}")
    
    # Print 3D landmark statistics for debugging
    head_center = np.mean(landmarks_3d_68, axis=0)
    face_width = landmarks_3d_68[:, 0].max() - landmarks_3d_68[:, 0].min()
    face_height = landmarks_3d_68[:, 1].max() - landmarks_3d_68[:, 1].min()
    face_size = max(face_width, face_height)
    
    print(f"Head center: {head_center}")
    print(f"Face size (3D): {face_size:.4f}")
    
    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            # Check if already processed
            output_name = os.path.splitext(img_name)[0] + '.txt'
            if os.path.exists(os.path.join(save_path_proj_matx_inv, output_name)):
                continue
            
            # Load image
            img_file = os.path.join(img_path, img_name)
            img = Image.open(img_file).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            img_height, img_width = img_array.shape[:2]
            
            # Detect 2D facial landmarks
            result = detector.get_lmk_full(img_array)
            
            if result is None or result['ldm68'] is None:
                print(f"Warning: Could not detect landmarks in {img_name}, skipping...")
                continue
            
            landmarks_2d_68 = result['ldm68']  # (68, 2) array
            landmarks_2d_68_tensor = torch.tensor(landmarks_2d_68, dtype=torch.float32, device=device)
            
            # Estimate initial Z distance based on face size in 2D vs 3D
            lmk_2d_size = max(landmarks_2d_68[:, 0].max() - landmarks_2d_68[:, 0].min(),
                             landmarks_2d_68[:, 1].max() - landmarks_2d_68[:, 1].min())
            
            # focal_pixels = 50mm * 512 / 36mm ≈ 711 pixels
            # For projection: 2d_size ≈ focal * 3d_size / z
            # So: z ≈ focal * 3d_size / 2d_size
            focal_pixels = 50.0 * img_width / 36.0
            init_z = focal_pixels * face_size / lmk_2d_size
            
            # Initial translation to place the head prior in front of camera
            # Head prior has face centered at approximately (0, 1.7, 0.09) in world space
            # After flip_yz transform: world (x, y, z) -> camera (x, -y, -z)
            # So head center at (0, 1.7, 0.09) becomes (0, -1.7, -0.09) in camera space
            # We need to add translation to:
            # 1. Move Z to positive (in front of camera): t.z should make final z positive
            # 2. Center Y in image: after -y, we get -1.7, so t.y should compensate
            # 
            # For a point at world (0, 1.7, 0.09):
            # After flip: (0, -1.7, -0.09)
            # After translation t: (t.x, t.y - 1.7, t.z - 0.09)
            # We want z > 0 and y ~ 0 (centered)
            # So: t.z > 0.09 (use init_z), t.y ~ 1.7 to center
            init_translation = [0.0, head_center[1], init_z]
            
            # Initialize camera
            camera = PerspectiveCamera(
                focal_length_mm=50.0,
                sensor_width_mm=36.0,
                image_width=img_width,
                image_height=img_height,
                init_translation=init_translation,
                device=device
            )
            
            # Optimize camera parameters
            final_loss = optimize_camera(
                camera=camera,
                landmarks_3d=landmarks_3d_68_tensor,
                landmarks_2d=landmarks_2d_68_tensor,
                num_steps=2000,
                min_steps=500,
                lr=0.05,
                patience=150,
                verbose=True
            )
            
            print(f"Final loss for {img_name}: {final_loss:.4f}")
            
            # Get final projected landmarks for visualization
            with torch.no_grad():
                final_projected = camera.project(landmarks_3d_68_tensor).cpu().numpy()
            
            # Save visualization
            vis_path = os.path.join(save_path_vis, os.path.splitext(img_name)[0] + '_landmarks.png')
            save_landmark_visualization(img_array, landmarks_2d_68, final_projected, vis_path)
            
            # Compute projection matrices from optimized camera
            P_camera, K_3x4 = compute_projection_matrices_from_camera(camera, img_width, img_height)
            
            # Debug: print projection matrix info
            print(f"  Optimized camera params:")
            print(f"    Focal length: {camera.get_focal_length().item():.1f} px")
            print(f"    Scale: {camera.get_scale().item():.4f}")
            print(f"    Translation: {camera.translation.cpu().detach().numpy()}")
            print(f"    Rotation (euler): {camera.rotation_euler.cpu().detach().numpy()}")
            print(f"  K_3x4 (final_projector):")
            print(f"    {K_3x4[0,:]}")
            print(f"    {K_3x4[1,:]}")
            print(f"    {K_3x4[2,:]}")
            
            # The final projection matrix P combines the coordinate transformation with camera
            # In original code: P = final_proj_matx @ full_matx
            # Here, P_camera already contains the camera extrinsics
            # We need to maintain the same convention
            
            # P transforms from BFM space to "our space"
            # Since we're using head_prior directly, we use P_camera as the transformation
            P = P_camera.copy()
            
            # Transform head prior vertices using P
            head_prior_hom = np.hstack([head_prior_vertices, np.ones((head_prior_vertices.shape[0], 1))])
            transformed_vertices_hom = head_prior_hom @ P.T
            transformed_vertices = transformed_vertices_hom[:, :3]
            
            # Create transformed mesh
            transformed_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=head_prior_mesh.faces)
            
            # Save point cloud of transformed landmarks
            landmarks_transformed = landmarks_3d_68 @ P[:3, :3].T + P[:3, 3]
            _ = trimesh.PointCloud(landmarks_transformed).export(
                os.path.join(save_path, os.path.splitext(img_name)[0] + '.obj')
            )
            
            # Save transformed mesh
            transformed_mesh.export(os.path.join(save_mesh_path, os.path.splitext(img_name)[0] + '.obj'))
            
            # Save projection matrix P
            np.savetxt(os.path.join(save_path_proj_matx, output_name), P)
            
            # Save final_projector (projection from head_prior space to pixels)
            # In original code: final_projector = cam_matrix @ np.linalg.inv(P)
            # Where cam_matrix projects BFM->pixels and P transforms BFM->"our space"
            # So final_projector projects "our space"->pixels
            #
            # In our case, K_3x4 = K @ P[:3,:] already projects from head_prior 
            # (which IS "our space") directly to pixels. So K_3x4 IS the final_projector.
            final_projector = K_3x4
            
            np.savetxt(os.path.join(save_path_proj_matx_inv, output_name), final_projector)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    
    parser.add_argument('--root_path', default='./', type=str,
                        help='Root path containing the dataset')
    parser.add_argument('--head_main', default='/localhome/aha220/HairProjects/Im2Haircut/data/head_prior.obj', type=str,
                        help='Path to head prior mesh')
    parser.add_argument('--lmk_indices_path', default='/localhome/aha220/HairProjects/Im2Haircut/data/head_prior_lmk_indices.npy', type=str,
                        help='Path to 478 landmark indices for head_prior.obj')
    parser.add_argument('--save_postfix', default='', type=str,
                        help='Postfix for output directories')
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    main(args)  