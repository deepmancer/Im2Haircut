#!/usr/bin/env python3
"""
Render Im2Haircut hair reconstruction outputs from multiple camera views.

This script renders hair strands from Im2Haircut prediction files (PLY format)
along with a head mesh, from four orbital camera views (front, right, left, back).

It automatically processes all subjects in the experiments directory.

Usage:
    blender --background --python render_im2haircut.py
    blender --background --python render_im2haircut.py -- --experiments_dir <path>
    
Example:
    # Render all subjects with defaults
    blender --background --python render_im2haircut.py
    
    # Render with custom settings
    blender --background --python render_im2haircut.py -- \\
        --experiments_dir /path/to/experiments \\
        --resolution 512 --samples 64
        
Output Structure:
    experiments_dir/
    ├── subject1/
    │   ├── pointclouds_train/
    │   │   ├── pred_000000.ply
    │   │   └── ...
    │   └── renders/           <-- Created by this script
    │       ├── pred_000000/
    │       │   ├── front.png
    │       │   ├── back.png
    │       │   ├── left.png
    │       │   └── right.png
    │       └── ...
    └── subject2/
        └── ...
"""

import bpy
import os
import sys
import math
import argparse
import glob
import numpy as np
from mathutils import Vector, Matrix
from typing import List, Tuple, Dict, Optional

import random
import time
# ============================================================================
# Hair strand loading from PLY
# ============================================================================

def read_im2haircut_ply(file_path: str, num_points: int = 100, blender_format: bool = True) -> List[List[Vector]]:
    """
    Read hair strand data from Im2Haircut PLY file format.
    
    Im2Haircut saves strands as a point cloud using trimesh.PointCloud.
    The vertices are stored as a flat array where every `num_points` consecutive 
    vertices form one hair strand.
    
    Args:
        file_path: Path to the PLY file
        num_points: Number of points per strand (default: 100, matching Im2Haircut)
        blender_format: If True, convert to Blender coordinate system (x, -z, y)
        
    Returns:
        List of strands, where each strand is a list of Vector points
    """
    import trimesh
    
    # Load the point cloud using trimesh (same library used to save it)
    point_cloud = trimesh.load(file_path)
    
    # Get vertices as numpy array
    if hasattr(point_cloud, 'vertices'):
        vertices = point_cloud.vertices
    else:
        # Fallback for PointCloud objects
        vertices = np.array(point_cloud.vertices)
    
    total_points = len(vertices)
    num_strands = total_points // num_points
    
    print(f"Reading {total_points} vertices ({num_strands} strands × {num_points} points) from {os.path.basename(file_path)}")
    
    if total_points % num_points != 0:
        print(f"  WARNING: Total points ({total_points}) is not divisible by num_points ({num_points})")
        print(f"  Using {num_strands} complete strands, ignoring {total_points % num_points} extra points")
    
    # Reshape into strands
    strands = []
    for i in range(num_strands):
        start_idx = i * num_points
        end_idx = start_idx + num_points
        strand_vertices = vertices[start_idx:end_idx]
        
        # Convert to Blender Vectors
        strand_points = []
        for v in strand_vertices:
            x, y, z = v[0], v[1], v[2]
            if blender_format:
                # Convert to Blender's coordinate system (Y-up to Z-up)
                strand_points.append(Vector((x, -z, y)))
            else:
                strand_points.append(Vector((x, y, z)))
        
        strands.append(strand_points)
    
    print(f"  Loaded {len(strands)} strands")
    return strands


# ============================================================================
# Scene setup utilities
# ============================================================================

def reset_scene():
    """Clear the Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear orphan data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for img in bpy.data.images:
        if img.users == 0:
            bpy.data.images.remove(img)


def import_head_mesh(mesh_path: str) -> bpy.types.Object:
    """Import the head mesh from OBJ file."""
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Head mesh not found: {mesh_path}")
    
    bpy.ops.wm.obj_import(filepath=mesh_path)
    head_obj = bpy.context.selected_objects[0]
    head_obj.name = "HeadMesh"
    
    # Apply smooth shading
    bpy.context.view_layer.objects.active = head_obj
    bpy.ops.object.shade_smooth()
    
    return head_obj


def create_hair_curves(strands: List[List[Vector]], radius: float = 0.0003) -> bpy.types.Object:
    """
    Create hair strands as Blender Curves object.
    
    Args:
        strands: List of strands, each is a list of Vector points
        radius: Radius of each hair strand
        
    Returns:
        Blender curves object
    """
    curves_data = bpy.data.curves.new(name="HairCurvesData", type='CURVE')
    curves_data.dimensions = '3D'
    curves_data.resolution_u = 2
    curves_data.bevel_depth = radius
    curves_data.bevel_resolution = 2
    
    for strand in strands:
        if len(strand) < 2:
            continue
        
        spline = curves_data.splines.new('POLY')
        spline.points.add(len(strand) - 1)
        
        for i, point in enumerate(strand):
            spline.points[i].co = (point.x, point.y, point.z, 1.0)
    
    curves_obj = bpy.data.objects.new("HairCurves", curves_data)
    bpy.context.collection.objects.link(curves_obj)
    
    return curves_obj


def create_hair_material() -> bpy.types.Material:
    """Create a basic hair material using Principled Hair BSDF."""
    mat = bpy.data.materials.new(name="HairMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    hair_bsdf = nodes.new('ShaderNodeBsdfHairPrincipled')
    
    output.location = (300, 0)
    hair_bsdf.location = (0, 0)
    
    # Configure hair BSDF
    hair_bsdf.parametrization = 'MELANIN'
    hair_bsdf.inputs['Melanin'].default_value = 0.6
    hair_bsdf.inputs['Melanin Redness'].default_value = 0.5
    hair_bsdf.inputs['Roughness'].default_value = 0.3
    hair_bsdf.inputs['Radial Roughness'].default_value = 0.4
    hair_bsdf.inputs['Coat'].default_value = 0.0
    hair_bsdf.inputs['IOR'].default_value = 1.55
    
    links.new(hair_bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def create_head_material() -> bpy.types.Material:
    """Create a completely matte, non-reflective material for the head mesh."""
    mat = bpy.data.materials.new(name="HeadMaterial")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create a simple diffuse-only setup for completely matte appearance
    output = nodes.new('ShaderNodeOutputMaterial')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    
    output.location = (300, 0)
    diffuse.location = (0, 0)
    
    # Muted gray-ish skin tone (darker to avoid brightness)
    diffuse.inputs['Color'].default_value = (0.45, 0.35, 0.32, 1.0)
    diffuse.inputs['Roughness'].default_value = 1.0  # Maximum roughness
    
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


# ============================================================================
# Camera setup for orbital views
# ============================================================================

def get_orbital_camera_configs(
    center: Tuple[float, float, float] = (0.0, 0.0, 1.68),
    distance: float = 1.1,
    focal_length: float = 50.0,
) -> Dict[str, Dict]:
    """
    Get camera configurations for four orbital views around the subject.
    
    Args:
        center: Center point to orbit around (x, y, z)
        distance: Distance from center to camera
        focal_length: Camera focal length
        
    Returns:
        Dictionary mapping view names to camera configurations
    """
    cx, cy, cz = center
    
    # In Blender coordinate system after Im2Haircut conversion (x, -z, y):
    # The head faces along the Y axis, so we orbit in the XY plane
    # Front/back move along Y, left/right move along X
    # 
    # Using proper orbital angles around Z axis:
    # 0° = front (-Y), 90° = left (-X), 180° = back (+Y), 270° = right (+X)
    
    configs = {
        'front': {
            'location': (cx, cy - distance, cz),
            'rotation': (math.radians(90), 0, 0),
            'focal_length': focal_length,
        },
        'front_left': {
            'location': (cx + distance * 0.707, cy - distance * 0.707, cz),
            'rotation': (math.radians(90), 0, math.radians(45)),
            'focal_length': focal_length,
        },
        'front_right': {
            'location': (cx - distance * 0.707, cy - distance * 0.707, cz),
            'rotation': (math.radians(90), 0, math.radians(-45)),
            'focal_length': focal_length,
        },
        'back': {
            'location': (cx, cy + distance, cz),
            'rotation': (math.radians(90), 0, math.radians(180)),
            'focal_length': focal_length,
        },
        'right': {
            'location': (cx - distance, cy, cz),
            'rotation': (math.radians(90), 0, math.radians(-90)),  # Look toward +X
            'focal_length': focal_length,
        },
        'left': {
            'location': (cx + distance, cy, cz),
            'rotation': (math.radians(90), 0, math.radians(90)),  # Look toward -X
            'focal_length': focal_length,
        },
    }
    
    return configs


def create_camera(config: Dict) -> bpy.types.Object:
    """Create a camera with the given configuration."""
    bpy.ops.object.camera_add(
        location=config['location'],
        rotation=config['rotation']
    )
    camera = bpy.context.active_object
    camera.data.lens = config['focal_length']
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera
    return camera


# ============================================================================
# Lighting setup
# ============================================================================

def setup_lighting():
    """Setup basic three-point lighting."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(1.0, -1.0, 1.5))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = 75
    key_light.data.size = 4.0
    key_light.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-1.0, -0.5, 1.0))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 50
    fill_light.data.size = 2.5
    fill_light.rotation_euler = (math.radians(60), 0, math.radians(-30))
    
    # Rim light
    bpy.ops.object.light_add(type='AREA', location=(0, 1.0, 1.5))
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    rim_light.data.energy = 75
    rim_light.data.size = 2.0
    rim_light.rotation_euler = (math.radians(45), 0, math.radians(180))
    
    return [key_light, fill_light, rim_light]


# ============================================================================
# Render settings
# ============================================================================

def enable_gpu_rendering() -> List[str]:
    """
    Activate GPU devices for Cycles rendering with automatic backend detection.
    Tries OPTIX first (best for denoising), then CUDA, then other backends.
    
    Returns:
        List of activated device names
    """
    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get('cycles')
    if not cycles_addon:
        print("Cycles addon not found, using CPU rendering")
        return []
    
    cycles_prefs = cycles_addon.preferences
    
    # Backend priority - OptiX is preferred for its superior denoising
    backend_priority = ['OPTIX', 'CUDA', 'HIPRT', 'HIP', 'ONEAPI', 'METAL']
    
    for backend in backend_priority:
        try:
            cycles_prefs.compute_device_type = backend
        except (TypeError, AttributeError):
            continue
        
        # Refresh devices
        if hasattr(cycles_prefs, 'get_devices_for_type'):
            try:
                cycles_prefs.get_devices_for_type(backend)
            except RuntimeError:
                continue
        elif hasattr(cycles_prefs, 'get_devices'):
            cycles_prefs.get_devices()
        
        if hasattr(cycles_prefs, 'refresh_devices'):
            cycles_prefs.refresh_devices()
        
        # Enable GPU devices, disable CPU
        gpu_types = {'GPU', 'CUDA', 'OPTIX', 'HIP', 'HIPRT', 'METAL', 'ONEAPI'}
        activated = []
        
        for device in getattr(cycles_prefs, 'devices', []):
            device_type = str(getattr(device, 'type', '')).upper()
            if device_type in gpu_types:
                try:
                    device.use = True
                    if getattr(device, 'use', False):
                        device_name = str(getattr(device, 'name', backend))
                        activated.append(device_name)
                except AttributeError:
                    pass
            elif device_type == 'CPU':
                try:
                    device.use = False
                except AttributeError:
                    pass
        
        if activated:
            bpy.context.scene.cycles.device = 'GPU'
            print(f"Activated Cycles backend: {backend} with devices: {activated}")
            return activated
    
    # Fallback to CPU
    print("No GPU backend available, falling back to CPU rendering")
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.render.threads_mode = 'AUTO'
    return []


def setup_render_settings(
    resolution: int = 1024,
    samples: int = 256,
    use_adaptive_sampling: bool = True,
    adaptive_threshold: float = 0.0025,
    min_samples: int = 128,
):
    """
    Configure high-quality render settings for Cycles hair rendering.
    
    Args:
        resolution: Output image resolution (square)
        samples: Maximum number of render samples
        use_adaptive_sampling: Enable adaptive sampling for efficiency
        adaptive_threshold: Noise threshold for adaptive sampling (lower = higher quality)
        min_samples: Minimum samples before adaptive sampling kicks in
    """
    scene = bpy.context.scene
    
    # -------------------------------------------------------------------------
    # Basic render settings
    # -------------------------------------------------------------------------
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True  # Faster re-renders
    
    # Output settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    # scene.render.image_settings.color_depth = '16'  # Higher bit depth
    # scene.render.image_settings.compression = 15
    
    # -------------------------------------------------------------------------
    # GPU rendering
    # -------------------------------------------------------------------------
    activated_devices = enable_gpu_rendering()
    
    # -------------------------------------------------------------------------
    # Sampling settings
    # -------------------------------------------------------------------------
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    if use_adaptive_sampling:
        scene.cycles.adaptive_threshold = adaptive_threshold
        scene.cycles.adaptive_min_samples = min_samples
        if hasattr(scene.cycles, 'adaptive_max_samples'):
            scene.cycles.adaptive_max_samples = min_samples * 4
    
    # -------------------------------------------------------------------------
    # Bounce settings - important for realistic hair rendering
    # Hair needs transmission and glossy bounces for light to pass through
    # -------------------------------------------------------------------------
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6      # Important for hair specular highlights
    scene.cycles.transmission_bounces = 8  # Important for light passing through hair
    scene.cycles.volume_bounces = 2
    scene.cycles.transparent_max_bounces = 8
    
    # -------------------------------------------------------------------------
    # Denoising - OptiX denoiser with albedo/normal for best quality
    # -------------------------------------------------------------------------
    scene.cycles.use_denoising = True
    
    # Try OptiX denoiser first (best quality), fall back to OpenImageDenoise
    try:
        scene.cycles.denoiser = 'OPTIX'
    except TypeError:
        try:
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        except TypeError:
            pass
    
    # Use albedo and normal passes for better denoising
    if hasattr(scene.cycles, 'denoising_input_passes'):
        try:
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        except TypeError:
            pass
    
    # -------------------------------------------------------------------------
    # Hair/Curves rendering optimization
    # -------------------------------------------------------------------------
    if hasattr(scene.cycles, 'use_curves'):
        scene.cycles.use_curves = True
    if hasattr(scene.cycles, 'curves_use_camera_cull'):
        scene.cycles.curves_use_camera_cull = True
    if hasattr(scene.cycles, 'curves_use_backfacing_cull'):
        scene.cycles.curves_use_backfacing_cull = True
    
    # -------------------------------------------------------------------------
    # Performance optimizations
    # -------------------------------------------------------------------------
    if hasattr(scene.cycles, 'use_light_tree'):
        scene.cycles.use_light_tree = True
    
    # -------------------------------------------------------------------------
    # Color management for accurate colors
    # -------------------------------------------------------------------------
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    
    print(f"Render settings configured: {resolution}x{resolution}, {samples} samples")


# ============================================================================
# Main rendering function
# ============================================================================

def render_prediction(
    ply_path: str,
    head_mesh_path: str,
    output_dir: str,
    resolution: int = 1024,
    samples: int = 128,
    hair_radius: float = 0.0003,
    num_points: int = 100,
):
    """
    Render a single prediction PLY file from four views.
    
    Args:
        ply_path: Path to the prediction PLY file
        head_mesh_path: Path to the head mesh OBJ file
        output_dir: Directory to save rendered images
        resolution: Render resolution (square)
        samples: Number of Cycles samples
        hair_radius: Hair strand radius
        num_points: Number of points per strand in PLY file
    """
    pred_name = os.path.splitext(os.path.basename(ply_path))[0]
    pred_output_dir = os.path.join(output_dir, pred_name)
    os.makedirs(pred_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Rendering: {pred_name}")
    print(f"{'='*60}")
    
    # Reset scene
    reset_scene()
    
    # Load hair strands
    print("Loading hair strands...")
    strands = read_im2haircut_ply(ply_path, num_points=num_points)
    if not strands:
        print(f"  WARNING: No valid strands found in {ply_path}")
        return
    
    # Create hair object
    print("Creating hair curves...")
    hair_obj = create_hair_curves(strands, radius=hair_radius)
    hair_mat = create_hair_material()
    hair_obj.data.materials.append(hair_mat)
    
    # Import head mesh
    print("Importing head mesh...")
    head_obj = import_head_mesh(head_mesh_path)
    head_mat = create_head_material()
    if head_obj.data.materials:
        head_obj.data.materials[0] = head_mat
    else:
        head_obj.data.materials.append(head_mat)
    
    # Setup lighting
    print("Setting up lighting...")
    setup_lighting()
    
    # Setup render settings
    setup_render_settings(resolution=resolution, samples=samples)
    
    # Get camera configurations
    camera_configs = get_orbital_camera_configs()
    
    # Render each view
    for view_name, cam_config in camera_configs.items():
        print(f"  Rendering {view_name} view...")
        
        # Create camera
        camera = create_camera(cam_config)
        
        # Set output path
        output_path = os.path.join(pred_output_dir, f"{view_name}.png")
        bpy.context.scene.render.filepath = output_path
        
        # Render
        bpy.ops.render.render(write_still=True)
        print(f"    Saved: {output_path}")
        
        # Remove camera for next view
        bpy.data.objects.remove(camera, do_unlink=True)
    
    print(f"Completed: {pred_name}")


def render_subject(
    subject_dir: str,
    head_mesh_path: str,
    resolution: int = 1024,
    samples: int = 128,
    hair_radius: float = 0.0003,
    num_points: int = 100,
) -> bool:
    """
    Render all predictions for a single subject.
    
    Args:
        subject_dir: Path to subject directory containing pointclouds_train/
        head_mesh_path: Path to the head mesh OBJ file
        resolution: Render resolution
        samples: Number of Cycles samples
        hair_radius: Hair strand radius
        num_points: Number of points per strand in PLY file
        
    Returns:
        True if rendering was successful, False otherwise
    """
    pointclouds_dir = os.path.join(subject_dir, "pointclouds_train")
    renders_dir = os.path.join(subject_dir, "renders")
    
    if not os.path.isdir(pointclouds_dir):
        print(f"  [SKIP] pointclouds_train not found: {pointclouds_dir}")
        return False
    
    # Find all prediction files
    ply_files = sorted(glob.glob(os.path.join(pointclouds_dir, "pred_*.ply")))
    
    if not ply_files:
        print(f"  [SKIP] No prediction PLY files found in {pointclouds_dir}")
        return False
    
    subject_name = os.path.basename(subject_dir)
    print(f"\n{'#'*60}")
    print(f"# Subject: {subject_name}")
    print(f"{'#'*60}")
    print(f"Directory: {subject_dir}")
    print(f"Head mesh: {head_mesh_path}")
    print(f"Output: {renders_dir}")
    print(f"Predictions: {len(ply_files)}")
    for f in ply_files:
        print(f"  - {os.path.basename(f)}")
    print(f"{'#'*60}\n")
    
    os.makedirs(renders_dir, exist_ok=True)
    
    for ply_path in ply_files:
        render_prediction(
            ply_path=ply_path,
            head_mesh_path=head_mesh_path,
            output_dir=renders_dir,
            resolution=resolution,
            samples=samples,
            hair_radius=hair_radius,
            num_points=num_points,
        )
    
    print(f"Completed subject: {subject_name}")
    return True


def render_all_subjects(
    experiments_dir: str,
    head_mesh_path: str,
    resolution: int = 1024,
    samples: int = 128,
    hair_radius: float = 0.0002,
    num_points: int = 100,
):
    """
    Render all subjects in the experiments directory.
    
    Args:
        experiments_dir: Path to directory containing subject folders
        head_mesh_path: Path to the head mesh OBJ file
        resolution: Render resolution
        samples: Number of Cycles samples
        hair_radius: Hair strand radius
        num_points: Number of points per strand in PLY file
    """
    if not os.path.isdir(experiments_dir):
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    # Find all subject directories
    subjects = sorted([
        d for d in os.listdir(experiments_dir)
        if os.path.isdir(os.path.join(experiments_dir, d))
    ])
    
    if not subjects:
        print(f"No subject directories found in {experiments_dir}")
        return
    
    print(f"\n{'='*60}")
    print("Im2Haircut Batch Rendering")
    print(f"{'='*60}")
    print(f"Experiments dir: {experiments_dir}")
    print(f"Head mesh: {head_mesh_path}")
    print(f"Resolution: {resolution}")
    print(f"Samples: {samples}")
    print(f"Hair radius: {hair_radius}")
    print(f"Subjects found: {len(subjects)}")
    for i, s in enumerate(subjects):
        print(f"  [{i+1}] {s}")
    print(f"{'='*60}\n")
    
    
    # Process each subject
    successful = 0
    skipped = 0
    
    
    random.seed(int(time.time()))
    random.shuffle(subjects)
    for idx, subject in enumerate(subjects):
        print(f"\n[{idx+1}/{len(subjects)}] Processing: {subject}")
        
        subject_dir = os.path.join(experiments_dir, subject)
        
        # Check if already rendered
        renders_dir = os.path.join(subject_dir, "renders")
        if os.path.isdir(renders_dir):
            # Check if all views are rendered for any prediction
            existing_renders = glob.glob(os.path.join(renders_dir, "pred_*", "front.png"))
            if existing_renders:
                print(f"  [SKIP] Already rendered - found {len(existing_renders)} predictions")
                skipped += 1
                continue
        
        try:
            result = render_subject(
                subject_dir=subject_dir,
                head_mesh_path=head_mesh_path,
                resolution=resolution,
                samples=samples,
                hair_radius=hair_radius,
                num_points=num_points,
            )
            if result:
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [ERROR] Failed to render {subject}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    print(f"\n{'='*60}")
    print("Batch Rendering Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(subjects)}")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    # Find the '--' separator
    try:
        idx = sys.argv.index('--')
        args = sys.argv[idx + 1:]
    except ValueError:
        args = []
    
    parser = argparse.ArgumentParser(
        description="Render Im2Haircut hair reconstruction outputs for all subjects"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="/localhome/aha220/HairProjects/Im2Haircut/exps_inverse_stage/try/new_data",
        help="Path to experiments directory containing subject folders"
    )
    parser.add_argument(
        "--head_mesh",
        type=str,
        default="/localhome/aha220/HairProjects/Im2Haircut/data/head_prior.obj",
        help="Path to head mesh OBJ file"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Render resolution (default: 512)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1024,
        help="Number of Cycles samples (default: 256)"
    )
    parser.add_argument(
        "--hair_radius",
        type=float,
        default=0.00025,
        help="Hair strand radius (default: 0.00025)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=100,
        help="Number of points per strand (default: 100, matching Im2Haircut)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-render even if renders already exist"
    )
    
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    
    render_all_subjects(
        experiments_dir=args.experiments_dir,
        head_mesh_path=args.head_mesh,
        resolution=args.resolution,
        samples=args.samples,
        hair_radius=args.hair_radius,
        num_points=args.num_points,
    )
