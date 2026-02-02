import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt


def crop_to_foreground_square(img, padding_ratio=0.1):
    """
    Crop image to smallest square with foreground having max distance of padding_ratio to borders,
    then resize back to original size.
    """
    original_size = img.size
    
    # Get alpha channel to find foreground
    if img.mode == 'RGBA':
        alpha = img.split()[3]
    else:
        # If no alpha, return original
        return img
    
    # Get bounding box of non-transparent pixels
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    
    left, top, right, bottom = bbox
    fg_width = right - left
    fg_height = bottom - top
    
    # Make it square by using the larger dimension
    fg_size = max(fg_width, fg_height)
    
    # Calculate the center of the foreground
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    # Calculate the required crop size so that foreground has padding_ratio distance to borders
    # foreground should occupy (1 - 2*padding_ratio) of the final crop
    crop_size = int(fg_size / (1 - 2 * padding_ratio))
    
    # Calculate crop coordinates centered on foreground
    crop_left = center_x - crop_size // 2
    crop_top = center_y - crop_size // 2
    crop_right = crop_left + crop_size
    crop_bottom = crop_top + crop_size
    
    # Adjust if crop goes outside image boundaries
    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > img.width:
        crop_left -= (crop_right - img.width)
        crop_right = img.width
    if crop_bottom > img.height:
        crop_top -= (crop_bottom - img.height)
        crop_bottom = img.height
    
    # Ensure coordinates are valid
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(img.width, crop_right)
    crop_bottom = min(img.height, crop_bottom)
    
    # Crop and resize back to original size
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    resized = cropped.resize(original_size, Image.LANCZOS)
    
    return resized


exp_dir = "/localhome/aha220/HairProjects/Im2Haircut/exps_inverse_stage/try/new_data"
output_collage_dir = "./collages"

os.makedirs(output_collage_dir, exist_ok=True)


for sample_id in os.listdir(exp_dir):
    sample_dir = os.path.join(exp_dir, sample_id)
    if not os.path.isdir(sample_dir):
        continue

    sample_name = sample_id.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".webp", "")
    collage_path = os.path.join(output_collage_dir, sample_name)
    os.makedirs(collage_path, exist_ok=True)
    
    original_model_renders_dir = os.path.join(sample_dir, "renders/pred_000000")
    optimized_model_renders_dir = os.path.join(sample_dir, "renders/pred_000300")
    
    images_file_names = ["front.png", "left.png", "right.png"]
    try:
        for renders_dir, output_name in [(original_model_renders_dir, "original.png"), 
                                        (optimized_model_renders_dir, "optimized.png")]:
            
            if not os.path.exists(renders_dir):
                print(f"Renders directory {renders_dir} does not exist, skipping.")
                continue
            
            # skip already processed
            output_file_path = os.path.join(collage_path, output_name)
            if os.path.exists(output_file_path):
                print(f"Collage {output_file_path} already exists, skipping.")
                continue
            
            # Load images
            front_img = Image.open(os.path.join(renders_dir, "front.png")).convert("RGBA")
            left_img = Image.open(os.path.join(renders_dir, "left.png")).convert("RGBA")
            right_img = Image.open(os.path.join(renders_dir, "right.png")).convert("RGBA")
            
            # Crop to foreground square with 0.1 padding ratio
            front_img = crop_to_foreground_square(front_img, padding_ratio=0.1)
            left_img = crop_to_foreground_square(left_img, padding_ratio=0.1)
            right_img = crop_to_foreground_square(right_img, padding_ratio=0.1)
            
            # Resize left and right images to half size (512x512)
            half_size = (512, 512)
            left_img_resized = left_img.resize(half_size, Image.LANCZOS)
            right_img_resized = right_img.resize(half_size, Image.LANCZOS)
            
            # Create collage canvas (1024 wide, 1024 + 512 = 1536 tall)
            collage = Image.new("RGBA", (1024, 1536))
            
            # Paste front image at top
            collage.paste(front_img, (0, 0))
            
            # Paste left and right images below front
            collage.paste(left_img_resized, (0, 1024))
            collage.paste(right_img_resized, (512, 1024))
            
            # Save collage
            collage.save(os.path.join(collage_path, output_name))
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        shutil.rmtree(collage_path)  # Remove incomplete collage directory    