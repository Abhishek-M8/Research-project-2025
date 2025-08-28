from openslide import OpenSlide
from PIL import Image
import numpy as np
import os
from skimage import measure
from skimage.color import rgb2gray

# Load the slide
slide = OpenSlide(r"C:\Users\Abhishek\Desktop\Research_Project_2025\Histo_images\EH-15740-040.svs")
print("Slide dimensions:", slide.dimensions)

# Get dimensions and levels
dims = slide.level_dimensions
num_levels = len(dims)
print("Number of levels in image:", num_levels)
print("Dimensions of various levels in image:", dims)

level = 0  # Highest resolution
width, height = slide.level_dimensions[level]
tile_size = 2048

def is_mostly_white_or_empty(img, white_threshold=0.8, luminance_threshold=200, min_tissue_entropy=2.0):
    # Downsample for efficiency
    img_small = img.resize((256, 256), Image.Resampling.LANCZOS)
    arr = np.array(img_small)
    
    # Convert to grayscale
    gray = rgb2gray(arr) * 255
    gray = gray.astype(np.uint8)
    
    # Calculate white pixel ratio
    white_pixels = np.sum(gray > luminance_threshold)
    total_pixels = gray.size
    white_ratio = white_pixels / total_pixels
    
    # Calculate entropy
    entropy = measure.shannon_entropy(gray)
    
    # Consider tile mostly white/empty if high white ratio and low entropy
    return white_ratio > white_threshold and entropy < min_tissue_entropy

output_dir = r"C:\Users\Abhishek\Desktop\Research_Project_2025\openslide-2048\15740-040-NEW"
os.makedirs(output_dir, exist_ok=True)

for y in range(0, height - tile_size + 1, tile_size):
    for x in range(0, width - tile_size + 1, tile_size):
        # Ensure the tile fits within the image boundaries
        if x + tile_size <= width and y + tile_size <= height:
            tile = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            
            if not is_mostly_white_or_empty(tile, white_threshold=0.8, luminance_threshold=200, min_tissue_entropy=2.0):
                tile.save(f"{output_dir}/tile_{x}_{y}.png")
                print(f"Saved tile at ({x}, {y})")
            else:
                print(f"Skipped white/empty tile at ({x}, {y})")

slide.close()
