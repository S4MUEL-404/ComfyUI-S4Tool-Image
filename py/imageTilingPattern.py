import random as _random

import numpy as np
import torch
from PIL import Image

from ..nodes_config import pil2tensor, tensor2pil, ImageUtils
from ..dependency_manager import S4ToolLogger

class ImageTilingPattern:
    """
    Tile the input image as a pattern on a transparent background, supporting spacing, offset, and rotation. Output size is set by width and height parameters.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "spacing": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "randomize_scale": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "slider"}),
                "randomize_rotation": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "slider"}),
                "randomize_opacity": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "slider"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "tile_pattern"
    CATEGORY = "💀S4Tool"

    def tile_pattern(self, image, width, height, spacing, offset, rotation, scale, opacity, randomize_scale=0, randomize_rotation=0, randomize_opacity=0, image2=None, image3=None, image4=None):
        # Use PIL implementation directly
        return self._tile_pattern_pil(image, width, height, spacing, offset, rotation, scale, opacity, randomize_scale, randomize_rotation, randomize_opacity, image2, image3, image4)
    
    
    def _tile_pattern_pil(self, image, width, height, spacing, offset, rotation, scale, opacity, randomize_scale, randomize_rotation, randomize_opacity, image2, image3, image4):
        """Highly optimized PIL implementation with aggressive performance improvements."""
        S4ToolLogger.info("ImageTilingPattern", f"Starting tile pattern generation: {width}x{height}")
        
        # Check for simple static case - if no randomization (rotation is OK for static)
        is_static = (randomize_scale == 0 and randomize_rotation == 0 and randomize_opacity == 0)
        
        # Collect images in sequence order
        images = [img for img in [image, image2, image3, image4] if img is not None]
        if len(images) == 0:
            images = [image]

        # Convert tensors to PIL RGBA with caching
        pil_images = []
        for img in images:
            pil_img = tensor2pil(img)
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            pil_images.append(pil_img)
        
        # Pre-compute base transformations to avoid repetition
        base_img = pil_images[0]
        orig_w, orig_h = base_img.size
        base_w = max(1, int(round(orig_w * max(0.01, scale))))
        base_h = max(1, int(round(orig_h * max(0.01, scale))))
        
        # Calculate spacing and offset in pixels
        spacing_x = int(base_w * spacing)
        spacing_y = int(base_h * spacing)
        offset_x = int(base_w * offset)

        # Guard against zero/negative step
        step_x = max(1, base_w + spacing_x)
        step_y = max(1, base_h + spacing_y)

        # Prepare output canvas (transparent)
        pattern = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Calculate tiling bounds to ensure coverage from (0,0)
        # Start from negative indices to cover the canvas from top-left
        start_col = int((-offset_x - base_w) / step_x) - 1
        end_col = int(np.ceil((width + base_w + offset_x) / step_x)) + 1
        start_row = -1  # Always start one row before to ensure coverage
        end_row = int(np.ceil((height + base_h) / step_y)) + 1
        
        total_tiles = (end_row - start_row) * (end_col - start_col)
        S4ToolLogger.info("ImageTilingPattern", f"Tiling grid: rows {start_row}-{end_row}, cols {start_col}-{end_col}, total: {total_tiles} tiles")
        
        # Early exit for unreasonably large tile counts
        if total_tiles > 50000:  # Configurable limit
            S4ToolLogger.info("ImageTilingPattern", f"WARNING: Tile count ({total_tiles}) exceeds recommended limit, reducing quality...")
            # Reduce tile density by increasing step size
            step_x = int(step_x * 1.5)
            step_y = int(step_y * 1.5)
            # Recalculate bounds with new step sizes
            start_col = int((-offset_x - base_w) / step_x) - 1
            end_col = int(np.ceil((width + base_w + offset_x) / step_x)) + 1
            start_row = -1  # Always start one row before to ensure coverage
            end_row = int(np.ceil((height + base_h) / step_y)) + 1
            total_tiles = (end_row - start_row) * (end_col - start_col)
            S4ToolLogger.info("ImageTilingPattern", f"Reduced to {total_tiles} tiles for performance")
        
        # Create transformation cache for common operations
        transform_cache = {}
        
        # Pre-transform base images for static cases
        if is_static:
            static_images = []
            for src_pil in pil_images:
                transformed = src_pil
                
                # Apply scaling if needed
                if base_w != orig_w or base_h != orig_h:
                    transformed = transformed.resize((base_w, base_h), Image.BICUBIC)
                
                # Apply rotation if needed - keep original size
                if abs(rotation) > 0.1:
                    orig_w_t, orig_h_t = transformed.size
                    
                    # Create a larger canvas for rotation to avoid clipping
                    diagonal = int(np.sqrt(orig_w_t**2 + orig_h_t**2)) + 2
                    canvas_size = max(diagonal, orig_w_t + 20, orig_h_t + 20)
                    
                    # Create transparent canvas and paste tile at center
                    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
                    paste_x = (canvas_size - orig_w_t) // 2
                    paste_y = (canvas_size - orig_h_t) // 2
                    canvas.paste(transformed, (paste_x, paste_y), transformed)
                    
                    # Rotate the canvas
                    rotated_canvas = canvas.rotate(rotation, resample=Image.BICUBIC)
                    
                    # Crop back to original size from center
                    crop_x = (rotated_canvas.width - orig_w_t) // 2
                    crop_y = (rotated_canvas.height - orig_h_t) // 2
                    transformed = rotated_canvas.crop((
                        crop_x, crop_y, 
                        crop_x + orig_w_t, crop_y + orig_h_t
                    ))
                
                static_images.append(transformed)
            pil_images = static_images
        
        processed_tiles = 0
        
        tile_index = 0
        
        # Skip factor for very large grids - process every nth tile for performance
        skip_factor = 1
        if total_tiles > 10000:
            skip_factor = 2
        elif total_tiles > 25000:
            skip_factor = 3
            
        # Progress reporting frequency
        progress_interval = min(1000, max(100, total_tiles // 20))
        
        for row in range(start_row, end_row, skip_factor):
            y = row * step_y
            row_offset_x = offset_x if (row % 2 == 1) else 0
            
            for col in range(start_col, end_col, skip_factor):
                x = col * step_x + row_offset_x
                
                # Aggressive culling: skip tiles completely outside canvas
                if (x + base_w < 0 or y + base_h < 0 or 
                    x >= width or y >= height):
                    tile_index += 1
                    processed_tiles += 1
                    continue

                # Select image in round-robin order
                img_idx = tile_index % len(pil_images)
                src_pil = pil_images[img_idx]

                if is_static:
                    # Static case - all transformations already applied in pre-processing
                    tile_img = src_pil
                    # Apply final opacity if needed (not applied in pre-processing for caching)
                    if opacity < 0.99:
                        r, g, b, a = tile_img.split()
                        a = a.point(lambda v: int(v * opacity))
                        tile_img = Image.merge('RGBA', (r, g, b, a))
                else:
                    # Dynamic case with randomization
                    tile_scale = scale
                    tile_rotation = rotation
                    tile_opacity = opacity
                    
                    # Calculate randomization only when needed
                    if randomize_scale > 0:
                        scale_amp = randomize_scale / 100.0
                        jitter = _random.uniform(-scale_amp, scale_amp)
                        tile_scale = max(0.01, min(10.0, scale * (1.0 + jitter)))

                    if randomize_rotation > 0:
                        rotation_amp = randomize_rotation / 100.0
                        jitter = _random.uniform(-100.0 * rotation_amp, 100.0 * rotation_amp)
                        tile_rotation = rotation + jitter

                    if randomize_opacity > 0:
                        opacity_amp = randomize_opacity / 100.0
                        jitter = _random.uniform(-opacity_amp, opacity_amp)
                        tile_opacity = max(0.01, min(1.0, opacity + jitter))

                    # Create cache key only for non-randomized transformations
                    use_cache = (randomize_scale == 0 and randomize_rotation == 0 and randomize_opacity == 0)
                    cache_key = (img_idx, tile_scale, tile_rotation, tile_opacity) if use_cache else None
                    
                    if use_cache and cache_key in transform_cache:
                        tile_img = transform_cache[cache_key]
                    else:
                        # Apply transformations efficiently
                        tile_img = src_pil
                        
                        # Scale if needed
                        if abs(tile_scale - 1.0) > 0.01:
                            tw = max(1, int(round(src_pil.width * tile_scale)))
                            th = max(1, int(round(src_pil.height * tile_scale)))
                            tile_img = tile_img.resize((tw, th), Image.BICUBIC)
                        
                        # Rotate if needed - keep original size for alignment
                        if abs(tile_rotation) > 0.1:
                            # Get original dimensions before rotation
                            orig_tile_w, orig_tile_h = tile_img.size
                            
                            # Create a larger canvas for rotation to avoid clipping
                            diagonal = int(np.sqrt(orig_tile_w**2 + orig_tile_h**2)) + 2
                            canvas_size = max(diagonal, orig_tile_w + 20, orig_tile_h + 20)
                            
                            # Create transparent canvas and paste tile at center
                            canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
                            paste_x = (canvas_size - orig_tile_w) // 2
                            paste_y = (canvas_size - orig_tile_h) // 2
                            canvas.paste(tile_img, (paste_x, paste_y), tile_img)
                            
                            # Rotate the canvas
                            rotated_canvas = canvas.rotate(tile_rotation, resample=Image.BICUBIC)
                            
                            # Crop back to original size from center
                            crop_x = (rotated_canvas.width - orig_tile_w) // 2
                            crop_y = (rotated_canvas.height - orig_tile_h) // 2
                            tile_img = rotated_canvas.crop((
                                crop_x, crop_y, 
                                crop_x + orig_tile_w, crop_y + orig_tile_h
                            ))
                        
                        # Apply opacity if needed
                        if tile_opacity < 0.99:
                            r, g, b, a = tile_img.split()
                            a = a.point(lambda v: int(v * tile_opacity))
                            tile_img = Image.merge('RGBA', (r, g, b, a))
                        
                        # Cache the result
                        if use_cache:
                            transform_cache[cache_key] = tile_img

                # Paste with bounds checking - adjust position for center alignment
                if tile_img.width > 0 and tile_img.height > 0:
                    # Calculate center offset for scaled tiles
                    center_offset_x = (base_w - tile_img.width) // 2
                    center_offset_y = (base_h - tile_img.height) // 2
                    
                    final_x = x + center_offset_x
                    final_y = y + center_offset_y
                    
                    try:
                        pattern.alpha_composite(tile_img, (final_x, final_y))
                    except Exception:
                        # Skip problematic tiles to avoid crashes
                        pass

                tile_index += 1
                processed_tiles += 1
                
                # Less frequent progress feedback
                if processed_tiles % progress_interval == 0:
                    progress = (processed_tiles / total_tiles) * 100
                    S4ToolLogger.info("ImageTilingPattern", f"Progress: {progress:.1f}% ({processed_tiles}/{total_tiles} tiles)")
        
        S4ToolLogger.info("ImageTilingPattern", f"Completed tiling pattern generation with {processed_tiles} tiles")

        output_tensor = pil2tensor(pattern)
        return (output_tensor,)