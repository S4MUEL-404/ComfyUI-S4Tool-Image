import numpy as np
from PIL import Image
import torch
from .ImageOverlay import pil2tensor, tensor2pil

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "tile_pattern"
    CATEGORY = "💀S4Tool"

    def tile_pattern(self, image, width, height, spacing, offset, rotation):
        # Convert tensor to PIL image
        src_pil = tensor2pil(image)
        if src_pil.mode != 'RGBA':
            src_pil = src_pil.convert('RGBA')
        src_w, src_h = src_pil.size

        # Pre-rotate source image if needed
        if rotation != 0:
            src_pil = src_pil.rotate(rotation, expand=True, resample=Image.BICUBIC)
            src_w, src_h = src_pil.size

        # Calculate spacing in pixels
        spacing_x = int(src_w * spacing)
        spacing_y = int(src_h * spacing)
        offset_x = int(src_w * offset)

        # Prepare output canvas (transparent)
        pattern = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Calculate tiling grid
        n_cols = int(np.ceil(width / (src_w + spacing_x))) + 2
        n_rows = int(np.ceil(height / (src_h + spacing_y))) + 2

        for row in range(n_rows):
            y = row * (src_h + spacing_y)
            # Offset every other row
            x_offset = offset_x if (row % 2 == 1) else 0
            for col in range(n_cols):
                x = col * (src_w + spacing_x) + x_offset
                # Paste only if inside canvas
                if x < width and y < height:
                    pattern.alpha_composite(src_pil, (x, y))

        # Convert to tensor
        output_tensor = pil2tensor(pattern)
        return (output_tensor,) 
