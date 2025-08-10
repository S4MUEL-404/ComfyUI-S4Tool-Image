import torch
import numpy as np
from PIL import Image, ImageEnhance
from ..nodes_config import pil2tensor, tensor2pil

class ImageAdjustment:
    """
    A node that adjusts brightness, contrast, and saturation of an image.
    Supports images with transparency (RGBA) and maintains alpha channel.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "contrast": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "saturation": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    FUNCTION = "adjust_image"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def adjust_image(self, image, brightness, contrast, saturation):
        """
        Adjust brightness, contrast, and saturation of the input image.
        Preserves alpha channel if present.
        """
        # Convert tensor to PIL image
        pil_image = tensor2pil(image)
        
        # Check if image has alpha channel
        has_alpha = pil_image.mode in ['RGBA', 'LA']
        original_mode = pil_image.mode
        
        # If image has alpha, separate it for processing
        if has_alpha:
            if pil_image.mode == 'RGBA':
                rgb_image = pil_image.convert('RGB')
                alpha_channel = pil_image.split()[3]  # Extract alpha channel
            else:  # LA mode
                rgb_image = pil_image.convert('RGB')
                alpha_channel = pil_image.split()[1]  # Extract alpha channel
        else:
            rgb_image = pil_image.convert('RGB')
            alpha_channel = None
        
        # Apply adjustments to RGB channels only
        adjusted_image = rgb_image
        
        # Adjust brightness (range: -100 to 100, where 0 is no change)
        if brightness != 0:
            brightness_factor = 1.0 + (brightness / 100.0)
            brightness_factor = max(0.0, brightness_factor)  # Ensure non-negative
            enhancer = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = enhancer.enhance(brightness_factor)
        
        # Adjust contrast (range: -100 to 100, where 0 is no change)
        if contrast != 0:
            contrast_factor = 1.0 + (contrast / 100.0)
            contrast_factor = max(0.0, contrast_factor)  # Ensure non-negative
            enhancer = ImageEnhance.Contrast(adjusted_image)
            adjusted_image = enhancer.enhance(contrast_factor)
        
        # Adjust saturation (range: -100 to 100, where 0 is no change)
        if saturation != 0:
            saturation_factor = 1.0 + (saturation / 100.0)
            saturation_factor = max(0.0, saturation_factor)  # Ensure non-negative
            enhancer = ImageEnhance.Color(adjusted_image)
            adjusted_image = enhancer.enhance(saturation_factor)
        
        # Restore alpha channel if original image had one
        if has_alpha and alpha_channel is not None:
            # Convert back to RGBA and restore alpha channel
            if original_mode == 'RGBA':
                adjusted_image = adjusted_image.convert('RGBA')
                # Replace alpha channel with original
                r, g, b, _ = adjusted_image.split()
                adjusted_image = Image.merge('RGBA', (r, g, b, alpha_channel))
            elif original_mode == 'LA':
                # Convert to LA mode
                adjusted_image = adjusted_image.convert('L')
                adjusted_image = Image.merge('LA', (adjusted_image, alpha_channel))
        
        # Convert back to tensor
        output_tensor = pil2tensor(adjusted_image)
        
        return (output_tensor,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ImageAdjustment": ImageAdjustment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAdjustment": "💀Image Adjustment"
}