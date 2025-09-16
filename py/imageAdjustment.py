import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance
from colorsys import rgb_to_hsv, hsv_to_rgb

from ..nodes_config import pil2tensor, tensor2pil
from ..dependency_manager import S4ToolLogger

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
                }),
                "color_hex": ("STRING", {
                    "default": "#FFFFFF",
                    "display": "text"
                }),
                "overlay_percent": ("INT", {
                    "default": 0,
                    "min": 0,
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

    def adjust_image(self, image, brightness, contrast, saturation, color_hex, overlay_percent):
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
        
        # Apply color overlay if needed
        overlay_amount = max(0.0, min(1.0, float(overlay_percent) / 100.0))
        if overlay_amount > 0.0:
            # Parse hex to RGB
            def hex_to_rgb_tuple(hx):
                s = (hx or "").lstrip('#')
                if len(s) == 3:
                    s = ''.join(c * 2 for c in s)
                try:
                    r = int(s[0:2], 16)
                    g = int(s[2:4], 16)
                    b = int(s[4:6], 16)
                except Exception:
                    r, g, b = 255, 255, 255
                return r, g, b

            cr, cg, cb = hex_to_rgb_tuple(color_hex)
            
            # Use PIL ImageEnhance for color modification (no pixel manipulation)
            from PIL.ImageEnhance import Color as ColorEnhancer
            from PIL import ImageChops
            
            # Create colored version using PIL's native methods
            # Convert to grayscale to get luminance
            luminance = adjusted_image.convert('L').convert('RGB')
            
            # Create color layer
            color_layer = Image.new('RGB', adjusted_image.size, (cr, cg, cb))
            
            # Use multiply blend to apply color to luminance (preserves shadows/highlights)
            colored_image = ImageChops.multiply(luminance, color_layer)
            
            # Blend original with colored version
            adjusted_image = Image.blend(adjusted_image, colored_image, overlay_amount)

        # Restore alpha channel if original image had one
        if has_alpha and alpha_channel is not None:
            # Convert back to RGBA and restore alpha channel
            if original_mode == 'RGBA':
                adjusted_image = adjusted_image.convert('RGBA')
                r, g, b, _ = adjusted_image.split()
                adjusted_image = Image.merge('RGBA', (r, g, b, alpha_channel))
            elif original_mode == 'LA':
                adjusted_image = adjusted_image.convert('L')
                adjusted_image = Image.merge('LA', (adjusted_image, alpha_channel))
        
        # Convert back to tensor
        output_tensor = pil2tensor(adjusted_image)
        
        return (output_tensor,)

# Node mappings are handled in __init__.py