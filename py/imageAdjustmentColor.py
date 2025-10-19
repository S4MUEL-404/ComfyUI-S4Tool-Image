import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance
from colorsys import rgb_to_hsv, hsv_to_rgb

from ..nodes_config import pil2tensor, tensor2pil
from ..dependency_manager import S4ToolLogger

class ImageAdjustmentColor:
    """
    A node that adjusts brightness, contrast, and saturation of a hex color value.
    Takes hex color input and outputs adjusted hex color.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_color_hex": ("STRING", {
                    "default": "#FF0000",
                    "display": "text"
                }),
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("adjusted_color_hex",)
    FUNCTION = "adjust_color"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def adjust_color(self, input_color_hex, brightness, contrast, saturation, color_hex, overlay_percent):
        """
        Adjust brightness, contrast, and saturation of the input hex color.
        Returns adjusted hex color.
        """
        try:
            S4ToolLogger.debug("ImageAdjustmentColor", f"Adjusting color {input_color_hex} with brightness={brightness}, contrast={contrast}, saturation={saturation}")
            
            # Parse input hex color to RGB
            def hex_to_rgb_tuple(hx):
                s = (hx or "").lstrip('#')
                if len(s) == 3:
                    s = ''.join(c * 2 for c in s)
                try:
                    r = int(s[0:2], 16)
                    g = int(s[2:4], 16)
                    b = int(s[4:6], 16)
                except Exception:
                    S4ToolLogger.warning("ImageAdjustmentColor", f"Invalid hex color {hx}, using default red")
                    r, g, b = 255, 0, 0
                return r, g, b

            # RGB to hex conversion
            def rgb_to_hex(r, g, b):
                # Clamp values to 0-255 range
                r = max(0, min(255, int(r)))
                g = max(0, min(255, int(g)))
                b = max(0, min(255, int(b)))
                return f"#{r:02X}{g:02X}{b:02X}"

            # Get RGB values from input hex
            r, g, b = hex_to_rgb_tuple(input_color_hex)
            
            # Create a 1x1 PIL image with the input color for processing
            color_image = Image.new('RGB', (1, 1), (r, g, b))
            adjusted_image = color_image
            
            # Apply adjustments to the color image (same logic as original)
            
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
            
            # Extract the adjusted color from the 1x1 image
            adjusted_rgb = adjusted_image.getpixel((0, 0))
            
            # Convert back to hex
            output_hex = rgb_to_hex(adjusted_rgb[0], adjusted_rgb[1], adjusted_rgb[2])
            
            S4ToolLogger.success("ImageAdjustmentColor", f"Color adjusted: {input_color_hex} -> {output_hex}")
            return (output_hex,)
            
        except Exception as e:
            S4ToolLogger.error("ImageAdjustmentColor", f"Color adjustment failed: {str(e)}")
            # Return input color as fallback
            return (input_color_hex,)

# Node mappings are handled in __init__.py