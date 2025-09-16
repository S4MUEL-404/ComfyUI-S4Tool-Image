import torch
import numpy as np
from PIL import Image
from ..nodes_config import pil2tensor, tensor2pil, ImageUtils
from ..dependency_manager import S4ToolLogger

class ImageResize:
    """
    A node that resizes images with various interpolation methods, scaling modes, and conditions.
    Supports images with transparency and maintains alpha channel.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "interpolation": ([
                    "nearest",
                    "bilinear", 
                    "bicubic",
                    "area",
                    "lanczos",
                    "lanczos3",
                    "mitchell",
                    "catrom"
                ], {"default": "lanczos"}),
                "method": ([
                    "stretch",
                    "keep proportion",
                    "fill / crop",
                    "pad"
                ], {"default": "stretch"}),
                "condition": ([
                    "always",
                    "downscale if bigger",
                    "upscale if smaller",
                    "if bigger area",
                    "if smaller area"
                ], {"default": "always"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_image"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def get_interpolation_method(self, method_name):
        """Convert interpolation method name to PIL constant."""
        methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "area": Image.BOX,  # PIL's BOX is similar to area interpolation
            "lanczos": Image.LANCZOS
        }
        return methods.get(method_name, Image.BILINEAR)

    def should_resize(self, original_width, original_height, target_width, target_height, condition):
        """Check if resize should be performed based on condition."""
        if condition == "always":
            return True
        elif condition == "downscale if bigger":
            return original_width > target_width or original_height > target_height
        elif condition == "upscale if smaller":
            return original_width < target_width or original_height < target_height
        elif condition == "if bigger area":
            return (original_width * original_height) > (target_width * target_height)
        elif condition == "if smaller area":
            return (original_width * original_height) < (target_width * target_height)
        return True

    def resize_image(self, image, width, height, interpolation, method, condition):
        """
        Resize image with specified parameters using high-quality pyvips processing.
        """
        # Convert tensor to PIL image
        pil_image = tensor2pil(image)
        original_width, original_height = pil_image.size
        
        # Check if resize should be performed
        if not self.should_resize(original_width, original_height, width, height, condition):
            S4ToolLogger.info("ImageResize", f"Condition '{condition}' not met - returning original image")
            return (image,)
        
        S4ToolLogger.info("ImageResize", f"Resizing from {original_width}x{original_height} to {width}x{height} using {method}")
        
        # Use high-quality resize with smart method handling
        resized_image = ImageUtils.smart_resize_with_method(
            pil_image, width, height, method, interpolation
        )
        
        # Convert back to tensor
        output_tensor = pil2tensor(resized_image)
        
        S4ToolLogger.success("ImageResize", f"Resize completed successfully")
        return (output_tensor,)

# Node mappings are handled in __init__.py