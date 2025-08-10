import torch
import numpy as np
from PIL import Image
from ..nodes_config import pil2tensor, tensor2pil

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
                    "lanczos"
                ], {"default": "bilinear"}),
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
        Resize image with specified parameters.
        """
        # Convert tensor to PIL image
        pil_image = tensor2pil(image)
        original_width, original_height = pil_image.size
        
        # Check if resize should be performed
        if not self.should_resize(original_width, original_height, width, height, condition):
            # Return original image if condition not met
            return (image,)
        
        # Get interpolation method
        interp_method = self.get_interpolation_method(interpolation)
        
        # Handle different scaling methods
        if method == "stretch":
            # Simple resize to exact dimensions
            resized_image = pil_image.resize((width, height), interp_method)
            
        elif method == "keep proportion":
            # Calculate scale to fit within target dimensions while maintaining aspect ratio
            scale_w = width / original_width
            scale_h = height / original_height
            scale = min(scale_w, scale_h)
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            resized_image = pil_image.resize((new_width, new_height), interp_method)
            
        elif method == "fill / crop":
            # Scale to fill target dimensions, then crop center
            scale_w = width / original_width
            scale_h = height / original_height
            scale = max(scale_w, scale_h)
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize first
            temp_image = pil_image.resize((new_width, new_height), interp_method)
            
            # Calculate crop box to center the image
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            
            resized_image = temp_image.crop((left, top, right, bottom))
            
        elif method == "pad":
            # Scale to fit within target dimensions, then pad with transparency
            scale_w = width / original_width
            scale_h = height / original_height
            scale = min(scale_w, scale_h)
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize first
            temp_image = pil_image.resize((new_width, new_height), interp_method)
            
            # Create new image with target size and transparent background
            if pil_image.mode in ['RGBA', 'LA']:
                # Image already has alpha channel
                resized_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                if temp_image.mode != 'RGBA':
                    temp_image = temp_image.convert('RGBA')
            else:
                # Convert to RGBA for transparency support
                resized_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                temp_image = temp_image.convert('RGBA')
            
            # Calculate position to center the resized image
            x = (width - new_width) // 2
            y = (height - new_height) // 2
            
            # Paste the resized image onto the padded canvas
            resized_image.paste(temp_image, (x, y), temp_image)
        
        else:
            # Default to stretch if unknown method
            resized_image = pil_image.resize((width, height), interp_method)
        
        # Convert back to tensor
        output_tensor = pil2tensor(resized_image)
        
        return (output_tensor,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "💀Image Resize"
}