import base64
import io
from PIL import Image
import numpy as np
import torch

# Import the shared helper function
from ..nodes_config import pil2tensor
from ..dependency_manager import S4ToolLogger

class ImageFromBase64:
    """
    Node to convert base64 string to image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_string": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "decode_base64_to_image"
    CATEGORY = "💀S4Tool"

    def decode_base64_to_image(self, base64_string):
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Preserve original format - convert to RGBA if it has transparency
            if image.mode in ['P', 'LA', 'PA']:
                # Convert palette or grayscale with alpha to RGBA
                image = image.convert('RGBA')
            elif image.mode == 'L':
                # Convert grayscale to RGB
                image = image.convert('RGB')
            elif image.mode not in ['RGB', 'RGBA']:
                # Convert other formats to RGB
                image = image.convert('RGB')
            
            # Use pil2tensor to convert to proper format
            output_tensor = pil2tensor(image)
            
            # Generate mask from alpha channel
            if image.mode == 'RGBA':
                # Extract alpha channel and convert to mask
                alpha = image.split()[-1]  # Get alpha channel
                mask_array = np.array(alpha).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            else:
                # Create white mask (fully opaque) for RGB images
                mask_array = np.ones((image.height, image.width), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            
            return (output_tensor, mask_tensor)
        except Exception as e:
            # Return a blank image and mask if decoding fails
            blank_image = Image.new('RGB', (64, 64), color=(0, 0, 0))
            output_tensor = pil2tensor(blank_image)
            blank_mask = torch.ones((1, 64, 64), dtype=torch.float32)
            return (output_tensor, blank_mask)