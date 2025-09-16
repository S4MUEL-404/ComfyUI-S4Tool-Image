from ..nodes_config import pil2tensor, tensor2pil
from ..dependency_manager import S4ToolLogger
from PIL import Image
import torch
import numpy as np

class ImageCropToFit:
    """
    Crop the input image to remove all fully transparent borders. Output keeps transparent background.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_to_fit"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def crop_to_fit(self, image):
        try:
            # Convert tensor to PIL image
            pil_img = tensor2pil(image)
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            arr = np.array(pil_img)
            
            # Find non-transparent area
            alpha = arr[..., 3]
            nonzero = np.argwhere(alpha > 0)
            if nonzero.size == 0:
                # All transparent, return a 1x1 transparent image
                cropped = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            else:
                y0, x0 = nonzero.min(axis=0)
                y1, x1 = nonzero.max(axis=0) + 1
                cropped = pil_img.crop((x0, y0, x1, y1))
            
            # Convert back to tensor
            return (pil2tensor(cropped),)
        except Exception as e:
            S4ToolLogger.error("ImageCropToFit", f"cropping image: {e}")
            # Return original image as fallback
            return (image,)
