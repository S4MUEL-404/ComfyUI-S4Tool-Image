import os
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import torch

from ..nodes_config import pil2tensor
from ..dependency_manager import S4ToolLogger

class ImageFromFolderByIndex:
    """
    Get a specific image by index from ImageFromFolder output.
    Connect filepaths output from ImageFromFolder to this node.
    Preserves original dimensions of the selected image.
    Manual index selection only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepaths": ("STRING", {"forceInput": True}),
                "index": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "filename", "width", "height")
    FUNCTION = "get_image_by_index"
    CATEGORY = "💀S4Tool"

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

    def _load_image_safe(self, file_path: str) -> Optional[Image.Image]:
        """Safely load an image file with error handling."""
        try:
            with open(file_path, 'rb') as f:
                img = Image.open(f)
                # Force load to avoid lazy file handle issues
                img.load()
            
            # Ensure image is in a compatible format
            if img.mode in ["RGBA", "LA"]:
                return img.convert("RGBA")
            elif img.mode == "P":
                # Palette images may include transparency info
                return img.convert("RGBA")
            elif img.mode == "L":
                return img.convert("RGB")
            elif img.mode not in ["RGB", "RGBA"]:
                return img.convert("RGB")
            
            return img
            
        except Exception as e:
            S4ToolLogger.warning("ImageFromFolderByIndex", f"Failed to load image {file_path}: {str(e)}")
            return None

    def _create_mask_from_image(self, img: Image.Image) -> torch.Tensor:
        """Create mask from image alpha channel or return white mask."""
        if img.mode == "RGBA":
            alpha = img.split()[-1]
            mask_array = (np.array(alpha).astype(np.float32) / 255.0)
        else:
            mask_array = np.ones((img.height, img.width), dtype=np.float32)
        
        return torch.from_numpy(mask_array).unsqueeze(0)

    def get_image_by_index(self, filepaths: List[str], index: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, int]:
        """Get a specific image by index from the filepaths list."""
        try:
            if not isinstance(filepaths, list) or len(filepaths) == 0:
                S4ToolLogger.error("ImageFromFolderByIndex", "No filepaths provided or invalid format")
                blank = Image.new("RGB", (64, 64), (0, 0, 0))
                blank_tensor = pil2tensor(blank)
                blank_mask = torch.ones((1, 64, 64), dtype=torch.float32)
                return (blank_tensor, blank_mask, "no_files", 64, 64)
            
            # Manual index mode only
            current_index = index

            # Validate index (convert to 0-based for array access)
            array_index = current_index - 1  # Convert 1-based to 0-based
            if array_index >= len(filepaths) or array_index < 0:
                S4ToolLogger.warning("ImageFromFolderByIndex", f"Index {current_index} out of range. Available: 1-{len(filepaths)}")
                array_index = min(max(0, array_index), len(filepaths) - 1)
                current_index = array_index + 1  # Convert back to 1-based for logging
                S4ToolLogger.info("ImageFromFolderByIndex", f"Using clamped index: {current_index}")

            # Get the filepath at the specified index (already full path from ImageFromFolder)
            target_filepath = filepaths[array_index]
            target_filename = os.path.basename(target_filepath)

            S4ToolLogger.info("ImageFromFolderByIndex", f"Loading image {current_index}: {target_filename}")

            # Load the specific image
            img = self._load_image_safe(target_filepath)
            if img is None:
                S4ToolLogger.error("ImageFromFolderByIndex", f"Failed to load image: {target_filepath}")
                blank = Image.new("RGB", (64, 64), (0, 0, 0))
                blank_tensor = pil2tensor(blank)
                blank_mask = torch.ones((1, 64, 64), dtype=torch.float32)
                return (blank_tensor, blank_mask, "load_failed", 64, 64)

            # Convert to tensor (keeping original size)
            img_tensor = pil2tensor(img)
            
            # Create mask
            mask_tensor = self._create_mask_from_image(img)
            
            S4ToolLogger.success("ImageFromFolderByIndex", f"Successfully loaded: {target_filename} ({img.width}x{img.height})")

            return (img_tensor, mask_tensor, target_filename, img.width, img.height)

        except Exception as e:
            S4ToolLogger.error("ImageFromFolderByIndex", f"Error getting image by index: {str(e)}")
            # Return blank image on error
            blank = Image.new("RGB", (64, 64), (0, 0, 0))
            blank_tensor = pil2tensor(blank)
            blank_mask = torch.ones((1, 64, 64), dtype=torch.float32)
            return (blank_tensor, blank_mask, f"error: {str(e)}", 64, 64)
