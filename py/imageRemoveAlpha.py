from ..nodes_config import pil2tensor, tensor2pil  # Import shared helper functions
from ..dependency_manager import S4ToolLogger

class ImageRemoveAlpha:
    """
    A node that removes the alpha channel from RGBA images, converting them to RGB format.
    Allows setting a background color for transparent areas.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (RGBA or RGB)
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # Hex color for transparent areas
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgb",)
    FUNCTION = "remove_alpha"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def remove_alpha(self, image, background_color):
        import torch
        from PIL import Image

        # Ensure image is float tensor in [0,1], shape (N,H,W,C)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        image = image.to(dtype=torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # Convert to (N,H,W,C) if needed
        if image.dim() == 4 and image.shape[1] in (1,3,4) and image.shape[-1] not in (3,4):
            image = image.permute(0,2,3,1)

        # Parse background color
        try:
            bg_color = background_color.strip()
            if bg_color.startswith('#'):
                bg_color = bg_color[1:]
            if len(bg_color) == 6:
                r = int(bg_color[0:2], 16) / 255.0
                g = int(bg_color[2:4], 16) / 255.0
                b = int(bg_color[4:6], 16) / 255.0
            else:
                raise ValueError("Invalid hex color format")
        except Exception as e:
            S4ToolLogger.error("ImageRemoveAlpha", f"Invalid background color '{background_color}', using white: {e}")
            r, g, b = 1.0, 1.0, 1.0

        # Process each image in batch
        result_images = []
        for img_tensor in image:
            # Check if image has alpha channel
            if img_tensor.shape[-1] == 4:
                # Extract RGB and alpha
                rgb = img_tensor[..., :3]
                alpha = img_tensor[..., 3:4]
                
                # Create background
                bg = torch.tensor([r, g, b], dtype=torch.float32, device=img_tensor.device)
                bg = bg.view(1, 1, 3).expand(img_tensor.shape[0], img_tensor.shape[1], 3)
                
                # Blend RGB with background using alpha
                result = rgb * alpha + bg * (1.0 - alpha)
            else:
                # Already RGB, no alpha channel
                result = img_tensor[..., :3]
            
            result_images.append(result)

        # Stack back to batch
        output = torch.stack(result_images, dim=0)
        output = torch.clamp(output, 0.0, 1.0)
        
        return (output,)
