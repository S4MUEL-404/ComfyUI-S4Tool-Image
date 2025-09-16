from ..nodes_config import pil2tensor, tensor2pil  # Import shared helper functions
from ..dependency_manager import S4ToolLogger

class ImageBlendWithAlpha:
    """
    A node that combines an image with an Alpha mask, cropping the input image to retain only the opaque areas of the Alpha mask.
    Output is in RGBA format, mimicking the behavior of JoinImageWithAlpha.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (RGB)
                "alpha": ("MASK",),   # Alpha mask (single channel)
                "mask_mode": (["mask", "invert mask"], {"default": "mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_alpha",)
    FUNCTION = "join_image_with_alpha"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def join_image_with_alpha(self, image, alpha, mask_mode):
        import torch
        import torch.nn.functional as F

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
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Prepare alpha: tensor (N or 1, H, W) in [0,1]
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        alpha = alpha.to(dtype=torch.float32, device=image.device)
        if alpha.dim() == 4:
            alpha = alpha.squeeze(1)
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        if mask_mode == "invert mask":
            alpha = 1.0 - alpha

        # Resize alpha to match image spatial size with high-quality interpolation
        n, h, w, _ = image.shape
        if alpha.shape[1:] != (h, w):
            alpha = F.interpolate(alpha.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False, antialias=True).squeeze(1)

        # Match batch size
        if alpha.shape[0] != n:
            if alpha.shape[0] == 1:
                alpha = alpha.expand(n, -1, -1)
            else:
                # If batch sizes mismatch and cannot expand, take first n masks
                alpha = alpha[:n]

        # Compose RGBA tensor
        rgba = torch.cat([image, alpha.unsqueeze(-1)], dim=-1)
        rgba = torch.clamp(rgba, 0.0, 1.0)
        return (rgba,)
