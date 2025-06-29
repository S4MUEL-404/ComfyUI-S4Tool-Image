from ..nodes_config import pil2tensor, tensor2pil  # Import shared helper functions

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
        from PIL import Image
        import torch
        import numpy as np

        # Convert input tensor to PIL image
        image_pil = tensor2pil(image).convert('RGB')  # Ensure it is RGB format
        
        # Process alpha channel
        if isinstance(alpha, torch.Tensor):
            # Ensure alpha is a 2D tensor
            if alpha.dim() == 4:
                alpha = alpha.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
            elif alpha.dim() == 3:
                alpha = alpha.squeeze(0)  # Only remove batch dimension
            # Convert alpha to numpy array and normalize
            alpha_np = alpha.cpu().numpy()
            if alpha_np.max() > 1.0:
                alpha_np = alpha_np / 255.0
            # Invert mask if needed
            if mask_mode == "invert mask":
                alpha_np = 1.0 - alpha_np
            alpha_pil = Image.fromarray((alpha_np * 255).astype(np.uint8), 'L')
        else:
            alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), 'L')
            if mask_mode == "invert mask":
                alpha_pil = Image.fromarray((255 - np.array(alpha_pil)).astype(np.uint8), 'L')

        # If alpha channel size does not match image, resize alpha
        if alpha_pil.size != image_pil.size:
            alpha_pil = alpha_pil.resize(image_pil.size, Image.Resampling.LANCZOS)

        # Directly merge RGB and alpha to RGBA (no invert, no paste)
        r, g, b = image_pil.split()
        rgba_image = Image.merge('RGBA', (r, g, b, alpha_pil))

        # Use pil2tensor function to convert to correct format
        output_tensor = pil2tensor(rgba_image)

        return (output_tensor,)