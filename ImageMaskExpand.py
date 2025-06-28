import torch
from PIL import Image, ImageFilter
import numpy as np

class ImageMaskExpand:
    """
    Expand the input mask by a specified number of pixels. Supports original mask or inverted mask mode.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_px": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "number"}),
                "mode": (["Mask", "Invert Mask"], {"default": "Mask"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_mask",)
    FUNCTION = "expand_mask"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def expand_mask(self, mask, expand_px, mode):
        # mask: (1, H, W) float32, 0~1
        arr = mask
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        # Invert mask if needed
        if mode == "Invert Mask":
            arr = 1.0 - arr
        pil_mask = Image.fromarray((arr * 255).astype("uint8"), mode="L")
        for _ in range(expand_px):
            if mode == "Invert Mask":
                # Use MinFilter for invert mode to shrink white areas and expand black areas
                pil_mask = pil_mask.filter(ImageFilter.MinFilter(3))
            else:
                # Use MaxFilter for normal mode to expand white areas
                pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))
        out = np.array(pil_mask).astype("float32") / 255.0
        out_tensor = torch.from_numpy(out).unsqueeze(0)
        return (out_tensor,)