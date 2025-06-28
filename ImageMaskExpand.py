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
                "expand_px": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "number"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_mask",)
    FUNCTION = "expand_mask"
    CATEGORY = "\U0001F480S4Tool"
    OUTPUT_NODE = False

    def expand_mask(self, mask, expand_px):
        # mask: (1, H, W) float32, 0~1
        arr = mask
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        # Always expand white area (mask area)
        pil_mask = Image.fromarray((arr * 255).astype("uint8"), mode="L")
        for _ in range(expand_px):
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(3))
        out = np.array(pil_mask).astype("float32") / 255.0
        out_tensor = torch.from_numpy(out).unsqueeze(0)
        return (out_tensor,)
