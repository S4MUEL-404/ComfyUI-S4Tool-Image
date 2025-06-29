import torch
from PIL import Image
import numpy as np
import cv2

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
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def expand_mask(self, mask, expand_px):
        # mask: (1, H, W) float32, 0~1
        arr = mask
        if hasattr(arr, 'cpu'):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]

        mask_uint8 = (arr * 255).astype(np.uint8)

        if expand_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px*2+1, expand_px*2+1))
            dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        else:
            dilated = mask_uint8
        out = dilated.astype("float32") / 255.0
        out_tensor = torch.from_numpy(out).unsqueeze(0)
        return (out_tensor,)