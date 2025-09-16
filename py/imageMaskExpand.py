import numpy as np
import torch
import cv2

from ..dependency_manager import S4ToolLogger

class ImageMaskExpand:
    """
    Expand or contract the input mask by a specified number of pixels with canvas size adjustment.
    Positive values expand (dilate), negative values contract (erode).
    Canvas size automatically adjusts to preserve all content.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_px": ("INT", {
                    "default": 0, 
                    "min": -100, 
                    "max": 100, 
                    "step": 1, 
                    "display": "number",
                    "tooltip": "Positive values expand (dilate), negative values contract (erode)"
                })
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "expand_mask"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def expand_mask(self, mask, expand_px):
        try:
            # Validate and clamp expand_px parameter to safe range
            expand_px = int(max(-100, min(expand_px, 100)))
            S4ToolLogger.info("ImageMaskExpand", f"Processing mask with expand_px: {expand_px}")

            # Normalize input to numpy float32 in [0,1], shape (H, W)
            device = mask.device if isinstance(mask, torch.Tensor) else torch.device('cpu')
            arr = mask
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
                if arr.ndim == 3:  # (1,H,W)
                    arr = arr[0]
            else:
                arr = np.asarray(arr)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
            arr = arr.astype(np.float32)

            original_h, original_w = arr.shape
            S4ToolLogger.info("ImageMaskExpand", f"Original mask size: {original_w}x{original_h}")

            if expand_px > 0:
                result = self._expand_with_sdf(arr, expand_px)
            elif expand_px < 0:
                result = self._contract_with_sdf(arr, abs(expand_px))
            else:
                result = arr

            out = torch.from_numpy(result.astype(np.float32)).unsqueeze(0).to(device)
            final_h, final_w = result.shape
            S4ToolLogger.info("ImageMaskExpand", f"Final mask size: {final_w}x{final_h} (change: {final_w-original_w}x{final_h-original_h})")
            return (out,)
        except Exception as e:
            S4ToolLogger.error("ImageMaskExpand", f"processing mask: {e}")
            import traceback
            traceback.print_exc()
            return (mask,)
    
    def _smoothstep(self, x):
        # Smoothstep function: 3x^2 - 2x^3 for x in [0,1]
        return x * x * (3.0 - 2.0 * x)

    def _signed_distance(self, binary):
        # binary is uint8 0/1
        dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        return dist_in - dist_out

    def _apply_threshold_with_aa(self, sdf, threshold, aa_width=0.75):
        # Anti-aliased step around the threshold over +-aa_width pixels
        v = sdf - threshold
        t = np.clip((v + aa_width) / (2.0 * aa_width), 0.0, 1.0)
        return self._smoothstep(t).astype(np.float32)

    def _expand_with_sdf(self, mask, expand_px):
        """Expand mask by expand_px pixels using SDF; grows canvas to keep content."""
        h, w = mask.shape
        pad = int(expand_px)
        new_h, new_w = h + 2 * pad, w + 2 * pad
        S4ToolLogger.info("ImageMaskExpand", f"Expanding canvas from {w}x{h} to {new_w}x{new_h}")

        padded = np.zeros((new_h, new_w), dtype=np.float32)
        padded[pad:pad + h, pad:pad + w] = mask

        binary = (padded >= 0.5).astype(np.uint8)
        sdf = self._signed_distance(binary)
        # Dilation by r: include pixels where sdf >= -r
        result = self._apply_threshold_with_aa(sdf, threshold=-float(expand_px), aa_width=0.75)
        return result

    def _contract_with_sdf(self, mask, erode_px):
        """Contract mask by erode_px pixels using SDF; keeps original canvas size."""
        h, w = mask.shape
        S4ToolLogger.info("ImageMaskExpand", f"Contracting mask by {erode_px}px")
        binary = (mask >= 0.5).astype(np.uint8)
        sdf = self._signed_distance(binary)
        # Erosion by r: keep pixels where sdf >= r
        result = self._apply_threshold_with_aa(sdf, threshold=float(erode_px), aa_width=0.75)
        return result
