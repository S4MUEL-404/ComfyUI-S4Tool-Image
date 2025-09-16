import os
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from PIL import Image
import numpy as np
import torch

from ..nodes_config import pil2tensor
from ..dependency_manager import S4ToolLogger


class ImageFromURL:
    """
    Load image from an HTTP(S) URL or local file path.
    Preserves alpha channel if present (e.g., transparent PNG).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter http(s) URL or local file path"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "💀S4Tool"

    def _open_from_url(self, url: str) -> Image.Image:
        headers = {
            "User-Agent": "ComfyUI-S4Tool-Image/1.0 (+https://github.com)"
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data))
        return img

    def _open_from_path(self, path: str) -> Image.Image:
        norm_path = os.path.expandvars(os.path.expanduser(path))
        with open(norm_path, "rb") as f:
            img = Image.open(f)
            # Force load to avoid lazy file handle issues
            img.load()
        return img

    def _ensure_alpha_preserved(self, img: Image.Image) -> Image.Image:
        # Convert modes while preserving transparency when available
        if img.mode in ["RGBA", "LA"]:
            return img.convert("RGBA")
        if img.mode == "P":
            # Palette images may include transparency info
            return img.convert("RGBA")
        if img.mode == "L":
            return img.convert("RGB")
        if img.mode not in ["RGB", "RGBA"]:
            return img.convert("RGB")
        return img

    def load_image(self, url_or_path: str):
        try:
            src = (url_or_path or "").strip()
            if not src:
                # Return a blank RGB image and white mask when input is empty
                blank = Image.new("RGB", (64, 64), (0, 0, 0))
                mask_array = np.ones((64, 64), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
                return (pil2tensor(blank), mask_tensor)

            parsed = urlparse(src)
            if parsed.scheme in ("http", "https"):
                img = self._open_from_url(src)
            else:
                img = self._open_from_path(src)

            img = self._ensure_alpha_preserved(img)

            # Build mask: use alpha channel when available; otherwise a white mask
            if img.mode == "RGBA":
                alpha = img.split()[-1]
                mask_array = (np.array(alpha).astype(np.float32) / 255.0)
            else:
                mask_array = np.ones((img.height, img.width), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

            return (pil2tensor(img), mask_tensor)

        except Exception:
            # On failure, return a blank RGB image and white mask
            blank = Image.new("RGB", (64, 64), (0, 0, 0))
            mask_array = np.ones((64, 64), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            return (pil2tensor(blank), mask_tensor)


