print("Loading ComfyUI-S4Tool nodes...")

from .ImageCombine import ImageCombine
from .ImageBlendWithAlpha import ImageBlendWithAlpha
from .ImageMaskExpand import ImageMaskExpand
from .imageCroptoFit import ImageCropToFit
from .ImageColor import ImageColor
from .ImageOverlay import ImageOverlay
from .ImagePalette import ImagePalette
from .ImagePalette631 import ImagePalette631
from .ImagePrimaryColor import ImagePrimaryColor
from .ImageFromBase64 import ImageFromBase64
from .ImageTilingPattern import ImageTilingPattern

NODE_CLASS_MAPPINGS = {
    "ImageCombine": ImageCombine,
    "ImageBlendWithAlpha": ImageBlendWithAlpha,
    "ImageMaskExpand": ImageMaskExpand,
    "ImageCropToFit": ImageCropToFit,
    "ImageColor": ImageColor,
    "ImageOverlay": ImageOverlay,
    "ImagePalette": ImagePalette,
    "ImagePalette631": ImagePalette631,
    "ImagePrimaryColor": ImagePrimaryColor,
    "ImageFromBase64": ImageFromBase64,
    "ImageTilingPattern": ImageTilingPattern
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCombine": "💀Image Combine",
    "ImageBlendWithAlpha": "💀Image Blend with Alpha",
    "ImageMaskExpand": "💀Image Mask Expand",
    "ImageCropToFit": "💀Image Crop To Fit",
    "ImageColor": "💀Image Color",
    "ImageOverlay": "💀Image Overlay",
    "ImagePalette": "💀Image Palette",
    "ImagePalette631": "💀Image Palette 6-3-1",
    "ImagePrimaryColor": "💀Image Primary Color",
    "ImageFromBase64": "💀Image from Base64",
    "ImageTilingPattern": "💀Image Tiling Pattern"
}