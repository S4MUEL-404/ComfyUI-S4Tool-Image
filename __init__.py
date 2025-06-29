__version__ = "1.0.0"

print("Loading ComfyUI-S4Tool nodes...")

from .py.ImageCombine import ImageCombine
from .py.ImageBlendWithAlpha import ImageBlendWithAlpha
from .py.ImageMaskExpand import ImageMaskExpand
from .py.imageCroptoFit import ImageCropToFit
from .py.ImageColor import ImageColor
from .py.ImageOverlay import ImageOverlay
from .py.ImagePalette import ImagePalette
from .py.ImagePalette631 import ImagePalette631
from .py.ImagePrimaryColor import ImagePrimaryColor
from .py.ImageFromBase64 import ImageFromBase64
from .py.ImageToBase64 import ImageToBase64
from .py.ImageTilingPattern import ImageTilingPattern

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
    "ImageToBase64": ImageToBase64,
    "ImageTilingPattern": ImageTilingPattern
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCombine": "💀Image Combine",
    "ImageBlendWithAlpha": "💀Image Blend With Alpha",
    "ImageMaskExpand": "💀Image Mask Expand",
    "ImageCropToFit": "💀Image Crop To Fit",
    "ImageColor": "💀Image Color",
    "ImageOverlay": "💀Image Overlay",
    "ImagePalette": "💀Image Palette",
    "ImagePalette631": "💀Image Palette 6-3-1",
    "ImagePrimaryColor": "💀Image Primary Color",
    "ImageFromBase64": "💀Image from Base64",
    "ImageToBase64": "💀Image To Base64",
    "ImageTilingPattern": "💀Image Tiling Pattern"
}
