__version__ = "1.1.0"

print("Loading ComfyUI-S4Tool nodes...")

from .py.imageCombine import ImageCombine
from .py.imageResize import ImageResize
from .py.imageBlendWithAlpha import ImageBlendWithAlpha
from .py.imageMaskExpand import ImageMaskExpand
from .py.imageCroptoFit import ImageCropToFit
from .py.imageColor import ImageColor
from .py.imageBoard import ImageBoard
from .py.imageAdjustment import ImageAdjustment
from .py.imageOverlay import ImageOverlay
from .py.imagePalette import ImagePalette
from .py.imagePalette631 import ImagePalette631
from .py.imagePrimaryColor import ImagePrimaryColor
from .py.imageFromBase64 import ImageFromBase64
from .py.imageToBase64 import ImageToBase64
from .py.imageTilingPattern import ImageTilingPattern
from .py.imageGetColor import ImageGetColor
from .py.setImageBatch import SetImageBatch
from .py.getImageBatch import GetImageBatch
from .py.combineImageBatch import CombineImageBatch

NODE_CLASS_MAPPINGS = {
    "ImageCombine": ImageCombine,
    "ImageResize": ImageResize,
    "ImageBlendWithAlpha": ImageBlendWithAlpha,
    "ImageMaskExpand": ImageMaskExpand,
    "ImageCropToFit": ImageCropToFit,
    "ImageColor": ImageColor,
    "ImageBoard": ImageBoard,
    "ImageAdjustment": ImageAdjustment,
    "ImageOverlay": ImageOverlay,
    "ImagePalette": ImagePalette,
    "ImagePalette631": ImagePalette631,
    "ImagePrimaryColor": ImagePrimaryColor,
    "ImageFromBase64": ImageFromBase64,
    "ImageToBase64": ImageToBase64,
    "ImageTilingPattern": ImageTilingPattern,
    "ImageGetColor": ImageGetColor,
    "SetImageBatch": SetImageBatch,
    "GetImageBatch": GetImageBatch,
    "CombineImageBatch": CombineImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCombine": "💀Image Combine",
    "ImageResize": "💀Image Resize",
    "ImageBlendWithAlpha": "💀Image Blend With Alpha",
    "ImageMaskExpand": "💀Image Mask Expand",
    "ImageCropToFit": "💀Image Crop To Fit",
    "ImageColor": "💀Image Color",
    "ImageBoard": "💀Image Board",
    "ImageAdjustment": "💀Image Adjustment",
    "ImageOverlay": "💀Image Overlay",
    "ImagePalette": "💀Image Palette",
    "ImagePalette631": "💀Image Palette 6-3-1",
    "ImagePrimaryColor": "💀Image Primary Color",
    "ImageFromBase64": "💀Image from Base64",
    "ImageToBase64": "💀Image To Base64",
    "ImageTilingPattern": "💀Image Tiling Pattern",
    "ImageGetColor": "💀Image Get Color",
    "SetImageBatch": "💀Set Image Batch",
    "GetImageBatch": "💀Get Image Batch",
    "CombineImageBatch": "💀Combine Image Batch"
}
