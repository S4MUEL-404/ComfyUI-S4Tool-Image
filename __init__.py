__version__ = "1.3.0"

# Import dependency manager first
from .dependency_manager import check_startup_dependencies, S4ToolLogger

# Check dependencies at startup
all_deps_ok, dep_status = check_startup_dependencies()

if not all_deps_ok:
    S4ToolLogger.error("Startup", "Production quality violated - missing required dependencies")
    S4ToolLogger.error("Startup", "Plugin functionality will be INCOMPLETE until all dependencies are installed")
else:
    S4ToolLogger.success("Startup", "PRODUCTION READY - All dependencies satisfied")
    S4ToolLogger.success("Startup", "Full S4Tool-Image functionality available")

from .py.imageColorPicker import ImageColorPicker
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
from .py.imageFromURL import ImageFromURL
from .py.imageFromFolder import ImageFromFolder
from .py.imageFromFolderByIndex import ImageFromFolderByIndex
from .py.imageToBase64 import ImageToBase64
from .py.imageTilingPattern import ImageTilingPattern
from .py.imageGetColor import ImageGetColor
from .py.imageRMBG import ImageRMBG

NODE_CLASS_MAPPINGS = {
    "ImageColorPicker": ImageColorPicker,
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
    "ImageFromURL": ImageFromURL,
    "ImageFromFolder": ImageFromFolder,
    "ImageFromFolderByIndex": ImageFromFolderByIndex,
    "ImageToBase64": ImageToBase64,
    "ImageTilingPattern": ImageTilingPattern,
    "ImageGetColor": ImageGetColor,
    "ImageRMBG": ImageRMBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColorPicker": "💀Image Color Picker",
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
    "ImageFromURL": "💀Image from URL",
    "ImageFromFolder": "💀Image from Folder",
    "ImageFromFolderByIndex": "💀Image from Folder by Index",
    "ImageToBase64": "💀Image To Base64",
    "ImageTilingPattern": "💀Image Tiling Pattern",
    "ImageGetColor": "💀Image Get Color",
    "ImageRMBG": "💀Image RMBG",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]