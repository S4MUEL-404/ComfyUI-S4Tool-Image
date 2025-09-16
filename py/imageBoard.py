from ..nodes_config import pil2tensor
from ..dependency_manager import S4ToolLogger
from PIL import Image


class ImageBoard:
    """
    A node that generates a fully transparent canvas with the specified width and height.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_board"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def create_board(self, width: int, height: int):
        try:
            # Validate and clamp input parameters to safe ranges
            width = max(1, min(width, 4096))
            height = max(1, min(height, 4096))
            
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            return (pil2tensor(canvas),)
        except Exception as e:
            S4ToolLogger.error("ImageBoard", f"Failed to create canvas: {str(e)}")
            # Return a small transparent canvas as fallback
            fallback_canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            return (pil2tensor(fallback_canvas),)


# Node mappings are handled in __init__.py

