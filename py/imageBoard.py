from ..nodes_config import pil2tensor
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
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        return (pil2tensor(canvas),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ImageBoard": ImageBoard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBoard": "💀Image Board",
}

