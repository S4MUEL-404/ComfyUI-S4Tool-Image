import torch

from ..nodes_config import pil2tensor, tensor2pil, ColorExtractor, ImageConstants, BaseColorExtractionNode
from ..dependency_manager import S4ToolLogger

class ImagePalette631(BaseColorExtractionNode):
    """
    Main color extraction with multiple algorithms and preview as three stacked color blocks.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return BaseColorExtractionNode.get_common_input_types()

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_image", "color1", "color2", "color3")
    FUNCTION = "extract_palette_631"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_palette_631(self, image, algorithm):
        # Prepare image for processing using base class method
        image_array = self.prepare_image_for_processing(image)
        
        # Extract colors using base class method
        colors = self.extract_colors_with_algorithm(image_array, algorithm, n_colors=3)
        
        # Create preview image using base class method
        preview_w, block_h = ImageConstants.PALETTE_BLOCK_WIDTH, ImageConstants.PALETTE_BLOCK_HEIGHT
        preview_img = self.create_color_preview(colors, preview_w, block_h, 3)
        
        # Convert to output format
        preview_tensor = pil2tensor(preview_img)
        hex_colors = self.colors_to_hex(colors, 3)
        
        return (preview_tensor, *hex_colors) 