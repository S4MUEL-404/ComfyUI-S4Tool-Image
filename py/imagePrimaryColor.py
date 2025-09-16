import torch

from ..nodes_config import pil2tensor, tensor2pil, ColorExtractor, ImageConstants, BaseColorExtractionNode
from ..dependency_manager import S4ToolLogger

class ImagePrimaryColor(BaseColorExtractionNode):
    """
    Extract the primary color of the image using different algorithms. Output a preview image (400x80) and the hex string of the primary color.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return BaseColorExtractionNode.get_common_input_types()

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "color")
    FUNCTION = "extract_primary_color"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_primary_color(self, image, algorithm):
        # Prepare image for processing using base class method
        image_array = self.prepare_image_for_processing(image)
        
        # Extract primary color using base class method
        colors = self.extract_colors_with_algorithm(image_array, algorithm, n_colors=1)
        
        # Create preview image using base class method
        preview_w, block_h = ImageConstants.PALETTE_BLOCK_WIDTH, ImageConstants.PALETTE_BLOCK_HEIGHT
        preview_img = self.create_color_preview(colors, preview_w, block_h, 1)
        
        # Convert to output format
        preview_tensor = pil2tensor(preview_img)
        hex_colors = self.colors_to_hex(colors, 1)
        
        return (preview_tensor, hex_colors[0]) 