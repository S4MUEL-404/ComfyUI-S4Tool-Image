import torch

from ..nodes_config import pil2tensor, tensor2pil, ColorExtractor, ImageConstants, BaseColorExtractionNode
from ..dependency_manager import S4ToolLogger

class ImagePalette(BaseColorExtractionNode):
    """
    Extract the five main colors of the image using different algorithms, support sorting by hue, saturation, or brightness, output a preview image with five color blocks, and five color hex strings.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = BaseColorExtractionNode.get_common_input_types()
        # Add sort_mode parameter specific to this node
        base_inputs["required"]["sort_mode"] = (["Hue", "Saturation", "Brightness"], {"default": "Hue"})
        return base_inputs

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("palette_image", "color1", "color2", "color3", "color4", "color5")
    FUNCTION = "extract_palette"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_palette(self, image, algorithm, sort_mode):
        # Prepare image for processing using base class method
        image_array = self.prepare_image_for_processing(image)
        
        # Extract colors using base class method
        colors = self.extract_colors_with_algorithm(image_array, algorithm, n_colors=5)
        
        # Sort colors by selected mode (specific to this node)
        colors = ColorExtractor.sort_colors(colors, sort_mode)
        
        # Create preview image using base class method
        block_w, block_h = ImageConstants.PALETTE_BLOCK_WIDTH, ImageConstants.PALETTE_BLOCK_HEIGHT
        palette_img = self.create_color_preview(colors, block_w, block_h, 5)
        
        # Convert to output format
        palette_tensor = pil2tensor(palette_img)
        hex_colors = self.colors_to_hex(colors, 5)
        
        return (palette_tensor, *hex_colors) 