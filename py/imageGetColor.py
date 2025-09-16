from ..nodes_config import tensor2pil
from ..dependency_manager import S4ToolLogger

class ImageGetColor:
    """
    A node that extracts the hex color value of a specific pixel from an image based on x,y coordinates.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex_color",)
    FUNCTION = "get_pixel_color"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def get_pixel_color(self, image, x, y):
        """
        Extract the hex color value of a pixel at the specified coordinates.
        """
        try:
            # Extract tensor data
            if image.dim() == 4:
                tensor_data = image[0]  # Remove batch dimension
            else:
                tensor_data = image
                
            height, width = tensor_data.shape[0], tensor_data.shape[1]
            
            # Clamp coordinates to image bounds
            clamped_x = max(0, min(x, width - 1))
            clamped_y = max(0, min(y, height - 1))
            
            # Get pixel from tensor directly (y first, then x for tensor indexing)
            pixel_tensor = tensor_data[clamped_y, clamped_x]
            
            # Convert to RGB values (0-255)
            if tensor_data.shape[2] >= 3:
                r = int(pixel_tensor[0].item() * 255)
                g = int(pixel_tensor[1].item() * 255)
                b = int(pixel_tensor[2].item() * 255)
            else:
                # Grayscale
                gray = int(pixel_tensor[0].item() * 255)
                r = g = b = gray
                
            # Ensure RGB values are within valid 0-255 range
            r = max(0, min(r, 255))
            g = max(0, min(g, 255))
            b = max(0, min(b, 255))
                
            # Convert RGB to hex format
            hex_color = "#%02X%02X%02X" % (r, g, b)
            
            return (hex_color,)
        except Exception as e:
            S4ToolLogger.error("ImageGetColor", f"getting pixel color: {e}")
            # Return black color as fallback
            return ("#000000",)

# Node mappings are handled in __init__.py 