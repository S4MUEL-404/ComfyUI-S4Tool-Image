from ..nodes_config import pil2tensor
from ..dependency_manager import S4ToolLogger

class ImageColor:
    """
    A node that generates a custom color image, supporting single color or gradient color, without transparency.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "color_hex": ("STRING", {
                    "default": "#FFFFFF",
                    "display": "text"
                }),
                "gradient_enabled": ("BOOLEAN", {
                    "default": False,
                    "display": "toggle"
                }),
                "gradient_start_hex": ("STRING", {
                    "default": "#000000",
                    "display": "text"
                }),
                "gradient_end_hex": ("STRING", {
                    "default": "#FFFFFF",
                    "display": "text"
                }),
                "gradient_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Remove mask output
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def generate_image(self, width, height, color_hex, gradient_enabled, 
                      gradient_start_hex, gradient_end_hex, gradient_angle):
        from PIL import Image
        import numpy as np

        # Parse HEX color to RGB (no alpha, always opaque)
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = ''.join(c * 2 for c in hex_color)
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)

        if not gradient_enabled:
            color = hex_to_rgb(color_hex)
            image = Image.new('RGB', (width, height), color=color)  # Use RGB mode, force opaque
        else:
            # Gradient color generation (numpy vectorized, edge-to-edge)
            start_color = np.array(hex_to_rgb(gradient_start_hex), dtype=np.float32)
            end_color = np.array(hex_to_rgb(gradient_end_hex), dtype=np.float32)

            # Create meshgrid for coordinates
            y, x = np.mgrid[0:height, 0:width]
            x = x - width / 2
            y = y - height / 2
            angle_rad = np.radians(gradient_angle)
            proj = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            t = (proj - proj.min()) / (proj.max() - proj.min())
            t = t[..., None]  # shape (H, W, 1)

            # Interpolate colors
            rgb = start_color + t * (end_color - start_color)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb, mode='RGB')

        # Convert to tensor output (RGB format)
        output_tensor = pil2tensor(image)
        return (output_tensor,)