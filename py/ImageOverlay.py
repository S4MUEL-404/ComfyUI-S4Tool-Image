import torch
import numpy as np
from PIL import Image, ImageChops
import copy
import cv2
from skimage import img_as_float, img_as_ubyte

# Import shared helper functions from nodes_config
from ..nodes_config import pil2tensor, tensor2pil, cv22ski, ski2cv2, blend_multiply, blend_screen, blend_overlay, blend_soft_light

class ImageOverlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer image": ("IMAGE",),
                "background image": ("IMAGE",),
                "x_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "y_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "mirror": (["None", "Horizontal", "Vertical"], {
                    "default": "None"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "opacity": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"], {
                    "default": "normal"
                }),
                "alignment": (["Top Left", "Top", "Top Right", "Left", "Center", "Right", "Bottom Left", "Bottom", "Bottom Right"], {
                    "default": "Top Left"
                }),
            },
            "optional": {
                "layer mask (optional)": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def image_rotate_extend_with_alpha(self, image, angle, alpha, method="lanczos", anti_aliasing=0):
        # Ensure the image is in RGBA mode
        image = image.convert('RGBA')
        alpha = alpha.convert('L')

        # Calculate the size after rotation
        w, h = image.size
        angle_rad = np.radians(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))
        new_w = int(w * cos_a + h * sin_a)
        new_h = int(w * sin_a + h * cos_a)

        # Create a new canvas
        rotated_image = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
        rotated_alpha = Image.new('L', (new_w, new_h), 0)

        # Calculate offset
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2

        # Paste the original image to the center of the new canvas
        rotated_image.paste(image, (offset_x, offset_y))
        rotated_alpha.paste(alpha, (offset_x, offset_y))

        # Set resampling method
        resample_method = {
            "lanczos": Image.Resampling.BICUBIC,
            "bicubic": Image.Resampling.BICUBIC,
            "hamming": Image.Resampling.BILINEAR,
            "bilinear": Image.Resampling.BILINEAR,
            "box": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST
        }.get(method.lower(), Image.Resampling.BICUBIC)

        # Apply anti-aliasing
        if anti_aliasing > 0:
            scale = 1 + anti_aliasing / 4
            temp_size = (int(new_w * scale), int(new_h * scale))
            rotated_image = rotated_image.resize(temp_size, resample_method)
            rotated_alpha = rotated_alpha.resize(temp_size, resample_method)
            rotated_image = rotated_image.rotate(angle, resample=resample_method, expand=False)
            rotated_alpha = rotated_alpha.rotate(angle, resample=resample_method, expand=False)
            rotated_image = rotated_image.resize((new_w, new_h), resample_method)
            rotated_alpha = rotated_alpha.resize((new_w, new_h), resample_method)
        else:
            rotated_image = rotated_image.rotate(angle, resample=resample_method, expand=False)
            rotated_alpha = rotated_alpha.rotate(angle, resample=resample_method, expand=False)

        return rotated_image, rotated_alpha, Image.merge('RGBA', (*rotated_image.split()[:3], rotated_alpha))

    def calculate_alignment_offset(self, background_size, layer_size, alignment):
        """
        Calculate the offset for different alignment modes
        
        Args:
            background_size: (width, height) of background image
            layer_size: (width, height) of layer image
            alignment: alignment mode string
            
        Returns:
            (offset_x, offset_y): offset to be added to x_position, y_position
        """
        bg_width, bg_height = background_size
        layer_width, layer_height = layer_size
        
        # Define background anchor points
        bg_anchors = {
            "Top Left": (0, 0),
            "Top": (bg_width // 2, 0),
            "Top Right": (bg_width, 0),
            "Left": (0, bg_height // 2),
            "Center": (bg_width // 2, bg_height // 2),
            "Right": (bg_width, bg_height // 2),
            "Bottom Left": (0, bg_height),
            "Bottom": (bg_width // 2, bg_height),
            "Bottom Right": (bg_width, bg_height)
        }
        
        # Define layer offset (to position layer's anchor point)
        layer_offsets = {
            "Top Left": (0, 0),
            "Top": (-layer_width // 2, 0),
            "Top Right": (-layer_width, 0),
            "Left": (0, -layer_height // 2),
            "Center": (-layer_width // 2, -layer_height // 2),
            "Right": (-layer_width, -layer_height // 2),
            "Bottom Left": (0, -layer_height),
            "Bottom": (-layer_width // 2, -layer_height),
            "Bottom Right": (-layer_width, -layer_height)
        }
        
        bg_anchor = bg_anchors[alignment]
        layer_offset = layer_offsets[alignment]
        
        # Final offset = background anchor + layer offset
        offset_x = bg_anchor[0] + layer_offset[0]
        offset_y = bg_anchor[1] + layer_offset[1]
        
        return offset_x, offset_y

    def blend_images(self, **kwargs):
        # Extract parameters
        Layer_image = kwargs.get("layer image")
        Background_image = kwargs.get("background image")
        x_position = kwargs.get("x_position", 0)
        y_position = kwargs.get("y_position", 0)
        mirror = kwargs.get("mirror", "None")
        rotation = kwargs.get("rotation", 0.0)
        scale = kwargs.get("scale", 1.0)
        opacity = kwargs.get("opacity", 100)
        blend_mode = kwargs.get("blend_mode", "normal")
        alignment = kwargs.get("alignment", "Top Left")
        Layer_mask = kwargs.get("layer mask (optional)")

        # Convert to PIL image
        background_pil = tensor2pil(Background_image).convert('RGBA')
        layer_pil = tensor2pil(Layer_image)

        # Handle optional mask
        if Layer_mask is not None:
            # Ensure mask is a 2D tensor
            if Layer_mask.dim() == 3:
                Layer_mask = Layer_mask.squeeze(0)  # Remove batch dimension
            elif Layer_mask.dim() == 4:
                Layer_mask = Layer_mask[0, 0]  # Take the first channel of the first batch
            
            # Convert to PIL image
            mask_pil = Image.fromarray((Layer_mask.cpu().numpy() * 255).astype(np.uint8), mode='L')
            
            if mask_pil.size != layer_pil.size:
                mask_pil = mask_pil.resize(layer_pil.size, Image.Resampling.LANCZOS)
            
            # Invert mask logic
            mask_array = np.array(mask_pil)
            mask_array = 255 - mask_array
            layer_alpha = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        else:
            if layer_pil.mode == 'RGBA':
                layer_alpha = layer_pil.split()[-1]
            else:
                layer_alpha = Image.new('L', layer_pil.size, 'white')

        # Apply scaling
        if scale != 1.0:
            target_width = int(layer_pil.width * scale)
            target_height = int(layer_pil.height * scale)
            layer_pil = layer_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            layer_alpha = layer_alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Apply mirroring
        if mirror == "Horizontal":
            layer_pil = layer_pil.transpose(Image.FLIP_LEFT_RIGHT)
            layer_alpha = layer_alpha.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == "Vertical":
            layer_pil = layer_pil.transpose(Image.FLIP_TOP_BOTTOM)
            layer_alpha = layer_alpha.transpose(Image.FLIP_TOP_BOTTOM)

        # Apply rotation
        if rotation != 0:
            layer_pil, layer_alpha, _ = self.image_rotate_extend_with_alpha(
                layer_pil, rotation, layer_alpha, "lanczos", anti_aliasing=4
            )

        # Calculate position based on alignment mode
        alignment_offset_x, alignment_offset_y = self.calculate_alignment_offset(
            background_pil.size, layer_pil.size, alignment
        )
        x = x_position + alignment_offset_x
        y = y_position + alignment_offset_y

        # Composite layers
        comp = copy.copy(background_pil)
        comp_mask = Image.new("RGB", comp.size, color='black')
        comp.paste(layer_pil, (x, y))
        comp_mask.paste(layer_alpha.convert('RGB'), (x, y))
        comp_mask = comp_mask.convert('L')

        # Apply blend mode
        if blend_mode == "normal":
            final_image = copy.deepcopy(comp)
        else:
            # Convert image to numpy array for processing
            bg_array = np.array(background_pil).astype(np.float32) / 255.0
            layer_array = np.array(comp).astype(np.float32) / 255.0
            
            if blend_mode == "multiply":
                result = bg_array * layer_array
            elif blend_mode == "screen":
                result = 1 - (1 - bg_array) * (1 - layer_array)
            elif blend_mode == "overlay":
                mask = bg_array < 0.5
                result = 2 * bg_array * layer_array * mask + \
                        (1 - mask) * (1 - 2 * (1 - bg_array) * (1 - layer_array))
            elif blend_mode == "soft_light":
                mask = layer_array < 0.5
                result = (2 * layer_array - 1) * (bg_array - bg_array * bg_array) + bg_array
                result = result * mask + \
                        ((2 * layer_array - 1) * (np.sqrt(bg_array) - bg_array) + bg_array) * (1 - mask)

            # Convert result back to PIL image
            final_array = np.clip(result * 255, 0, 255).astype(np.uint8)
            final_image = Image.fromarray(final_array)

        # Apply transparency
        if opacity < 100:
            alpha = 1.0 - float(opacity) / 100
            final_image = Image.blend(final_image, background_pil, alpha)

        # Final composition
        background_pil.paste(final_image, mask=comp_mask)

        # Convert to tensor and return
        return (pil2tensor(background_pil),)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ImageOverlay": ImageOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageOverlay": "💀Image Overlay"
}