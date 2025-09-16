import torch
import numpy as np
from PIL import Image, ImageEnhance
from ..nodes_config import pil2tensor, tensor2pil, ImageUtils
from ..dependency_manager import S4ToolLogger

class ImageOverlay:
    """
    A node that overlays a layer image onto a background image using pyvips for professional alpha compositing.
    Supports position, rotation, scaling, opacity, mirroring, and different blend modes with perfect transparency handling.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "x_position": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "y_position": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "mirror": (["none", "horizontal", "vertical", "both"], {"default": "none"}),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "soft-light", 
                    "hard-light", "colour-dodge", "colour-burn", "darken", 
                    "lighten", "difference", "exclusion", "add", "subtract"
                ], {"default": "normal"}),
                "alignment": ([
                    "top_left", "top", "top_right", 
                    "left", "center", "right", 
                    "bottom_left", "bottom", "bottom_right"
                ], {"default": "top_left"})
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "overlay_image"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False
    
    def ensure_rgba(self, pil_image):
        """Convert PIL image to RGBA format ensuring proper alpha channel."""
        if pil_image.mode == 'RGBA':
            return pil_image
        elif pil_image.mode == 'RGB':
            return pil_image.convert('RGBA')
        elif pil_image.mode in ['L', 'P']:
            return pil_image.convert('RGBA')
        else:
            return pil_image.convert('RGBA')
    
    def calculate_alignment_offset(self, bg_size, layer_size, alignment, x_pos, y_pos):
        """Calculate the final position based on alignment and offset."""
        bg_w, bg_h = bg_size
        layer_w, layer_h = layer_size
        
        # Base alignment positions
        alignment_positions = {
            "top_left": (0, 0),
            "top": ((bg_w - layer_w) // 2, 0),
            "top_right": (bg_w - layer_w, 0),
            "left": (0, (bg_h - layer_h) // 2),
            "center": ((bg_w - layer_w) // 2, (bg_h - layer_h) // 2),
            "right": (bg_w - layer_w, (bg_h - layer_h) // 2),
            "bottom_left": (0, bg_h - layer_h),
            "bottom": ((bg_w - layer_w) // 2, bg_h - layer_h),
            "bottom_right": (bg_w - layer_w, bg_h - layer_h)
        }
        
        base_x, base_y = alignment_positions.get(alignment, (0, 0))
        return base_x + x_pos, base_y + y_pos
    
    def calculate_target_center_position(self, bg_size, original_layer_size, alignment, x_offset, y_offset):
        """Calculate where the center of the original layer should be positioned."""
        bg_w, bg_h = bg_size
        layer_w, layer_h = original_layer_size
        
        # Get the center position for each alignment
        alignment_centers = {
            "top_left": (layer_w/2.0, layer_h/2.0),
            "top": (bg_w/2.0, layer_h/2.0),
            "top_right": (bg_w - layer_w/2.0, layer_h/2.0),
            "left": (layer_w/2.0, bg_h/2.0),
            "center": (bg_w/2.0, bg_h/2.0),
            "right": (bg_w - layer_w/2.0, bg_h/2.0),
            "bottom_left": (layer_w/2.0, bg_h - layer_h/2.0),
            "bottom": (bg_w/2.0, bg_h - layer_h/2.0),
            "bottom_right": (bg_w - layer_w/2.0, bg_h - layer_h/2.0)
        }
        
        base_center_x, base_center_y = alignment_centers.get(alignment, (layer_w/2.0, layer_h/2.0))
        return base_center_x + x_offset, base_center_y + y_offset
    
    def apply_transformations(self, pil_img, mirror, rotation, scale, opacity):
        """Apply transformations using professional OpenCV methods."""
        result = self.ensure_rgba(pil_img)
        
        # 1. Apply mirroring first
        if mirror == "horizontal":
            result = result.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == "vertical":
            result = result.transpose(Image.FLIP_TOP_BOTTOM)
        elif mirror == "both":
            result = result.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        
        # 2-4. Apply scaling, rotation, and opacity using premium quality methods
        result = ImageUtils.apply_transforms_premium(result, scale, rotation, opacity)
        
        return result
    
    def overlay_image(self, background_image, layer_image, x_position, y_position, 
                     mirror, rotation, scale, opacity, blend_mode, alignment, layer_mask=None):
        """
        Overlay the layer image onto the background image using pyvips for perfect alpha compositing.
        """
        # Convert tensors to PIL images
        bg_pil = tensor2pil(background_image)
        layer_pil = tensor2pil(layer_image)
        
        # Process layer mask if provided
        if layer_mask is not None:
            # Convert mask tensor to PIL
            if isinstance(layer_mask, torch.Tensor):
                if layer_mask.dim() == 4:
                    layer_mask = layer_mask.squeeze(0).squeeze(0)
                elif layer_mask.dim() == 3:
                    layer_mask = layer_mask.squeeze(0)
                mask_array = layer_mask.cpu().numpy()
                if mask_array.max() > 1.0:
                    mask_array = mask_array / 255.0
                mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')
            else:
                mask_pil = Image.fromarray((layer_mask * 255).astype(np.uint8), 'L')
            
            # Apply mask to layer image
            if layer_pil.mode != 'RGBA':
                layer_pil = layer_pil.convert('RGBA')
            
            # Resize mask to match layer if needed
            if mask_pil.size != layer_pil.size:
                mask_pil = mask_pil.resize(layer_pil.size, Image.LANCZOS)
            
            # Replace or multiply with existing alpha
            r, g, b, a = layer_pil.split()
            # Multiply existing alpha with mask
            mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
            alpha_array = np.array(a, dtype=np.float32) / 255.0
            combined_alpha = mask_array * alpha_array
            new_alpha = Image.fromarray((combined_alpha * 255).astype(np.uint8), 'L')
            layer_pil = Image.merge('RGBA', (r, g, b, new_alpha))
        
        # Ensure both images are RGBA
        bg_pil = self.ensure_rgba(bg_pil)
        layer_pil = self.ensure_rgba(layer_pil)
        
        # Store original layer center for center-based positioning
        original_center_x = layer_pil.width / 2.0
        original_center_y = layer_pil.height / 2.0
        
        # Apply transformations to layer (this will change dimensions for scaling/rotation)
        layer_transformed = self.apply_transformations(layer_pil, mirror, rotation, scale, opacity)
        
        # Calculate new center of transformed layer
        transformed_center_x = layer_transformed.width / 2.0
        transformed_center_y = layer_transformed.height / 2.0
        
        # Calculate where we want the original center to be positioned
        # Start with alignment-based position for the original image center
        target_center_x, target_center_y = self.calculate_target_center_position(
            (bg_pil.width, bg_pil.height), 
            (layer_pil.width, layer_pil.height), 
            alignment, x_position, y_position
        )
        
        # Calculate final top-left position of transformed image
        # to place its center at the target center position
        final_x = int(target_center_x - transformed_center_x)
        final_y = int(target_center_y - transformed_center_y)
        
        S4ToolLogger.info("Overlay", f"Original center: ({original_center_x:.1f}, {original_center_y:.1f})")
        S4ToolLogger.info("Overlay", f"Target center: ({target_center_x:.1f}, {target_center_y:.1f})")
        S4ToolLogger.info("Overlay", f"Final position: ({final_x}, {final_y})")
        
        # Handle cropping if the layer extends beyond background boundaries
        layer_to_paste = layer_transformed
        paste_x, paste_y = final_x, final_y
        
        # If layer extends beyond boundaries, crop it
        if final_x < 0 or final_y < 0 or \
           final_x + layer_transformed.width > bg_pil.width or \
           final_y + layer_transformed.height > bg_pil.height:
            
            # Calculate the region that will be visible
            visible_left = max(0, final_x)
            visible_top = max(0, final_y)
            visible_right = min(bg_pil.width, final_x + layer_transformed.width)
            visible_bottom = min(bg_pil.height, final_y + layer_transformed.height)
            
            if visible_right > visible_left and visible_bottom > visible_top:
                # Calculate crop region from the transformed layer
                crop_left = max(0, -final_x)
                crop_top = max(0, -final_y)
                crop_right = crop_left + (visible_right - visible_left)
                crop_bottom = crop_top + (visible_bottom - visible_top)
                
                # Crop the transformed layer
                layer_to_paste = layer_transformed.crop((crop_left, crop_top, crop_right, crop_bottom))
                paste_x, paste_y = visible_left, visible_top
            else:
                # No visible region
                layer_to_paste = None
        
        # Perform compositing using PIL
        result_pil = bg_pil.copy()
        
        if layer_to_paste is not None:
            # Use ImageUtils.advanced_composite for professional blending
            result_pil = ImageUtils.advanced_composite(result_pil, layer_to_paste, paste_x, paste_y, blend_mode)
        
        # Convert to tensor
        result_tensor = pil2tensor(result_pil)
        
        # Create output mask from alpha channel
        if result_pil.mode == 'RGBA':
            alpha_pil = result_pil.split()[-1]
            alpha_array = np.array(alpha_pil, dtype=np.float32) / 255.0
        else:
            # If no alpha, create full opacity mask
            alpha_array = np.ones((result_pil.height, result_pil.width), dtype=np.float32)
        
        mask_tensor = torch.from_numpy(alpha_array).unsqueeze(0).unsqueeze(0)
        
        return (result_tensor, mask_tensor)


# Node mappings are handled in __init__.py