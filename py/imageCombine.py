import torch
import numpy as np

class ImageCombine:
    """
    A node that combines 4 images into one with customizable layout and spacing.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["horizontal", "vertical"],),
                "spacing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "use_transparent_bg": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Transparent",
                    "label_off": "Solid Color"
                }),
                "bg_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                })
            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_image",)
    FUNCTION = "combine_images"
    CATEGORY = "💀S4Tool"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def combine_images(self, image1, image2, direction, spacing, use_transparent_bg, bg_color, image3=None, image4=None):
        """
        Combine two to four images into one with specified direction and spacing.
        """
        # Convert all images from tensor to numpy arrays
        imgs = [image1[0].cpu().numpy(), image2[0].cpu().numpy()]
        
        # Add optional images if provided
        if image3 is not None:
            imgs.append(image3[0].cpu().numpy())
        if image4 is not None:
            imgs.append(image4[0].cpu().numpy())
        
        # Get dimensions of all images
        heights = [img.shape[0] for img in imgs]
        widths = [img.shape[1] for img in imgs]
        
        # Convert background color
        bg_rgb = np.array(self.hex_to_rgb(bg_color))
        
        if direction == "horizontal":
            # Calculate dimensions for the combined image (including outer spacing)
            total_width = sum(widths) + spacing * (len(imgs) + 1)  # inner spaces + 2 outer spaces
            max_height = max(heights) + spacing * 2  # Add top and bottom spacing
            
            # Create the combined image
            if use_transparent_bg:
                # Create RGBA image with transparent background
                combined = np.zeros((max_height, total_width, 4), dtype=np.float32)
                combined[..., 3] = 0.0  # Set alpha to 0 (transparent)
            else:
                # Create RGB image with solid background color
                combined = np.ones((max_height, total_width, 3), dtype=np.float32) * bg_rgb
            
            # Place each image
            x_offset = spacing  # Start after left spacing
            for i, img in enumerate(imgs):
                h, w = img.shape[:2]
                # Center vertically
                y_offset = (max_height - h) // 2
                
                # Get the target region
                if use_transparent_bg:
                    target_region = combined[y_offset:y_offset+h, x_offset:x_offset+w]
                else:
                    target_region = combined[y_offset:y_offset+h, x_offset:x_offset+w]
                
                # Copy the image
                if img.shape[2] == 4:  # If image has alpha channel
                    if use_transparent_bg:
                        # For transparent background, preserve alpha channel
                        target_region[...] = img
                    else:
                        # For solid background, apply alpha blending
                        alpha = img[..., 3:]
                        target_region[...] = (
                            target_region * (1 - alpha) +
                            img[..., :3] * alpha
                        )
                else:  # If RGB image
                    if use_transparent_bg:
                        # Convert RGB to RGBA with full opacity
                        target_region[..., :3] = img
                        target_region[..., 3] = 1.0
                    else:
                        # Copy RGB directly
                        target_region[...] = img
                
                x_offset += w + spacing
                
        else:  # vertical
            # Calculate dimensions for the combined image (including outer spacing)
            max_width = max(widths) + spacing * 2  # Add left and right spacing
            total_height = sum(heights) + spacing * (len(imgs) + 1)  # inner spaces + 2 outer spaces
            
            # Create the combined image
            if use_transparent_bg:
                # Create RGBA image with transparent background
                combined = np.zeros((total_height, max_width, 4), dtype=np.float32)
                combined[..., 3] = 0.0  # Set alpha to 0 (transparent)
            else:
                # Create RGB image with solid background color
                combined = np.ones((total_height, max_width, 3), dtype=np.float32) * bg_rgb
            
            # Place each image
            y_offset = spacing  # Start after top spacing
            for i, img in enumerate(imgs):
                h, w = img.shape[:2]
                # Center horizontally
                x_offset = (max_width - w) // 2
                
                # Get the target region
                if use_transparent_bg:
                    target_region = combined[y_offset:y_offset+h, x_offset:x_offset+w]
                else:
                    target_region = combined[y_offset:y_offset+h, x_offset:x_offset+w]
                
                # Copy the image
                if img.shape[2] == 4:  # If image has alpha channel
                    if use_transparent_bg:
                        # For transparent background, preserve alpha channel
                        target_region[...] = img
                    else:
                        # For solid background, apply alpha blending
                        alpha = img[..., 3:]
                        target_region[...] = (
                            target_region * (1 - alpha) +
                            img[..., :3] * alpha
                        )
                else:  # If RGB image
                    if use_transparent_bg:
                        # Convert RGB to RGBA with full opacity
                        target_region[..., :3] = img
                        target_region[..., 3] = 1.0
                    else:
                        # Copy RGB directly
                        target_region[...] = img
                
                y_offset += h + spacing
        
        # Convert back to tensor
        return (torch.from_numpy(combined).unsqueeze(0),)

# Node mappings are handled in __init__.py 