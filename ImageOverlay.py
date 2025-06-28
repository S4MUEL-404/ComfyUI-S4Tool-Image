import torch
import numpy as np
from PIL import Image, ImageChops
import copy
import cv2
from skimage import img_as_float, img_as_ubyte

# Helper function definitions (shared with other files)
def pil2tensor(image):
    # Ensure the image is in RGB or RGBA format
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGBA')
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)

def tensor2pil(image):
    # Ensure it is a PyTorch tensor
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input must be a PyTorch tensor, got {type(image)}")
    
    # Remove batch dimension (if exists)
    if image.dim() == 4:
        image = image[0]
    
    # Convert to numpy array
    array = image.cpu().numpy()
    
    # Adjust dimension order
    if array.ndim == 3 and array.shape[0] in [3, 4]:
        array = np.transpose(array, (1, 2, 0))
    
    # Scale to 0-255 range
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    
    # Convert to PIL image
    if array.shape[-1] == 3:
        return Image.fromarray(array, mode='RGB')
    elif array.shape[-1] == 4:
        return Image.fromarray(array, mode='RGBA')
    else:
        return Image.fromarray(array, mode='L')

def cv22ski(cv2_image):
    return img_as_float(cv2_image)

def ski2cv2(ski):
    return img_as_ubyte(ski)

def blend_multiply(background_image, layer_image):
    img_1 = cv22ski(np.array(background_image))
    img_2 = cv22ski(np.array(layer_image))
    img = img_1 * img_2
    return Image.fromarray((img * 255).astype(np.uint8))

def blend_screen(background_image, layer_image):
    img_1 = cv22ski(np.array(background_image))
    img_2 = cv22ski(np.array(layer_image))
    img = 1 - (1 - img_1) * (1 - img_2)
    return Image.fromarray((img * 255).astype(np.uint8))

def blend_overlay(background_image, layer_image):
    img_1 = cv22ski(np.array(background_image))
    img_2 = cv22ski(np.array(layer_image))
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))
    return Image.fromarray((img * 255).astype(np.uint8))

def blend_soft_light(background_image, layer_image):
    img_1 = cv22ski(np.array(background_image))
    img_2 = cv22ski(np.array(layer_image))
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return Image.fromarray((img * 255).astype(np.uint8))

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

        # Calculate position (use pixel positioning)
        x = x_position
        y = y_position

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