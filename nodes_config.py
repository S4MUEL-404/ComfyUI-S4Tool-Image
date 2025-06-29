import torch
import numpy as np
from PIL import Image, ImageChops
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