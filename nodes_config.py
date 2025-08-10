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

# Global batch storage for S4Tool Image Batch nodes
GLOBAL_BATCH_STORAGE = {}
print("=" * 80)
print("🚀 [S4Tool] BATCH STORAGE MODULE LOADED!")
print(f"🚀 [S4Tool] Current batches in storage: {len(GLOBAL_BATCH_STORAGE)}")
print("🚀 [S4Tool] Ready for batch operations...")
print("=" * 80)

def store_batch(batch_name, images, masks):
    """Store a batch with the given name."""
    print(f"🔵 [S4Tool-Store] Attempting to store batch '{batch_name}'")
    
    if not batch_name or not batch_name.strip():
        error_msg = "Batch name cannot be empty!"
        print(f"🔴 [S4Tool-Store] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    if not images or not masks:
        error_msg = "Images and masks cannot be empty!"
        print(f"🔴 [S4Tool-Store] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    if len(images) != len(masks):
        error_msg = f"Number of images ({len(images)}) must match number of masks ({len(masks)})"
        print(f"🔴 [S4Tool-Store] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    clean_name = batch_name.strip()
    
    # Show existing batches before storing
    existing_batches = list(GLOBAL_BATCH_STORAGE.keys())
    print(f"🟡 [S4Tool-Store] Before storing - existing batches: {existing_batches}")
    
    # Store the batch
    GLOBAL_BATCH_STORAGE[clean_name] = (images, masks)
    
    # Show updated batches after storing
    updated_batches = list(GLOBAL_BATCH_STORAGE.keys())
    print(f"✅ [S4Tool-Store] SUCCESS: Stored batch '{clean_name}' with {len(images)} images")
    print(f"✅ [S4Tool-Store] Updated storage - all batches: {updated_batches}")
    print(f"✅ [S4Tool-Store] Storage now contains {len(GLOBAL_BATCH_STORAGE)} total batches")

def get_batch(batch_name):
    """Retrieve a batch by name."""
    print(f"🔵 [S4Tool-Get] Attempting to retrieve batch '{batch_name}'")
    
    all_batches = list(GLOBAL_BATCH_STORAGE.keys())
    
    if not batch_name or not batch_name.strip():
        error_msg = f"Batch name is empty! Available batches: {all_batches}"
        print(f"🔴 [S4Tool-Get] ERROR: {error_msg}")
        return None
        
    clean_name = batch_name.strip()
    print(f"🟡 [S4Tool-Get] Looking for batch '{clean_name}' in storage...")
    print(f"🟡 [S4Tool-Get] Currently stored batches: {all_batches}")
    print(f"🟡 [S4Tool-Get] Total batches in storage: {len(GLOBAL_BATCH_STORAGE)}")
    
    batch_data = GLOBAL_BATCH_STORAGE.get(clean_name)
    if batch_data:
        print(f"✅ [S4Tool-Get] SUCCESS: Found batch '{clean_name}' with {len(batch_data[0])} images")
        return batch_data
    else:
        print(f"🔴 [S4Tool-Get] ERROR: Batch '{clean_name}' not found!")
        print(f"🔴 [S4Tool-Get] Available batches: {all_batches}")
        print(f"🔴 [S4Tool-Get] Storage contains {len(GLOBAL_BATCH_STORAGE)} batches total")
        return None

def list_batches():
    """List all stored batch names."""
    return list(GLOBAL_BATCH_STORAGE.keys())

def clear_all_batches():
    """Clear all stored batches."""
    GLOBAL_BATCH_STORAGE.clear()
    print("[S4Tool] Cleared all stored batches")

def get_batch_info():
    """Get information about all stored batches."""
    info = {}
    for name, (images, masks) in GLOBAL_BATCH_STORAGE.items():
        info[name] = {
            'count': len(images),
            'image_shapes': [img.shape if hasattr(img, 'shape') else 'unknown' for img in images[:3]]  # Show first 3
        }
    return info

def debug_print_all_batches():
    """Debug function to print all stored batches."""
    print("🔍 [S4Tool-DEBUG] ==========================================")
    print(f"🔍 [S4Tool-DEBUG] STORAGE STATUS: {len(GLOBAL_BATCH_STORAGE)} batches total")
    if GLOBAL_BATCH_STORAGE:
        for name, (images, masks) in GLOBAL_BATCH_STORAGE.items():
            print(f"🔍 [S4Tool-DEBUG]   📦 Batch '{name}': {len(images)} images, {len(masks)} masks")
    else:
        print("🔍 [S4Tool-DEBUG]   ❌ NO BATCHES STORED YET")
    print("🔍 [S4Tool-DEBUG] ==========================================") 