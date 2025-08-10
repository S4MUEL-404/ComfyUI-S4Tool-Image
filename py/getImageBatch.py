import torch

class GetImageBatch:
    """
    A node that extracts a specific image and mask from an image batch using a 1-based index.
    Retrieves batches from global storage by name - no physical connections needed.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "batch_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter batch name"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "get_from_batch"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def get_from_batch(self, batch_index, batch_name):
        """
        Extract a specific image and mask from the batch using 1-based indexing.
        Retrieves the batch from global storage using the batch_name.
        """
        print(f"🟢 [GetImageBatch] EXECUTION STARTED - looking for batch: '{batch_name}', index: {batch_index}")
        
        from ..nodes_config import get_batch, list_batches
        
        # Validate batch_name
        if not batch_name or not batch_name.strip():
            available_batches = list_batches()
            error_msg = f"Batch name cannot be empty! Available batches: {available_batches}"
            print(f"🔴 [GetImageBatch] ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Debug: Print all stored batches BEFORE trying to retrieve
        print(f"🟡 [GetImageBatch] Checking current storage status...")
        from ..nodes_config import debug_print_all_batches
        debug_print_all_batches()
        
        # Retrieve batch from global storage
        print(f"🟡 [GetImageBatch] Attempting to retrieve batch '{batch_name}'...")
        batch_data = get_batch(batch_name)
        
        if batch_data is None:
            available_batches = list_batches()
            error_msg = f"Batch '{batch_name}' not found! Available batches: {available_batches}"
            print(f"🔴 [GetImageBatch] CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        images, masks = batch_data
        
        # Validate batch_index
        if batch_index < 1:
            raise ValueError(f"batch_index must be >= 1, got {batch_index}")
        
        # Convert 1-based index to 0-based
        index = batch_index - 1
        
        # Check if index is valid
        if index >= len(images):
            raise ValueError(f"Index {batch_index} out of range for batch '{batch_name}' (has {len(images)} images)")
        
        # Get the image and mask at the specified index
        selected_image = images[index]
        selected_mask = masks[index]
        
        # Debug mask format
        import torch
        print(f"[GetImageBatch] Image shape: {selected_image.shape}")
        print(f"[GetImageBatch] Mask shape: {selected_mask.shape}")
        print(f"[GetImageBatch] Mask min/max: {selected_mask.min().item():.3f}/{selected_mask.max().item():.3f}")
        
        # Ensure mask is in correct format (should be single channel, 0-1 range)
        if selected_mask.dim() == 4:  # (batch, height, width, channels)
            if selected_mask.shape[-1] == 1:
                selected_mask = selected_mask.squeeze(-1)  # Remove last dim if it's 1
            elif selected_mask.shape[-1] > 1:
                # If multi-channel, convert to grayscale
                selected_mask = torch.mean(selected_mask, dim=-1)
        elif selected_mask.dim() == 3:  # (batch, height, width)
            pass  # Already correct format
        
        # Ensure values are in 0-1 range
        if selected_mask.max() > 1.0:
            selected_mask = selected_mask / 255.0
            print(f"[GetImageBatch] Normalized mask from 0-255 to 0-1 range")
        
        print(f"🟡 [GetImageBatch] Final mask shape: {selected_mask.shape}")
        print(f"✅ [GetImageBatch] SUCCESS: Retrieved image {batch_index} from batch '{batch_name}'")
        print(f"🟢 [GetImageBatch] EXECUTION COMPLETED")
        return (selected_image, selected_mask)

def get_batch_list():
    """Helper function to get list of available batches."""
    from ..nodes_config import list_batches, get_batch_info
    batches = list_batches()
    if batches:
        info = get_batch_info()
        batch_info = [f"{name}({info[name]['count']} images)" for name in batches]
        return batch_info
    else:
        return ["No batches stored yet"]

# Node mappings
NODE_CLASS_MAPPINGS = {
    "GetImageBatch": GetImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageBatch": "💀Get Image Batch"
} 