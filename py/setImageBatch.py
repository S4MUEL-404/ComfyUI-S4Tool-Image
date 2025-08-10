import torch

class SetImageBatch:
    """
    A node that creates an image batch from two images and their corresponding masks.
    The batch is stored globally with a name for access by GetImageBatch nodes.
    The output is optional and only needed for connecting to CombineImageBatch.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "batch_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter batch name (optional - leave empty for temp batch)"
                }),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE_BATCH", "STRING", "INT")
    RETURN_NAMES = ("batch_combine", "status", "batch_count")
    FUNCTION = "create_batch"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = True  # Force execution even if outputs are not connected

    def create_batch(self, image1, mask1, batch_name, image2=None, mask2=None):
        """
        Create a batch containing one or two images and their masks.
        If batch_name is provided, stores the batch globally for access by GetImageBatch nodes.
        If batch_name is empty, only creates batch data for direct connection to other batch nodes.
        """
        print(f"🟢 [SetImageBatch] EXECUTION STARTED - batch_name: '{batch_name}'")
        
        from ..nodes_config import store_batch
        
        # Start with first image and mask
        images = [image1]
        masks = [mask1]
        
        # Add second image and mask if provided
        if image2 is not None and mask2 is not None:
            images.append(image2)
            masks.append(mask2)
            print(f"🟡 [SetImageBatch] Added second image, total: {len(images)} images")
        
        # Store batch globally ONLY if batch_name is provided
        if batch_name and batch_name.strip():
            clean_batch_name = batch_name.strip()
            print(f"🟡 [SetImageBatch] About to store batch '{clean_batch_name}' with {len(images)} images")
            try:
                store_batch(clean_batch_name, images, masks)
                print(f"✅ [SetImageBatch] SUCCESS: Stored batch '{clean_batch_name}'")
                
                # Debug: Print all stored batches after storing
                from ..nodes_config import debug_print_all_batches
                debug_print_all_batches()
                
                status_message = f"Batch '{clean_batch_name}' created and stored successfully with {len(images)} images"
                
            except Exception as e:
                print(f"🔴 [SetImageBatch] CRITICAL ERROR storing batch: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"🟡 [SetImageBatch] No batch name provided - creating batch data only (not storing)")
            status_message = f"Batch created (not stored) with {len(images)} images"
        
        # Return as tuple (images_list, masks_list) for optional connection to CombineImageBatch
        batch_data = (images, masks)
        
        print(f"🟢 [SetImageBatch] EXECUTION COMPLETED - returning status: {status_message}")
        return (batch_data, status_message, len(images))

# Node mappings
NODE_CLASS_MAPPINGS = {
    "SetImageBatch": SetImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetImageBatch": "💀Set Image Batch"
} 