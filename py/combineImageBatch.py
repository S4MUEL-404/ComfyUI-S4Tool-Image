import torch

class CombineImageBatch:
    """
    A node that combines two image batches into one larger batch.
    The combined batch is stored globally with a name for access by GetImageBatch nodes.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch1": ("IMAGE_BATCH",),
                "batch2": ("IMAGE_BATCH",),
                "batch_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter combined batch name (optional - leave empty for temp batch)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE_BATCH", "STRING", "INT")
    RETURN_NAMES = ("batch_combine", "status", "batch_count")
    FUNCTION = "combine_batches"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = True  # Force execution even if outputs are not connected

    def combine_batches(self, batch1, batch2, batch_name):
        """
        Combine two image batches into one.
        If batch_name is provided, stores the combined batch globally for access by GetImageBatch nodes.
        If batch_name is empty, only creates combined batch data for direct connection to other batch nodes.
        """
        print(f"🟢 [CombineImageBatch] EXECUTION STARTED - batch_name: '{batch_name}'")
        
        from ..nodes_config import store_batch
        
        # Debug batch inputs
        print(f"🟡 [CombineImageBatch] Input batch1 has {len(batch1[0])} images, batch2 has {len(batch2[0])} images")
        
        # Extract images and masks from both batches
        images1, masks1 = batch1
        images2, masks2 = batch2
        
        # Combine the lists
        combined_images = images1 + images2
        combined_masks = masks1 + masks2
        
        print(f"🟡 [CombineImageBatch] Combined batch will have {len(combined_images)} images total")
        
        # Store combined batch globally ONLY if batch_name is provided
        if batch_name and batch_name.strip():
            clean_batch_name = batch_name.strip()
            print(f"🟡 [CombineImageBatch] About to store combined batch '{clean_batch_name}'")
            try:
                store_batch(clean_batch_name, combined_images, combined_masks)
                print(f"✅ [CombineImageBatch] SUCCESS: Stored combined batch '{clean_batch_name}'")
                
                # Debug: Print all stored batches after storing
                from ..nodes_config import debug_print_all_batches
                debug_print_all_batches()
                
                status_message = f"Combined batch '{clean_batch_name}' stored successfully with {len(combined_images)} images"
                
            except Exception as e:
                print(f"🔴 [CombineImageBatch] CRITICAL ERROR storing batch: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"🟡 [CombineImageBatch] No batch name provided - creating combined batch data only (not storing)")
            status_message = f"Combined batch created (not stored) with {len(combined_images)} images"
        
        # Return combined batch for optional further connection
        combined_batch = (combined_images, combined_masks)
        
        print(f"🟢 [CombineImageBatch] EXECUTION COMPLETED - returning status: {status_message}")
        return (combined_batch, status_message, len(combined_images))

# Node mappings
NODE_CLASS_MAPPINGS = {
    "CombineImageBatch": CombineImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineImageBatch": "💀Combine Image Batch"
} 