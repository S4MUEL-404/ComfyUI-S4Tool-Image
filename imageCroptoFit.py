from .ImageOverlay import pil2tensor, tensor2pil

class ImageCropToFit:
    """
    Automatically crop the image to fit the content.
    No scaling is performed.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image, supports alpha channel
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_to_fit"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def crop_to_fit(self, image):
        from PIL import Image
        import numpy as np

        # Convert to PIL object, keep alpha channel
        pil_img = tensor2pil(image).convert('RGBA')
        arr = np.array(pil_img)
        alpha = arr[..., 3]
        # Find the boundary of the non-transparent area
        nonzero = np.argwhere(alpha > 0)
        if nonzero.size == 0:
            # Fully transparent, return the original image
            cropped = pil_img
        else:
            top_left = nonzero.min(axis=0)
            bottom_right = nonzero.max(axis=0) + 1
            cropped = pil_img.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))
        # Convert back to tensor
        return (pil2tensor(cropped),) 