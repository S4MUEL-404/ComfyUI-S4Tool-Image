import base64
from io import BytesIO
from .ImageOverlay import pil2tensor, tensor2pil

class ImageToBase64:
    """
    Node to convert image to base64 string.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_base64"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = True

    def to_base64(self, image):

        pil_image = tensor2pil(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return {"result": (base64_str,)} 
