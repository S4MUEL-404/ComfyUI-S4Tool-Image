from .ImageOverlay import pil2tensor
import torch

class ImagePrimaryColor:
    """
    Extract the primary color of the image using different algorithms. Output a preview image (400x80) and the hex string of the primary color.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": ([
                    "All Pixels",
                    "Saliency Mask",
                    "High Saturation",
                    "Area x Saturation Weighted"
                ], {"default": "All Pixels"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "color")
    FUNCTION = "extract_primary_color"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_primary_color(self, image, algorithm):
        from PIL import Image
        import numpy as np
        import cv2
        from sklearn.cluster import KMeans
        from colorsys import rgb_to_hsv

        def kmeans_colors(pixels, n=1):
            kmeans = KMeans(n_clusters=n, n_init=5, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(labels)
            sorted_idx = np.argsort(-counts)
            return centers[sorted_idx], counts[sorted_idx]

        def get_saturation(color):
            r, g, b = color
            return rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]

        pil_img = image
        if hasattr(image, 'dim'):
            from .ImageOverlay import tensor2pil
            pil_img = tensor2pil(image)
        small_img = pil_img.resize((128, 128))
        arr = np.array(small_img)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        h, w, _ = arr.shape
        flat_pixels = arr.reshape(-1, 3)

        if algorithm == "All Pixels":
            colors, _ = kmeans_colors(flat_pixels, n=1)
        elif algorithm == "Saliency Mask":
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            gray = np.float32(gray)
            dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude, angle = cv2.cartToPolar(dft_shift[..., 0], dft_shift[..., 1])
            log_mag = np.log(magnitude + 1)
            avg_log_mag = cv2.blur(log_mag, (3, 3))
            spectral_residual = np.exp(log_mag - avg_log_mag)
            real, imag = cv2.polarToCart(spectral_residual, angle)
            dft_shift[..., 0] = real
            dft_shift[..., 1] = imag
            idft_shift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(idft_shift)
            saliency_map = cv2.magnitude(img_back[..., 0], img_back[..., 1])
            saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 2.5)
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            mask = saliency_map > 0.5
            salient_pixels = arr[mask]
            if len(salient_pixels) < 1:
                salient_pixels = flat_pixels
            colors, _ = kmeans_colors(salient_pixels, n=1)
        elif algorithm == "High Saturation":
            hsv = np.array([rgb_to_hsv(r/255.0, g/255.0, b/255.0) for r, g, b in flat_pixels])
            high_sat_pixels = flat_pixels[hsv[:,1] > 0.3]
            if len(high_sat_pixels) < 1:
                high_sat_pixels = flat_pixels
            colors, _ = kmeans_colors(high_sat_pixels, n=1)
            colors = sorted(colors, key=get_saturation, reverse=True)
        elif algorithm == "Area x Saturation Weighted":
            kmeans = KMeans(n_clusters=1, n_init=5, random_state=42)
            labels = kmeans.fit_predict(flat_pixels)
            centers = kmeans.cluster_centers_.astype(int)
            colors = centers
        else:
            colors, _ = kmeans_colors(flat_pixels, n=1)

        # Only one color
        color = tuple(colors[0])
        preview_w, block_h = 400, 80
        preview_img = Image.new('RGB', (preview_w, block_h), color)
        preview_tensor = pil2tensor(preview_img)
        hex_color = "#%02X%02X%02X" % color
        return (preview_tensor, hex_color) 