from ..nodes_config import pil2tensor
import torch

class ImagePalette631:
    """
    Main color extraction with multiple algorithms and preview as three stacked color blocks.
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_image", "color1", "color2", "color3")
    FUNCTION = "extract_palette_631"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_palette_631(self, image, algorithm):
        from PIL import Image
        import numpy as np
        import cv2
        from sklearn.cluster import KMeans
        from colorsys import rgb_to_hsv

        def kmeans_colors(pixels, n=3):
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
            from ..nodes_config import tensor2pil
            pil_img = tensor2pil(image)
        small_img = pil_img.resize((128, 128))
        arr = np.array(small_img)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        h, w, _ = arr.shape
        flat_pixels = arr.reshape(-1, 3)

        if algorithm == "All Pixels":
            colors, _ = kmeans_colors(flat_pixels)
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
            if len(salient_pixels) < 3:
                salient_pixels = flat_pixels
            colors, _ = kmeans_colors(salient_pixels)
        elif algorithm == "High Saturation":
            hsv = np.array([rgb_to_hsv(r/255.0, g/255.0, b/255.0) for r, g, b in flat_pixels])
            high_sat_pixels = flat_pixels[hsv[:,1] > 0.3]
            if len(high_sat_pixels) < 3:
                high_sat_pixels = flat_pixels
            colors, _ = kmeans_colors(high_sat_pixels)
            colors = sorted(colors, key=get_saturation, reverse=True)
        elif algorithm == "Area x Saturation Weighted":
            kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
            labels = kmeans.fit_predict(flat_pixels)
            centers = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(labels)
            weights = []
            for i, c in enumerate(centers):
                s = get_saturation(c)
                weights.append(counts[i] * (0.5 + s))
            sorted_idx = np.argsort(-np.array(weights))
            colors = centers[sorted_idx]
        else:
            colors, _ = kmeans_colors(flat_pixels)

        preview_w, block_h = 400, 80
        preview_img = Image.new('RGB', (preview_w, block_h*3), (255,255,255))
        for i, color in enumerate(colors):
            if i >= 3:
                break
            block = Image.new('RGB', (preview_w, block_h), tuple(color))
            preview_img.paste(block, (0, i*block_h))
        preview_tensor = pil2tensor(preview_img)
        hex_colors = ["#%02X%02X%02X" % tuple(color) for color in colors]
        while len(hex_colors) < 3:
            hex_colors.append("")
        return (preview_tensor, *hex_colors[:3]) 