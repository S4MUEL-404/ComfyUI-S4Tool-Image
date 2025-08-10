import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from ..nodes_config import pil2tensor, tensor2pil
from colorsys import rgb_to_hsv

class ImagePalette:
    """
    Extract the five main colors of the image using different algorithms, support sorting by hue, saturation, or brightness, output a preview image with five color blocks, and five color hex strings.
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
                ], {"default": "All Pixels"}),
                "sort_mode": (["Hue", "Saturation", "Brightness"], {"default": "Hue"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("palette_image", "color1", "color2", "color3", "color4", "color5")
    FUNCTION = "extract_palette"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def extract_palette(self, image, algorithm, sort_mode):
        def kmeans_colors(pixels, n=5):
            kmeans = KMeans(n_clusters=n, n_init=5, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(labels)
            sorted_idx = np.argsort(-counts)
            return centers[sorted_idx], counts[sorted_idx]

        def get_hsv(color):
            return rgb_to_hsv(color[0]/255.0, color[1]/255.0, color[2]/255.0)

        def get_saturation(color):
            r, g, b = color
            return rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]

        pil_img = image
        if hasattr(image, 'dim'):
            pil_img = tensor2pil(image)
        small_img = pil_img.resize((128, 128))
        arr = np.array(small_img)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        h, w, _ = arr.shape
        flat_pixels = arr.reshape(-1, 3)

        if algorithm == "All Pixels":
            # KMeans on all pixels, sorted by area
            colors, _ = kmeans_colors(flat_pixels, n=5)
        elif algorithm == "Saliency Mask":
            # Saliency detection (spectral residual)
            import cv2
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
            if len(salient_pixels) < 5:
                salient_pixels = flat_pixels
            colors, _ = kmeans_colors(salient_pixels, n=5)
        elif algorithm == "High Saturation":
            # Only use high saturation pixels
            hsv = np.array([rgb_to_hsv(r/255.0, g/255.0, b/255.0) for r, g, b in flat_pixels])
            high_sat_pixels = flat_pixels[hsv[:,1] > 0.3]
            if len(high_sat_pixels) < 5:
                high_sat_pixels = flat_pixels
            colors, _ = kmeans_colors(high_sat_pixels, n=5)
        elif algorithm == "Area x Saturation Weighted":
            # KMeans on all pixels, then sort by (area * saturation)
            kmeans = KMeans(n_clusters=5, n_init=5, random_state=42)
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
            colors, _ = kmeans_colors(flat_pixels, n=5)

        # Sort colors by selected mode
        if sort_mode == "Hue":
            colors = sorted(colors, key=lambda c: get_hsv(c)[0])
        elif sort_mode == "Saturation":
            colors = sorted(colors, key=lambda c: get_hsv(c)[1], reverse=True)
        elif sort_mode == "Brightness":
            colors = sorted(colors, key=lambda c: get_hsv(c)[2], reverse=True)

        # Generate preview image: 5 color blocks vertically, each 400x80
        block_w, block_h = 400, 80
        palette_img = Image.new('RGB', (block_w, block_h * 5))
        for i, color in enumerate(colors):
            block = Image.new('RGB', (block_w, block_h), tuple(color))
            palette_img.paste(block, (0, i * block_h))
        palette_tensor = pil2tensor(palette_img)
        hex_colors = ["#%02X%02X%02X" % tuple(color) for color in colors]
        while len(hex_colors) < 5:
            hex_colors.append("")
        return (palette_tensor, *hex_colors[:5]) 