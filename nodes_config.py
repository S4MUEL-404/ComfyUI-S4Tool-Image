import torch
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageEnhance, ImageOps
import cv2
import io
from colorsys import rgb_to_hsv

# Import dependency manager for consistent logging
from .dependency_manager import require_dependency, S4ToolLogger

# Helper function definitions (shared with other files)
def pil2tensor(image):
    """Convert PIL Image to tensor. No fallback; raise on failure."""
    # Ensure the image is in RGB or RGBA format
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGBA')
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)

def tensor2pil(image):
    """Convert tensor to PIL Image. No fallback; raise on failure."""
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

# Optimized blend functions using only PIL and numpy
def blend_multiply(background_image, layer_image):
    """Multiply blend using PIL."""
    # Ensure same size
    if background_image.size != layer_image.size:
        layer_image = layer_image.resize(background_image.size, Image.LANCZOS)
    return ImageChops.multiply(background_image.convert('RGB'), layer_image.convert('RGB'))

def blend_screen(background_image, layer_image):
    """Screen blend using PIL."""
    if background_image.size != layer_image.size:
        layer_image = layer_image.resize(background_image.size, Image.LANCZOS)
    return ImageChops.screen(background_image.convert('RGB'), layer_image.convert('RGB'))

def blend_overlay(background_image, layer_image):
    """Overlay blend using PIL."""
    if background_image.size != layer_image.size:
        layer_image = layer_image.resize(background_image.size, Image.LANCZOS)
    return ImageChops.overlay(background_image.convert('RGB'), layer_image.convert('RGB'))

def blend_soft_light(background_image, layer_image):
    """Soft light blend using PIL."""
    if background_image.size != layer_image.size:
        layer_image = layer_image.resize(background_image.size, Image.LANCZOS)
    return ImageChops.soft_light(background_image.convert('RGB'), layer_image.convert('RGB'))


# Professional Color Extraction using multiple algorithms
class ColorExtractor:
    """Professional color extraction using K-Means, median cut, and other proven algorithms."""
    
    @staticmethod
    def filter_transparent_pixels(pixels):
        """Filter out transparent and invalid pixels."""
        flat_pixels = pixels.reshape(-1, pixels.shape[-1])
        original_count = len(flat_pixels)
        
        print(f"[ColorExtractor] Starting with {original_count} pixels")
        
        # Create mask for valid pixels
        valid_mask = np.ones(len(flat_pixels), dtype=bool)
        filters_applied = []
        
        # 1. Filter transparency markers (254,254,254)
        transparency_mask = np.all(flat_pixels == [254, 254, 254], axis=1)
        if np.any(transparency_mask):
            valid_mask &= ~transparency_mask
            filters_applied.append(f"transparency markers: {np.sum(transparency_mask)}")
        
        # 2. Filter neutral gray (128,128,128)
        neutral_mask = np.all(flat_pixels == [128, 128, 128], axis=1)
        if np.any(neutral_mask):
            valid_mask &= ~neutral_mask
            filters_applied.append(f"neutral gray: {np.sum(neutral_mask)}")
        
        # 3. Filter excessive black (only if it's more than 40% and we have other colors)
        black_mask = np.all(flat_pixels == [0, 0, 0], axis=1)
        black_count = np.sum(black_mask)
        if black_count > 0 and black_count > len(flat_pixels) * 0.4 and np.sum(valid_mask) - black_count > 10:
            valid_mask &= ~black_mask
            filters_applied.append(f"excessive black: {black_count}")
        
        filtered_pixels = flat_pixels[valid_mask]
        
        if filters_applied:
            print(f"[ColorExtractor] Filtered out: {', '.join(filters_applied)}")
            print(f"[ColorExtractor] {len(filtered_pixels)} valid pixels remaining")
        
        return filtered_pixels
    
    @staticmethod
    def kmeans_colors(pixels, n_colors=5, max_iter=20):
        """Extract dominant colors using K-Means clustering."""
        require_dependency('sklearn', 'K-Means Color Clustering')
        from sklearn.cluster import KMeans
        
        filtered_pixels = ColorExtractor.filter_transparent_pixels(pixels)
        
        if len(filtered_pixels) < n_colors:
            S4ToolLogger.warning("ColorExtractor", f"Too few valid pixels ({len(filtered_pixels)}) for K-Means clustering")
            return ColorExtractor.simple_dominant_colors(pixels, n_colors)
        
        # Normalize pixel values to 0-1 for better clustering
        normalized_pixels = filtered_pixels.astype(np.float32) / 255.0
        
        # Use K-Means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, max_iter=max_iter, n_init=10)
        kmeans.fit(normalized_pixels)
        
        # Get cluster centers and convert back to RGB
        centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)
        
        # Calculate weights based on cluster sizes
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        S4ToolLogger.info("ColorExtractor", f"K-Means extracted {len(centers)} colors from {len(filtered_pixels)} pixels")
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        dominant_colors = centers[unique_labels[sorted_indices]]
        dominant_counts = counts[sorted_indices]
        
        return dominant_colors, dominant_counts
    
    @staticmethod
    def median_cut_colors(pixels, n_colors=5):
        """Extract colors using median cut algorithm (similar to PIL's quantize)."""
        filtered_pixels = ColorExtractor.filter_transparent_pixels(pixels)
        
        if len(filtered_pixels) < n_colors:
            return ColorExtractor.simple_dominant_colors(pixels, n_colors)
        
        def median_cut_recursive(pixel_list, depth):
            if depth == 0 or len(pixel_list) <= 1:
                # Return the average color of this group
                if len(pixel_list) > 0:
                    return [np.mean(pixel_list, axis=0).astype(np.uint8)]
                else:
                    return []
            
            # Find the dimension with the largest range
            ranges = np.max(pixel_list, axis=0) - np.min(pixel_list, axis=0)
            split_dim = np.argmax(ranges)
            
            # Sort pixels by the dimension with largest range
            sorted_pixels = pixel_list[pixel_list[:, split_dim].argsort()]
            
            # Split at median
            median_idx = len(sorted_pixels) // 2
            
            # Recursively process both halves
            left_colors = median_cut_recursive(sorted_pixels[:median_idx], depth - 1)
            right_colors = median_cut_recursive(sorted_pixels[median_idx:], depth - 1)
            
            return left_colors + right_colors
        
        # Calculate depth needed for n_colors
        depth = int(np.ceil(np.log2(n_colors)))
        colors = median_cut_recursive(filtered_pixels, depth)
        
        # Ensure we have exactly n_colors
        while len(colors) < n_colors and len(colors) > 0:
            colors.append(colors[0])  # Duplicate most frequent
        
        colors = colors[:n_colors]
        
        print(f"[ColorExtractor] Median cut extracted {len(colors)} colors")
        
        # Calculate frequency for each color
        color_counts = []
        for color in colors:
            distances = np.sum((filtered_pixels - color) ** 2, axis=1)
            closest_count = np.sum(distances < 1000)  # Threshold for "close" colors
            color_counts.append(closest_count)
        
        return np.array(colors), np.array(color_counts)
    
    @staticmethod
    def simple_dominant_colors(pixels, n_colors=5):
        """Simple frequency-based color extraction (fallback method)."""
        filtered_pixels = ColorExtractor.filter_transparent_pixels(pixels)
        
        if len(filtered_pixels) == 0:
            # Return default palette if no valid pixels
            return np.array([
                [70, 130, 180],   # Steel Blue
                [255, 69, 0],     # Orange Red  
                [50, 205, 50],    # Lime Green
                [255, 215, 0],    # Gold
                [186, 85, 211]    # Medium Orchid
            ][:n_colors]), np.ones(n_colors)
        
        # Quantize colors to reduce noise (group similar colors)
        quantized = (filtered_pixels // 8) * 8  # Reduce precision
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_idx = np.argsort(-counts)
        dominant_colors = unique_colors[sorted_idx[:n_colors]]
        dominant_counts = counts[sorted_idx[:n_colors]]
        
        # Pad if needed
        if len(dominant_colors) < n_colors:
            padding_needed = n_colors - len(dominant_colors)
            if len(dominant_colors) > 0:
                # Repeat the most frequent color
                for _ in range(padding_needed):
                    dominant_colors = np.vstack([dominant_colors, dominant_colors[0]])
                    dominant_counts = np.append(dominant_counts, dominant_counts[0])
            else:
                return ColorExtractor.simple_dominant_colors(pixels.reshape(1, -1, 3), n_colors)
        
        print(f"[ColorExtractor] Simple extraction: {len(unique_colors)} unique -> {len(dominant_colors)} final")
        
        return dominant_colors[:n_colors], dominant_counts[:n_colors]
    
    @staticmethod
    def get_saturation(color):
        """Get saturation value of a color."""
        from colorsys import rgb_to_hsv
        r, g, b = color
        return rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
    
    @staticmethod
    def get_saturation(color):
        """Get saturation value of a color."""
        from colorsys import rgb_to_hsv
        r, g, b = color
        return rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
    
    @staticmethod
    def get_hsv(color):
        """Get HSV values of a color."""
        from colorsys import rgb_to_hsv
        return rgb_to_hsv(color[0]/255.0, color[1]/255.0, color[2]/255.0)
    
    @staticmethod
    def extract_colors_by_algorithm(image_array, algorithm, n_colors=5):
        """Extract colors using professional algorithms with transparency awareness."""
        print(f"[ColorExtractor] Using algorithm '{algorithm}' to extract {n_colors} colors")
        
        if algorithm == "K-Means Clustering":
            colors, _ = ColorExtractor.kmeans_colors(image_array, n_colors)
            
        elif algorithm == "Median Cut":
            colors, _ = ColorExtractor.median_cut_colors(image_array, n_colors)
            
        elif algorithm == "Histogram Frequency":
            colors, _ = ColorExtractor.simple_dominant_colors(image_array, n_colors)
            
        elif algorithm == "High Saturation":
            # Filter by saturation first, then extract
            filtered_pixels = ColorExtractor.filter_transparent_pixels(image_array)
            
            if len(filtered_pixels) > 0:
                from colorsys import rgb_to_hsv
                high_sat_pixels = []
                for pixel in filtered_pixels:
                    r, g, b = pixel / 255.0
                    h, s, v = rgb_to_hsv(r, g, b)
                    if s > 0.4:  # High saturation threshold
                        high_sat_pixels.append(pixel)
                
                print(f"[ColorExtractor] Found {len(high_sat_pixels)} high saturation pixels")
                if len(high_sat_pixels) >= n_colors:
                    high_sat_array = np.array(high_sat_pixels).reshape(1, -1, 3)
                    colors, _ = ColorExtractor.kmeans_colors(high_sat_array, n_colors)
                else:
                    colors, _ = ColorExtractor.kmeans_colors(image_array, n_colors)
            else:
                colors, _ = ColorExtractor.kmeans_colors(image_array, n_colors)
            
        elif algorithm == "Center Weighted":
            # Extract colors from center region with more weight
            h, w = image_array.shape[:2]
            center_h, center_w = h // 2, w // 2
            radius_h, radius_w = h // 4, w // 4
            
            # Extract center region
            center_region = image_array[
                max(0, center_h - radius_h):min(h, center_h + radius_h),
                max(0, center_w - radius_w):min(w, center_w + radius_w)
            ]
            
            if center_region.size > 0:
                colors, _ = ColorExtractor.kmeans_colors(center_region, n_colors)
            else:
                colors, _ = ColorExtractor.kmeans_colors(image_array, n_colors)
                
        else:
            # Default to K-Means for best results
            colors, _ = ColorExtractor.kmeans_colors(image_array, n_colors)
        
        print(f"[ColorExtractor] Final colors: {[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in colors[:min(3, len(colors))]]}")
        return colors
    
    @staticmethod
    def sort_colors(colors, sort_mode="Hue"):
        """Sort colors by specified mode."""
        if sort_mode == "Hue":
            return sorted(colors, key=lambda c: ColorExtractor.get_hsv(c)[0])
        elif sort_mode == "Saturation":
            return sorted(colors, key=lambda c: ColorExtractor.get_hsv(c)[1], reverse=True)
        elif sort_mode == "Brightness":
            return sorted(colors, key=lambda c: ColorExtractor.get_hsv(c)[2], reverse=True)
        return colors

# Constants for consistent sizing
class ImageConstants:
    """Constants for consistent image processing."""
    PALETTE_BLOCK_WIDTH = 400
    PALETTE_BLOCK_HEIGHT = 80
    COLOR_EXTRACTION_RESIZE = (128, 128)
    MAX_IMAGE_SIZE = 4096

# Professional-grade image processing using OpenCV + PIL
class ImageUtils:
    """Utilities for high-quality image processing using OpenCV and PIL."""
    
    @staticmethod
    def ensure_rgba(pil_image):
        """Ensure image is in RGBA format."""
        if pil_image.mode == 'RGBA':
            return pil_image
        elif pil_image.mode == 'RGB':
            return pil_image.convert('RGBA')
        elif pil_image.mode in ['P', 'LA', 'PA']:
            return pil_image.convert('RGBA')
        elif pil_image.mode == 'L':
            return pil_image.convert('RGBA')
        else:
            return pil_image.convert('RGBA')
    
    @staticmethod
    def pil_to_cv2(pil_image):
        """Convert PIL Image to OpenCV format (BGR + Alpha)."""
        if pil_image.mode == 'RGBA':
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        elif pil_image.mode == 'RGB':
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Convert to RGBA first
            rgba_image = pil_image.convert('RGBA')
            cv_image = cv2.cvtColor(np.array(rgba_image), cv2.COLOR_RGBA2BGRA)
        return cv_image
    
    @staticmethod
    def cv2_to_pil(cv_image):
        """Convert OpenCV format back to PIL Image."""
        if cv_image.shape[2] == 4:  # BGRA
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgb_image, 'RGBA')
        elif cv_image.shape[2] == 3:  # BGR
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image, 'RGB')
        else:  # Grayscale
            return Image.fromarray(cv_image, 'L')
    
    @staticmethod
    def get_pil_resampling(method_name):
        """Get PIL resampling method from name."""
        methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "area": Image.BOX,
            "lanczos": Image.LANCZOS,
            "lanczos3": Image.LANCZOS,
            "mitchell": Image.LANCZOS,  # Use LANCZOS as approximation
            "catrom": Image.LANCZOS,    # Use LANCZOS as approximation
        }
        return methods.get(method_name, Image.LANCZOS)
    
    @staticmethod
    def high_quality_resize(pil_image, target_width, target_height, interpolation="lanczos"):
        """High-quality image resize using PIL."""
        resampling = ImageUtils.get_pil_resampling(interpolation)
        return pil_image.resize((target_width, target_height), resampling)
    
    @staticmethod
    def smart_resize_with_method(pil_image, target_width, target_height, method, interpolation="lanczos"):
        """Smart resize with different scaling methods using PIL."""
        current_width, current_height = pil_image.size
        resampling = ImageUtils.get_pil_resampling(interpolation)
        
        if method == "stretch":
            # Direct resize to target dimensions
            return pil_image.resize((target_width, target_height), resampling)
            
        elif method == "keep proportion":
            # Scale to fit within target dimensions while maintaining aspect ratio
            scale_w = target_width / current_width
            scale_h = target_height / current_height
            scale = min(scale_w, scale_h)
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            return pil_image.resize((new_width, new_height), resampling)
            
        elif method == "fill / crop":
            # Scale to fill target dimensions, then crop center
            scale_w = target_width / current_width
            scale_h = target_height / current_height
            scale = max(scale_w, scale_h)
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize first
            resized = pil_image.resize((new_width, new_height), resampling)
            
            # Calculate crop box to center the image
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            return resized.crop((left, top, right, bottom))
            
        elif method == "pad":
            # Scale to fit within target dimensions, then pad with transparency
            scale_w = target_width / current_width
            scale_h = target_height / current_height
            scale = min(scale_w, scale_h)
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize first
            resized = pil_image.resize((new_width, new_height), resampling)
            
            # Create transparent background
            if pil_image.mode == 'RGBA':
                background = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            else:
                background = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
                resized = resized.convert('RGBA')
            
            # Calculate position to center the resized image
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            # Paste the resized image onto the background
            background.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)
            return background
        
        else:
            # Default to stretch
            return pil_image.resize((target_width, target_height), resampling)
    
    @staticmethod
    def apply_transforms_premium(pil_image, scale=1.0, rotation=0.0, opacity=1.0):
        """Apply transformations using scikit-image with center-based transforms only. No fallback."""
        require_dependency('skimage', 'Professional Image Transforms')
        S4ToolLogger.info("ImageUtils", f"Applying professional transforms - scale={scale}, rotation={rotation}°, opacity={opacity}")
        result = ImageUtils._apply_center_transforms_skimage(pil_image, scale, rotation, opacity)
        if result is None:
            raise RuntimeError("High-quality transform failed with scikit-image")
        return result
    
    @staticmethod
    def _apply_center_transforms_skimage(pil_image, scale=1.0, rotation=0.0, opacity=1.0):
        """Perfect center-based transformations with guaranteed no cropping (scikit-image only)."""
        rgba_image = ImageUtils.ensure_rgba(pil_image)
        w, h = rgba_image.size
        print(f"[ImageUtils] Input: {w}x{h}, scale={scale}, rotation={rotation}°, opacity={opacity}")
        # Always use the high-quality scikit-image pipeline; do not fallback to PIL
        result = ImageUtils._skimage_transform_corrected(rgba_image, scale, rotation, opacity)
        if result is None:
            return None
        S4ToolLogger.success("ImageUtils", f"Professional transform completed: {w}x{h} -> {result.size}")
        return result
    
    @staticmethod
    def _skimage_transform_corrected(pil_image, scale, rotation, opacity):
        """Corrected scikit-image transformation with proper center handling."""
        from skimage.transform import warp, AffineTransform
        
        img_array = np.array(pil_image, dtype=np.float64) / 255.0
        h, w = img_array.shape[:2]
        
        # Calculate exact output dimensions after rotation and scaling
        if rotation != 0:
            rad = np.deg2rad(abs(rotation))
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            # Rotated bounding box
            rw = w * cos_r + h * sin_r
            rh = w * sin_r + h * cos_r
        else:
            rw, rh = w, h
        
        # Apply scaling
        out_w = int(np.ceil(rw * abs(scale)))
        out_h = int(np.ceil(rh * abs(scale)))
        
        # Ensure even dimensions for better centering
        if out_w % 2 != w % 2:
            out_w += 1
        if out_h % 2 != h % 2:
            out_h += 1
        
        # Calculate precise centers
        in_cx, in_cy = w / 2.0, h / 2.0
        out_cx, out_cy = out_w / 2.0, out_h / 2.0
        
        # Create a single transformation matrix for center-based rotation and scaling
        # We need to: translate to origin -> scale -> rotate -> translate to output center
        
        # Manual matrix composition for precise control
        from skimage.transform import AffineTransform
        
        # Create individual transformation matrices
        cos_r = np.cos(np.deg2rad(rotation))
        sin_r = np.sin(np.deg2rad(rotation))
        
        # Combined transformation matrix: scale * rotation * translation
        # This ensures the transformation happens around the center
        transform_matrix = np.array([
            [scale * cos_r, -scale * sin_r, out_cx - scale * (cos_r * in_cx - sin_r * in_cy)],
            [scale * sin_r,  scale * cos_r, out_cy - scale * (sin_r * in_cx + cos_r * in_cy)],
            [0,              0,             1]
        ])
        
        final_transform = AffineTransform(matrix=transform_matrix)
        
        # Debug: verify center point transformation
        center_test = transform_matrix @ np.array([in_cx, in_cy, 1])
        S4ToolLogger.info("ImageUtils", f"Center transform verification: ({in_cx:.1f},{in_cy:.1f}) -> ({center_test[0]:.1f},{center_test[1]:.1f}), expected: ({out_cx:.1f},{out_cy:.1f})")
        
        # Apply with maximum quality
        transformed = warp(
            img_array,
            final_transform.inverse,
            output_shape=(out_h, out_w),
            order=5,  # Maximum quality
            mode='constant',
            cval=0.0,
            preserve_range=True,
            clip=False
        )
        
        # Apply opacity
        if opacity < 1.0:
            transformed[:, :, 3] *= opacity
        
        # Verify we have content
        if np.sum(transformed[:, :, 3] > 0.001) == 0:
            return None  # Failed, will fallback to PIL
        
        # Convert back
        result_array = np.clip(transformed * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array, 'RGBA')
    
    
    @staticmethod
    def advanced_composite(background, overlay, x=0, y=0, blend_mode='normal'):
        """Advanced image compositing with perfect alpha handling - no transparency artifacts."""
        # Ensure both images are RGBA
        bg = ImageUtils.ensure_rgba(background)
        ovl = ImageUtils.ensure_rgba(overlay)
        
        print(f"[ImageUtils] Compositing with mode '{blend_mode}' at position ({x}, {y})")
        
        # Handle position boundaries  
        bg_width, bg_height = bg.size
        ovl_width, ovl_height = ovl.size
        
        # Calculate visible region
        left = max(0, x)
        top = max(0, y)
        right = min(bg_width, x + ovl_width)
        bottom = min(bg_height, y + ovl_height)
        
        if right <= left or bottom <= top:
            return bg  # No visible area, return unchanged background
        
        # Calculate overlay crop region if needed
        ovl_left = max(0, -x)
        ovl_top = max(0, -y)
        ovl_right = ovl_left + (right - left)
        ovl_bottom = ovl_top + (bottom - top)
        
        # Crop overlay to visible region
        if ovl_left > 0 or ovl_top > 0 or ovl_right < ovl_width or ovl_bottom < ovl_height:
            ovl = ovl.crop((ovl_left, ovl_top, ovl_right, ovl_bottom))
        
        # Convert to numpy arrays for precise alpha compositing
        bg_array = np.array(bg, dtype=np.float64) / 255.0
        ovl_array = np.array(ovl, dtype=np.float64) / 255.0
        
        # Create result array as copy of background
        result_array = bg_array.copy()
        
        # Extract region to composite onto
        bg_region = result_array[top:bottom, left:right, :]
        
        # Perform precise alpha blending based on blend mode
        if blend_mode == 'over' or blend_mode == 'normal':
            # Standard alpha over blending
            ImageUtils._alpha_blend_arrays(bg_region, ovl_array)
        else:
            # Special blend modes with proper alpha handling
            ImageUtils._blend_with_mode(bg_region, ovl_array, blend_mode)
        
        # Update the region in result
        result_array[top:bottom, left:right, :] = bg_region
        
        # Convert back to PIL
        result_uint8 = np.clip(result_array * 255.0, 0, 255).astype(np.uint8)
        result = Image.fromarray(result_uint8, 'RGBA')
        
        print(f"[ImageUtils] Composite completed successfully")
        return result
    
    @staticmethod
    def _alpha_blend_arrays(bg_region, ovl_array):
        """Perform precise alpha blending using Porter-Duff 'over' operator."""
        # Extract alpha channels
        bg_alpha = bg_region[:, :, 3:4]
        ovl_alpha = ovl_array[:, :, 3:4]
        
        # Calculate composite alpha
        comp_alpha = ovl_alpha + bg_alpha * (1.0 - ovl_alpha)
        
        # Avoid division by zero
        safe_alpha = np.where(comp_alpha > 0.0001, comp_alpha, 1.0)
        
        # Calculate composite RGB
        for c in range(3):  # RGB channels
            bg_region[:, :, c:c+1] = (
                ovl_array[:, :, c:c+1] * ovl_alpha + 
                bg_region[:, :, c:c+1] * bg_alpha * (1.0 - ovl_alpha)
            ) / safe_alpha
        
        # Update alpha channel
        bg_region[:, :, 3:4] = comp_alpha
    
    @staticmethod
    def _blend_with_mode(bg_region, ovl_array, blend_mode):
        """Apply blend modes with correct formulas and proper alpha composition."""
        # Extract RGB and alpha channels
        bg_rgb = bg_region[:, :, :3]
        bg_alpha = bg_region[:, :, 3:4]
        ovl_rgb = ovl_array[:, :, :3]
        ovl_alpha = ovl_array[:, :, 3:4]
        
        print(f"[ImageUtils] Applying blend mode: {blend_mode}")
        
        # Apply the correct blend formula
        if blend_mode == 'multiply':
            # Multiply: C = A * B
            blended_rgb = bg_rgb * ovl_rgb
        elif blend_mode == 'screen':
            # Screen: C = 1 - (1-A) * (1-B)
            blended_rgb = 1.0 - (1.0 - bg_rgb) * (1.0 - ovl_rgb)
        elif blend_mode == 'overlay':
            # Overlay: if A < 0.5: C = 2*A*B, else: C = 1 - 2*(1-A)*(1-B)
            mask = bg_rgb < 0.5
            blended_rgb = np.where(
                mask,
                2.0 * bg_rgb * ovl_rgb,
                1.0 - 2.0 * (1.0 - bg_rgb) * (1.0 - ovl_rgb)
            )
        elif blend_mode == 'soft-light':
            # Photoshop-style soft light
            mask = ovl_rgb <= 0.5
            g = np.where(bg_rgb <= 0.25, 
                        ((16.0 * bg_rgb - 12.0) * bg_rgb + 4.0) * bg_rgb, 
                        np.sqrt(bg_rgb))
            blended_rgb = np.where(
                mask,
                bg_rgb - (1.0 - 2.0 * ovl_rgb) * bg_rgb * (1.0 - bg_rgb),
                bg_rgb + (2.0 * ovl_rgb - 1.0) * (g - bg_rgb)
            )
        elif blend_mode == 'hard-light':
            # Hard light: if B < 0.5: C = 2*A*B, else: C = 1 - 2*(1-A)*(1-B)
            mask = ovl_rgb < 0.5
            blended_rgb = np.where(
                mask,
                2.0 * bg_rgb * ovl_rgb,
                1.0 - 2.0 * (1.0 - bg_rgb) * (1.0 - ovl_rgb)
            )
        elif blend_mode == 'colour-dodge' or blend_mode == 'color-dodge':
            # Color dodge: C = A / (1 - B)
            blended_rgb = np.where(
                ovl_rgb >= 1.0 - 1e-10,
                1.0,
                np.clip(bg_rgb / np.maximum(1.0 - ovl_rgb, 1e-10), 0.0, 1.0)
            )
        elif blend_mode == 'colour-burn' or blend_mode == 'color-burn':
            # Color burn: C = 1 - (1 - A) / B
            blended_rgb = np.where(
                ovl_rgb <= 1e-10,
                0.0,
                np.clip(1.0 - (1.0 - bg_rgb) / np.maximum(ovl_rgb, 1e-10), 0.0, 1.0)
            )
        elif blend_mode == 'darken':
            # Darken: C = min(A, B)
            blended_rgb = np.minimum(bg_rgb, ovl_rgb)
        elif blend_mode == 'lighten':
            # Lighten: C = max(A, B)
            blended_rgb = np.maximum(bg_rgb, ovl_rgb)
        elif blend_mode == 'difference':
            # Difference: C = |A - B|
            blended_rgb = np.abs(bg_rgb - ovl_rgb)
        elif blend_mode == 'exclusion':
            # Exclusion: C = A + B - 2*A*B
            blended_rgb = bg_rgb + ovl_rgb - 2.0 * bg_rgb * ovl_rgb
        elif blend_mode == 'add':
            # Add: C = A + B (clipped)
            blended_rgb = np.clip(bg_rgb + ovl_rgb, 0.0, 1.0)
        elif blend_mode == 'subtract':
            # Subtract: C = A - B (clipped)
            blended_rgb = np.clip(bg_rgb - ovl_rgb, 0.0, 1.0)
        else:
            # Fallback to normal
            blended_rgb = ovl_rgb
        
        # Ensure blended result is in valid range
        blended_rgb = np.clip(blended_rgb, 0.0, 1.0)
        
        # Create composite using proper alpha blending
        # Result alpha: αr = αo + αb * (1 - αo)
        result_alpha = ovl_alpha + bg_alpha * (1.0 - ovl_alpha)
        
        # Avoid division by zero
        safe_alpha = np.where(result_alpha > 1e-10, result_alpha, 1.0)
        
        # Composite RGB: Cr = (Co * αo + Cb * αb * (1 - αo)) / αr
        # But we want to blend Co with the background, so:
        # Cr = (blended_rgb * αo + Cb * αb * (1 - αo)) / αr
        result_rgb = (blended_rgb * ovl_alpha + bg_rgb * bg_alpha * (1.0 - ovl_alpha)) / safe_alpha
        
        # Update the background region
        bg_region[:, :, :3] = result_rgb
        bg_region[:, :, 3:4] = result_alpha


# Base class for color extraction nodes to reduce code duplication
class BaseColorExtractionNode:
    """Base class for color extraction nodes with common functionality."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_common_input_types():
        """Get common input types for color extraction nodes."""
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": ([
                    "K-Means Clustering",
                    "Median Cut", 
                    "Histogram Frequency",
                    "High Saturation",
                    "Center Weighted"
                ], {"default": "K-Means Clustering"})
            }
        }
    
    def prepare_image_for_processing(self, image):
        """Prepare image for color extraction processing with proper alpha handling."""
        pil_img = image
        if hasattr(image, 'dim'):
            pil_img = tensor2pil(image)
        
        # Resize for processing efficiency
        small_img = pil_img.resize(ImageConstants.COLOR_EXTRACTION_RESIZE)
        arr = np.array(small_img)
        
        print(f"[ColorExtractor] Input image shape: {arr.shape}, mode: {small_img.mode}")
        
        # Handle transparency properly
        if arr.shape[2] == 4:  # RGBA image
            print(f"[ColorExtractor] Processing RGBA image with transparency handling")
            # Extract alpha channel
            alpha = arr[..., 3]
            rgb = arr[..., :3]
            
            # Use a more strict threshold for transparency (alpha < 10 is considered transparent)
            transparent_mask = alpha < 10  # Pixels with very low alpha are transparent
            opaque_mask = ~transparent_mask
            
            total_pixels = alpha.size
            transparent_pixels = np.sum(transparent_mask)
            opaque_pixels = np.sum(opaque_mask)
            
            print(f"[ColorExtractor] Pixel analysis: {total_pixels} total, {transparent_pixels} transparent, {opaque_pixels} opaque")
            
            if opaque_pixels > 0:
                # Create a new array with only opaque pixels for processing
                # Instead of replacing transparent areas, we'll work only with non-transparent data
                processed_arr = rgb.copy()
                
                # Mark transparent pixels with a special marker that will be filtered out later
                # Using a very specific RGB value that's unlikely to appear naturally
                processed_arr[transparent_mask] = [254, 254, 254]  # Near-white marker
                
                print(f"[ColorExtractor] Marked {transparent_pixels} transparent pixels for exclusion")
                
                # Also store the transparency mask for later use
                self._transparency_mask = transparent_mask
                
                return processed_arr
            else:
                print(f"[ColorExtractor] Warning: Image is completely transparent, creating fallback data")
                # Create a simple fallback with basic colors
                processed_arr = rgb.copy()
                processed_arr[:] = [100, 100, 100]  # Dark gray fallback
                return processed_arr
        else:
            # RGB image - no transparency handling needed
            print(f"[ColorExtractor] Processing RGB image, no transparency handling needed")
            self._transparency_mask = None
            return arr
    
    def extract_colors_with_algorithm(self, image_array, algorithm, n_colors):
        """Extract colors using specified algorithm."""
        colors = ColorExtractor.extract_colors_by_algorithm(image_array, algorithm, n_colors)
        
        # Special handling for High Saturation algorithm
        if algorithm == "High Saturation":
            colors = ColorExtractor.sort_colors(colors, "Saturation")
        
        return colors
    
    def create_color_preview(self, colors, block_width, block_height, num_colors):
        """Create a preview image with color blocks."""
        from PIL import Image
        
        preview_img = Image.new('RGB', (block_width, block_height * num_colors))
        for i, color in enumerate(colors[:num_colors]):
            block = Image.new('RGB', (block_width, block_height), tuple(color))
            preview_img.paste(block, (0, i * block_height))
        
        return preview_img
    
    def colors_to_hex(self, colors, num_colors):
        """Convert colors to hex strings."""
        hex_colors = ["#%02X%02X%02X" % tuple(color) for color in colors]
        while len(hex_colors) < num_colors:
            hex_colors.append("")
        return hex_colors[:num_colors]