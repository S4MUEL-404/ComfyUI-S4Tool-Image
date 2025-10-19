import torch
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from PIL import Image

from ..nodes_config import pil2tensor, tensor2pil, ColorExtractor, ImageConstants, BaseColorExtractionNode
from ..dependency_manager import S4ToolLogger

class ImagePaletteContrary(BaseColorExtractionNode):
    """
    Extract main color using K-Means and find the color with maximum brightness contrast.
    Returns primary color (K-Means main color) and contrary color (maximum brightness contrast).
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color1_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Adjust color1 intensity: positive makes bright colors brighter/dark colors darker, negative reverses"
                }),
                "color2_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Adjust color2 intensity: positive makes bright colors brighter/dark colors darker, negative reverses"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Adjust color saturation: positive increases saturation, negative decreases saturation"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview_image", "color1", "color2")
    FUNCTION = "extract_contrary_palette"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    @staticmethod
    def get_brightness(color):
        """Calculate brightness using relative luminance formula"""
        r, g, b = color
        # Using ITU-R BT.709 luma coefficients for perceptual brightness
        return 0.2126 * (r / 255.0) + 0.7152 * (g / 255.0) + 0.0722 * (b / 255.0)
    
    @staticmethod
    def get_brightness_hsv(color):
        """Calculate brightness using HSV value component"""
        r, g, b = color
        h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return v
    
    @staticmethod
    def calculate_dynamic_multiplier_hsl(primary_l, contrary_l, is_contrary_color=False):
        """
        Calculate dynamic multiplier based on HSL L-value ranges and their contrast.
        The more extreme the colors already are, the smaller the adjustment needed.
        """
        # Base multiplier
        base_multiplier = 120 if is_contrary_color else 100
        
        # Calculate the L-value contrast achieved
        l_contrast = abs(primary_l - contrary_l)
        
        # Determine range factors based on individual L-values
        if primary_l >= 0.85:  # Very bright primary
            primary_factor = 0.3  # Small adjustments
            primary_name = "very bright"
        elif primary_l >= 0.65:  # Bright primary
            primary_factor = 0.7
            primary_name = "bright"
        elif primary_l >= 0.35:  # Medium primary
            primary_factor = 1.0
            primary_name = "medium"
        elif primary_l >= 0.15:  # Dark primary
            primary_factor = 0.8
            primary_name = "dark"
        else:  # Very dark primary
            primary_factor = 0.4
            primary_name = "very dark"
        
        # Adjust factor based on achieved contrast
        if l_contrast >= 0.6:  # Excellent contrast already
            contrast_factor = 0.5  # Very small adjustments
            contrast_name = "excellent"
        elif l_contrast >= 0.4:  # Good contrast
            contrast_factor = 0.7
            contrast_name = "good"
        elif l_contrast >= 0.2:  # Moderate contrast
            contrast_factor = 1.0
            contrast_name = "moderate"
        else:  # Low contrast
            contrast_factor = 1.3  # Larger adjustments needed
            contrast_name = "low"
        
        # Combine factors
        range_factor = primary_factor * contrast_factor
        dynamic_multiplier = base_multiplier * range_factor
        
        # Additional boost for contrary colors with low contrast
        if is_contrary_color and l_contrast < 0.3:
            dynamic_multiplier *= 1.2
        
        S4ToolLogger.info("ImagePaletteContrary", f"Dynamic multiplier HSL - Primary L: {primary_l:.3f} ({primary_name}), Contrast: {l_contrast:.3f} ({contrast_name}), Factor: {range_factor:.2f}, Final: {dynamic_multiplier:.0f}")
        
        return dynamic_multiplier
    
    @staticmethod
    def create_artificial_contrast_color(primary_color):
        """
        Create an artificial high-contrast color based on the primary color.
        Maintains some color harmony while ensuring good contrast.
        """
        primary_brightness = ImagePaletteContrary.get_brightness(primary_color)
        
        # Convert to HSV to maintain hue relationships
        r, g, b = primary_color
        h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        
        if primary_brightness > 0.35:  # Even lower threshold for "bright"
            # Primary is bright, create an extremely dark contrary
            contrast_h = (h + 0.5) % 1.0  # Complementary hue
            contrast_s = 0.9   # Very high saturation for dramatic effect
            contrast_v = 0.05  # Extremely dark (almost black but with color)
            S4ToolLogger.info("ImagePaletteContrary", f"Creating extremely dark artificial contrary (primary brightness: {primary_brightness:.3f})")
        else:
            # Primary is dark, create a very bright contrary  
            contrast_h = (h + 0.5) % 1.0  # Complementary hue
            contrast_s = 0.2   # Low saturation for clean brightness
            contrast_v = 0.98  # Almost white
            S4ToolLogger.info("ImagePaletteContrary", f"Creating very bright artificial contrary (primary brightness: {primary_brightness:.3f})")
        
        # Convert back to RGB
        contrast_r, contrast_g, contrast_b = hsv_to_rgb(contrast_h, contrast_s, contrast_v)
        
        # Convert to 0-255 range
        contrary_color = np.array([
            max(0, min(255, int(contrast_r * 255))),
            max(0, min(255, int(contrast_g * 255))),
            max(0, min(255, int(contrast_b * 255)))
        ], dtype=np.uint8)
        
        # Fallback: if still not enough contrast, use extreme colors
        final_brightness = ImagePaletteContrary.get_brightness(contrary_color)
        contrast_achieved = abs(primary_brightness - final_brightness)
        
        if contrast_achieved < 0.4:  # Still not enough contrast
            S4ToolLogger.warning("ImagePaletteContrary", f"Generated color still low contrast ({contrast_achieved:.3f}), using extreme fallback")
            if primary_brightness > 0.35:
                # Use pure dark color
                contrary_color = np.array([15, 15, 25], dtype=np.uint8)  # Very dark blue-gray
            else:
                # Use pure light color  
                contrary_color = np.array([250, 250, 245], dtype=np.uint8)  # Off-white
        
        return contrary_color
    
    @staticmethod
    def adjust_color_saturation(color, contrast_strength):
        """
        Adjust ONLY color saturation (contrast), keeping brightness unchanged.
        - Positive values increase saturation (more vivid)
        - Negative values decrease saturation (more muted)
        """
        if contrast_strength == 0.0:
            return color
        
        r, g, b = color
        h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        
        # Store original brightness
        original_v = v
        
        # Adjust ONLY saturation, preserve hue and value
        if contrast_strength > 0:
            # Increase saturation
            s = min(1.0, s + contrast_strength * (1.0 - s))
        else:
            # Decrease saturation (move towards gray)
            s = max(0.0, s + contrast_strength * s)
        
        # Ensure brightness stays exactly the same
        v = original_v
        
        # Convert back to RGB
        adjusted_r, adjusted_g, adjusted_b = hsv_to_rgb(h, s, v)
        
        adjusted_color = np.array([
            max(0, min(255, int(adjusted_r * 255))),
            max(0, min(255, int(adjusted_g * 255))),
            max(0, min(255, int(adjusted_b * 255)))
        ], dtype=np.uint8)
        
        # Verify brightness didn't change
        original_brightness = ImagePaletteContrary.get_brightness(color)
        new_brightness = ImagePaletteContrary.get_brightness(adjusted_color)
        
        S4ToolLogger.info("ImagePaletteContrary", f"Saturation adjustment - Original: #{color[0]:02X}{color[1]:02X}{color[2]:02X} (brightness: {original_brightness:.3f}), Strength: {contrast_strength:.2f}, Result: #{adjusted_color[0]:02X}{adjusted_color[1]:02X}{adjusted_color[2]:02X} (brightness: {new_brightness:.3f})")
        
        return adjusted_color
    
    @staticmethod
    def adjust_color_strength(color, strength, is_contrary_color=False):
        """
        Adjust color strength to match ImageEnhance.Brightness behavior.
        - strength = 1.0 equals brightness = 100 (factor 2.0)
        - strength = -1.0 equals brightness = -100 (factor 0.0)
        - For bright colors: positive strength makes brighter, negative makes darker
        - For dark colors: positive strength makes darker, negative makes brighter
        """
        if strength == 0.0:
            return color
        
        r, g, b = color
        brightness = ImagePaletteContrary.get_brightness(color)
        
        # Determine if color is bright or dark
        # Use a more reasonable threshold - colors below 0.5 are considered dark
        brightness_threshold = 0.5  # Colors below 0.5 brightness are "dark", above 0.5 are "bright"
        is_bright = brightness > brightness_threshold
        
        S4ToolLogger.info("ImagePaletteContrary", f"Brightness analysis: {brightness:.3f} vs threshold {brightness_threshold} -> {'BRIGHT' if is_bright else 'DARK'}")
        
        # Convert color to PIL Image for consistent processing with ImageAdjustment
        color_image = Image.new('RGB', (1, 1), tuple(color))
        
        # SIMPLIFIED LOGIC: Use a fixed, reasonable multiplier instead of complex dynamic calculation
        base_multiplier = 50  # Much smaller, more predictable multiplier
        
        # User expected logic: positive strength enhances the color's nature
        # For bright colors: positive = brighter, negative = darker  
        # For dark colors: positive = darker, negative = brighter
        # 
        # PIL behavior: factor > 1.0 = brighter, factor < 1.0 = darker
        if is_bright:
            # Bright colors: positive strength makes them brighter (factor > 1.0)
            brightness_value = strength * base_multiplier  # positive -> factor > 1.0 -> brighter
        else:
            # Dark colors: positive strength makes them darker (factor < 1.0)
            brightness_value = -strength * base_multiplier  # positive strength -> negative value -> factor < 1.0 -> darker
        
        # No need to limit since we're using a reasonable base_multiplier
        
        brightness_factor = 1.0 + (brightness_value / 100.0)
        brightness_factor = max(0.1, min(3.0, brightness_factor))  # Limit factor to 0.1-3.0
        
        # Apply PIL ImageEnhance.Brightness
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(color_image)
        adjusted_image = enhancer.enhance(brightness_factor)
        
        # Extract the adjusted color
        adjusted_rgb = adjusted_image.getpixel((0, 0))
        
        # Store debug info
        S4ToolLogger.info("ImagePaletteContrary", f"Color adjustment - Original: #{color[0]:02X}{color[1]:02X}{color[2]:02X}, Brightness: {brightness:.3f} ({'bright' if is_bright else 'dark'}), Strength: {strength:.2f}, Contrary: {is_contrary_color}")
        S4ToolLogger.info("ImagePaletteContrary", f"PIL Brightness - Value: {brightness_value:.1f}, Factor: {brightness_factor:.3f}, Base Multiplier: {base_multiplier}")
        S4ToolLogger.info("ImagePaletteContrary", f"Result: #{adjusted_rgb[0]:02X}{adjusted_rgb[1]:02X}{adjusted_rgb[2]:02X}")
        
        adjusted_color = np.array(adjusted_rgb, dtype=np.uint8)
        
        return adjusted_color
    
    @staticmethod
    def rgb_to_hsl(color):
        """Convert RGB to HSL"""
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        if diff == 0:
            h = s = 0.0
        else:
            # Saturation
            s = diff / (2.0 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
            
            # Hue
            if max_val == r:
                h = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / diff + 2
            else:
                h = (r - g) / diff + 4
            h /= 6.0
        
        return h, s, l
    
    @staticmethod
    def find_contrary_color(primary_color, all_colors):
        """
        Find the color with maximum L-value contrast using HSL.
        If primary has high L-value, find lowest L-value; if low L-value, find highest L-value.
        """
        primary_h, primary_s, primary_l = ImagePaletteContrary.rgb_to_hsl(primary_color)
        
        S4ToolLogger.info("ImagePaletteContrary", f"Primary color HSL: H={primary_h:.3f}, S={primary_s:.3f}, L={primary_l:.3f}")
        
        # Determine if primary is bright or dark based on L-value
        is_primary_bright = primary_l > 0.5
        
        best_contrast_color = primary_color
        target_l = 1.0 if is_primary_bright else 0.0  # Target opposite extreme
        best_l_diff = 0.0
        
        S4ToolLogger.info("ImagePaletteContrary", f"Primary is {'bright' if is_primary_bright else 'dark'} (L={primary_l:.3f}), searching for {'darkest' if is_primary_bright else 'brightest'} color")
        
        for color in all_colors:
            color_h, color_s, color_l = ImagePaletteContrary.rgb_to_hsl(color)
            
            if is_primary_bright:
                # Primary is bright, find the darkest (lowest L-value)
                if color_l < primary_l:  # Only consider darker colors
                    l_diff = primary_l - color_l  # Difference from primary
                    if l_diff > best_l_diff:
                        best_l_diff = l_diff
                        best_contrast_color = color
                        S4ToolLogger.debug("ImagePaletteContrary", f"Found darker color: #{color[0]:02X}{color[1]:02X}{color[2]:02X} (L={color_l:.3f})")
            else:
                # Primary is dark, find the brightest (highest L-value)
                if color_l > primary_l:  # Only consider brighter colors
                    l_diff = color_l - primary_l  # Difference from primary
                    if l_diff > best_l_diff:
                        best_l_diff = l_diff
                        best_contrast_color = color
                        S4ToolLogger.debug("ImagePaletteContrary", f"Found brighter color: #{color[0]:02X}{color[1]:02X}{color[2]:02X} (L={color_l:.3f})")
        
        final_h, final_s, final_l = ImagePaletteContrary.rgb_to_hsl(best_contrast_color)
        S4ToolLogger.info("ImagePaletteContrary", f"Best L-contrast color: #{best_contrast_color[0]:02X}{best_contrast_color[1]:02X}{best_contrast_color[2]:02X} (L={final_l:.3f}, diff={best_l_diff:.3f})")
        
        return best_contrast_color

    def extract_contrary_palette(self, image, color1_strength, color2_strength, contrast):
        """
        Extract primary color using K-Means and find contrary color with maximum brightness contrast.
        Apply strength adjustments to both colors.
        """
        try:
            S4ToolLogger.info("ImagePaletteContrary", "Starting contrary palette extraction")
            
            # Prepare image for processing using base class method
            image_array = self.prepare_image_for_processing(image)
            
            # Extract more colors using K-Means to have a better pool for contrast selection
            # Use base class method to match imagePalette631.py exactly
            all_colors = self.extract_colors_with_algorithm(image_array, "K-Means Clustering", n_colors=10)
            
            # Primary color is the first one (most dominant from K-Means)
            primary_color = all_colors[0]
            
            S4ToolLogger.info("ImagePaletteContrary", f"Primary color (K-Means main): #{primary_color[0]:02X}{primary_color[1]:02X}{primary_color[2]:02X}")
            
            # Find contrary color with maximum brightness contrast
            contrary_color = self.find_contrary_color(primary_color, all_colors)
            
            # Check if we have low contrast situation using HSL L-values
            primary_h, primary_s, primary_l = self.rgb_to_hsl(primary_color)
            contrary_h, contrary_s, contrary_l = self.rgb_to_hsl(contrary_color)
            l_contrast_diff = abs(primary_l - contrary_l)
            
            S4ToolLogger.info("ImagePaletteContrary", f"Contrary color found: #{contrary_color[0]:02X}{contrary_color[1]:02X}{contrary_color[2]:02X}")
            S4ToolLogger.info("ImagePaletteContrary", f"HSL contrast analysis - Primary L: {primary_l:.3f}, Contrary L: {contrary_l:.3f}, L-diff: {l_contrast_diff:.3f}")
            
            # Handle low contrast situations using HSL L-value (more accurate)
            low_l_contrast_threshold = 0.3  # If L-difference < 0.3, consider it low contrast
            is_low_contrast = l_contrast_diff < low_l_contrast_threshold
            
            # Emergency fallback: if both colors are too similar in RGB space
            rgb_diff = np.sum(np.abs(primary_color.astype(int) - contrary_color.astype(int)))
            is_very_similar = rgb_diff < 120  # Sum of RGB differences < 120
            
            S4ToolLogger.info("ImagePaletteContrary", f"Low contrast check - L-diff: {l_contrast_diff:.3f}, RGB diff: {rgb_diff}, L-threshold: {low_l_contrast_threshold}")
            S4ToolLogger.info("ImagePaletteContrary", f"Low L-contrast: {is_low_contrast}, Very similar: {is_very_similar}")
            
            if is_low_contrast or is_very_similar:
                reason = "L-value" if is_low_contrast else "RGB similarity"
                S4ToolLogger.warning("ImagePaletteContrary", f"Low contrast detected ({reason}: {l_contrast_diff:.3f}/{rgb_diff}), creating artificial contrast")
                
                # For low contrast images, create artificial high contrast based on primary color
                contrary_color = self.create_artificial_contrast_color(primary_color)
                S4ToolLogger.info("ImagePaletteContrary", f"Low contrast fix: created artificial contrary #{contrary_color[0]:02X}{contrary_color[1]:02X}{contrary_color[2]:02X}")
                # Update the flag
                is_low_contrast = True
            
            # If primary and contrary are the same (shouldn't happen but safety check)
            elif np.array_equal(primary_color, contrary_color):
                # Create a high contrast color manually
                if primary_brightness > 0.5:
                    # Primary is bright, create dark contrary
                    contrary_color = np.array([30, 30, 30], dtype=np.uint8)  # Dark gray
                else:
                    # Primary is dark, create bright contrary
                    contrary_color = np.array([225, 225, 225], dtype=np.uint8)  # Light gray
                
                S4ToolLogger.warning("ImagePaletteContrary", f"Primary and contrary were identical, created manual contrast: #{contrary_color[0]:02X}{contrary_color[1]:02X}{contrary_color[2]:02X}")
            
            # Apply strength adjustments
            original_primary = primary_color.copy()
            original_contrary = contrary_color.copy()
            
            # Apply strength adjustments with special handling for low contrast
            if color1_strength != 0.0:
                # For low contrast situations, reduce strength to prevent over-adjustment
                adjusted_strength1 = color1_strength * 0.6 if is_low_contrast else color1_strength
                primary_color = self.adjust_color_strength(primary_color, adjusted_strength1, is_contrary_color=False)
                S4ToolLogger.info("ImagePaletteContrary", f"Color1 strength {color1_strength} (adjusted: {adjusted_strength1:.2f}): #{original_primary[0]:02X}{original_primary[1]:02X}{original_primary[2]:02X} -> #{primary_color[0]:02X}{primary_color[1]:02X}{primary_color[2]:02X}")
            
            if color2_strength != 0.0:
                # For low contrast, be even more careful with the artificial contrary color
                adjusted_strength2 = color2_strength * 0.3 if is_low_contrast else color2_strength
                contrary_color = self.adjust_color_strength(contrary_color, adjusted_strength2, is_contrary_color=True)
                S4ToolLogger.info("ImagePaletteContrary", f"Color2 strength {color2_strength} (adjusted: {adjusted_strength2:.2f}): #{original_contrary[0]:02X}{original_contrary[1]:02X}{original_contrary[2]:02X} -> #{contrary_color[0]:02X}{contrary_color[1]:02X}{contrary_color[2]:02X}")
            
            # Apply saturation adjustment (contrast) to both colors
            if contrast != 0.0:
                S4ToolLogger.info("ImagePaletteContrary", f"Applying saturation adjustment: {contrast:.2f}")
                primary_color = self.adjust_color_saturation(primary_color, contrast)
                contrary_color = self.adjust_color_saturation(contrary_color, contrast)
            
            # Create preview image with two color blocks
            colors_for_preview = [primary_color, contrary_color]
            preview_w = ImageConstants.PALETTE_BLOCK_WIDTH
            block_h = ImageConstants.PALETTE_BLOCK_HEIGHT
            
            preview_img = Image.new('RGB', (preview_w, block_h * 2))
            for i, color in enumerate(colors_for_preview):
                block = Image.new('RGB', (preview_w, block_h), tuple(color))
                preview_img.paste(block, (0, i * block_h))
            
            # Convert to output format
            preview_tensor = pil2tensor(preview_img)
            
            # Convert colors to hex
            primary_hex = f"#{primary_color[0]:02X}{primary_color[1]:02X}{primary_color[2]:02X}"
            contrary_hex = f"#{contrary_color[0]:02X}{contrary_color[1]:02X}{contrary_color[2]:02X}"
            
            S4ToolLogger.success("ImagePaletteContrary", f"Contrary palette extraction completed: {primary_hex} -> {contrary_hex}")
            
            return (preview_tensor, primary_hex, contrary_hex)
            
        except Exception as e:
            S4ToolLogger.error("ImagePaletteContrary", f"Contrary palette extraction failed: {str(e)}")
            
            # Return fallback colors
            fallback_preview = Image.new('RGB', (400, 160), (128, 128, 128))
            preview_tensor = pil2tensor(fallback_preview)
            return (preview_tensor, "#808080", "#FFFFFF")

# Node mappings are handled in __init__.py