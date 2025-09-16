import os
import sys
import types
import importlib.util
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter

# Import shared conversion functions
from ..nodes_config import pil2tensor, tensor2pil
from ..dependency_manager import require_dependency, S4ToolLogger

try:
    import folder_paths  # ComfyUI helper
except Exception:
    class _FolderPathsMock:
        models_dir = os.path.join(os.path.expanduser("~"), ".cache", "comfyui", "models")
    folder_paths = _FolderPathsMock()

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from transformers import AutoModelForImageSegmentation, PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ----------------------
# Utils
# ----------------------
def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL using shared function for consistency."""
    return tensor2pil(image.unsqueeze(0) if image.dim() == 3 else image)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL to tensor using shared function for consistency."""
    return pil2tensor(image)


def _compose_rgba(rgb_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    rgba = rgb_pil.convert("RGBA")
    r, g, b, _ = rgba.split()
    return Image.merge("RGBA", (r, g, b, mask_pil))


def _refine_foreground(image_bchw: torch.Tensor, masks_b1hw: torch.Tensor) -> torch.Tensor:
    # Simple mask refinement using lightweight blur on transition region
    batch, _, h, w = image_bchw.shape
    refined_list = []
    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()

    for i in range(batch):
        mask = mask_np[i, 0]
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        # Blur using PIL to avoid cv2 dependency
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_blur_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
        mask_blur = np.array(mask_blur_img).astype(np.float32) / 255.0

        transition = (mask > 0.05) & (mask < 0.95)
        alpha = 0.85
        mask_refined = np.where(transition, alpha * mask + (1 - alpha) * mask_blur, (mask > 0.45).astype(np.float32))
        mask_refined = np.clip(mask_refined, 0.0, 1.0)

        channels = []
        for c in range(image_np.shape[1]):
            channels.append(image_np[i, c] * mask_refined)
        refined_list.append(np.stack(channels))

    return torch.from_numpy(np.stack(refined_list))


def _dilate_erode_mask(mask_img: Image.Image, offset: int) -> Image.Image:
    """Enhanced mask dilation/erosion with better edge preservation."""
    if offset == 0:
        return mask_img
    if offset > 0:
        for _ in range(offset):
            mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    else:
        for _ in range(-offset):
            mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    return mask_img


def _apply_feathering(mask_img: Image.Image, radius: float) -> Image.Image:
    """Apply soft feathering to mask edges for natural transitions."""
    if radius <= 0:
        return mask_img
    
    # Convert to numpy for distance-based feathering
    mask_array = np.array(mask_img, dtype=np.float32) / 255.0
    
    # Distance-based feathering using scipy
    require_dependency('scipy', 'Advanced Edge Feathering')
    from scipy import ndimage
    
    # Calculate distance transform from edges
    edges = np.abs(ndimage.sobel(mask_array)) > 0.1
    distance = ndimage.distance_transform_edt(~edges)
    
    # Apply feathering based on distance
    feather_mask = np.clip(distance / radius, 0, 1)
    feathered = mask_array * feather_mask
    
    S4ToolLogger.info("ImageRMBG", f"Applied distance-based feathering with radius {radius}")
    return Image.fromarray((feathered * 255).astype(np.uint8))


def _guided_filter_refinement(image: Image.Image, mask: Image.Image, radius: int = 8) -> Image.Image:
    """Apply guided filter for edge-aware mask refinement."""
    require_dependency('cv2', 'Guided Filter Edge Refinement')
    import cv2
    
    # Convert to arrays
    img_array = np.array(image.convert('RGB'))
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    
    # Apply guided filter using image as guidance
    eps = 0.01
    refined = cv2.ximgproc.guidedFilter(img_array, mask_array, radius, eps)
    refined = np.clip(refined, 0, 1)
    
    S4ToolLogger.info("ImageRMBG", f"Applied guided filter refinement with radius {radius}")
    return Image.fromarray((refined * 255).astype(np.uint8))


def _alpha_matting_refinement(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Alpha matting approximation for hair and fine details."""
    require_dependency('scipy', 'Alpha Matting Refinement')
    from scipy import ndimage
    
    img_array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    
    # Simple trimap generation
    trimap = np.zeros_like(mask_array)
    trimap[mask_array > 0.8] = 1.0  # Definite foreground
    trimap[mask_array < 0.2] = 0.0  # Definite background
    # Unknown region between 0.2 and 0.8
    
    # Simple local color analysis for unknown regions
    unknown_mask = (mask_array >= 0.2) & (mask_array <= 0.8)
    if np.any(unknown_mask):
        # Use local gradient information for refinement
        grad_magnitude = np.sqrt(
            ndimage.sobel(img_array[:,:,0])**2 + 
            ndimage.sobel(img_array[:,:,1])**2 + 
            ndimage.sobel(img_array[:,:,2])**2
        )
        # Lower gradient areas are more likely to be foreground
        alpha_adjustment = 1.0 - np.clip(grad_magnitude * 2, 0, 1)
        mask_array[unknown_mask] *= alpha_adjustment[unknown_mask]
    
    S4ToolLogger.info("ImageRMBG", f"Applied alpha matting refinement to {np.sum(unknown_mask)} unknown pixels")
    return Image.fromarray((mask_array * 255).astype(np.uint8))


def _apply_background_color(image: Image.Image, mask: Image.Image, bg_color: str) -> Image.Image:
    """Replace transparent background with solid color."""
    if not bg_color.strip():
        return image
    
    try:
        # Parse color (support hex, rgb, color names)
        if bg_color.startswith('#'):
            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
        elif bg_color.startswith('rgb('):
            # Parse rgb(r,g,b) format
            rgb_str = bg_color.strip('rgb()').split(',')
            bg_rgb = tuple(int(x.strip()) for x in rgb_str)
        else:
            # Try PIL color names
            from PIL import ImageColor
            bg_rgb = ImageColor.getrgb(bg_color)
        
        # Create solid background
        background = Image.new('RGB', image.size, bg_rgb)
        
        # Composite with alpha
        if image.mode == 'RGBA':
            result = Image.alpha_composite(background.convert('RGBA'), image)
            return result.convert('RGB')
        else:
            # Use mask for compositing
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            img_array = np.array(image.convert('RGB'), dtype=np.float32)
            bg_array = np.array(background, dtype=np.float32)
            
            # Alpha blend
            result_array = img_array * mask_array[..., None] + bg_array * (1 - mask_array[..., None])
            return Image.fromarray(result_array.astype(np.uint8))
    except Exception as e:
        raise RuntimeError(f"Failed to apply background color '{bg_color}': {e}")


def _get_quality_settings(quality_mode: str) -> dict:
    """Get processing settings based on quality mode."""
    settings = {
        "Fast": {
            "tile_size": 512,
            "overlap": 32,
            "post_process": False,
            "iterations": 1
        },
        "Balanced": {
            "tile_size": 1024,
            "overlap": 64,
            "post_process": True,
            "iterations": 1
        },
        "High Quality": {
            "tile_size": 1536,
            "overlap": 128,
            "post_process": True,
            "iterations": 2
        },
        "Ultra": {
            "tile_size": 2048,
            "overlap": 256,
            "post_process": True,
            "iterations": 3
        }
    }
    return settings.get(quality_mode, settings["Balanced"])


# ----------------------
# RMBG-2.0 Loader (self-contained)
# ----------------------
class RMBG20Loader:
    def __init__(self) -> None:
        self.model = None
        self.loaded = False
        base_dir = getattr(folder_paths, "models_dir", os.path.join(os.path.expanduser("~"), ".cache", "comfyui", "models"))
        self.cache_dir = os.path.join(base_dir, "RMBG", "RMBG-2.0")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _ensure_files(self) -> None:
        files = {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
        }
        missing = [fname for fname in files if not os.path.exists(os.path.join(self.cache_dir, files[fname]))]
        if not missing:
            return
        print("🟡 [RMBG-Loader] Required model files are missing:", missing)
        print(f"🟡 [RMBG-Loader] Target install directory: {self.cache_dir}")
        print("🟡 [RMBG-Loader] You can place files manually or allow auto download from '1038lab/RMBG-2.0'.")
        if not HF_AVAILABLE:
            print("🔴 [RMBG-Loader] huggingface_hub is not available. Please install: pip install huggingface_hub")
            print(f"🔴 [RMBG-Loader] Then re-run, or manually put the files into: {self.cache_dir}")
            raise RuntimeError("huggingface_hub is required to download RMBG-2.0 files")
        repo_id = "1038lab/RMBG-2.0"
        for fname in files:
            local_path = os.path.join(self.cache_dir, files[fname])
            if os.path.exists(local_path):
                continue
            print(f"🟡 [RMBG-Loader] Downloading '{fname}' from {repo_id} → {self.cache_dir}")
            hf_hub_download(repo_id=repo_id, filename=fname, local_dir=self.cache_dir)
            print(f"✅ [RMBG-Loader] Downloaded '{fname}'")

    def _load_dynamic(self) -> None:
        # Load BiRefNet_config
        config_path = os.path.join(self.cache_dir, "BiRefNet_config.py")
        spec_cfg = importlib.util.spec_from_file_location("BiRefNetConfig", config_path)
        cfg_mod = importlib.util.module_from_spec(spec_cfg)
        sys.modules["BiRefNetConfig"] = cfg_mod
        spec_cfg.loader.exec_module(cfg_mod)

        # Load birefnet.py and fix relative import
        biref_path = os.path.join(self.cache_dir, "birefnet.py")
        with open(biref_path, "r", encoding="utf-8") as f:
            code = f.read()
        code = code.replace("from .BiRefNet_config import BiRefNetConfig", "from BiRefNetConfig import BiRefNetConfig")
        module_name = f"removebg_birefnet_{abs(hash(biref_path))}"
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
        exec(code, mod.__dict__)

        # Instantiate model class derived from PreTrainedModel
        model_cls = None
        for name in dir(mod):
            attr = getattr(mod, name)
            if TRANSFORMERS_AVAILABLE and isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr is not PreTrainedModel:
                model_cls = attr
                break
        if model_cls is None:
            raise RuntimeError("RMBG-2.0 model class not found in birefnet.py")

        model_config = getattr(cfg_mod, "BiRefNetConfig")()
        self.model = model_cls(model_config)

        # Load weights (strict, no fallback)
        weights_path = os.path.join(self.cache_dir, "model.safetensors")
        require_dependency('safetensors', 'RMBG-2.0 Weights')
        import safetensors.torch
        state = safetensors.torch.load_file(weights_path)
        self.model.load_state_dict(state)

    def load(self) -> None:
        if self.loaded:
            return
        self._ensure_files()
        print(f"🟢 [RMBG-Loader] Loading RMBG-2.0 from: {self.cache_dir}")
        self._load_dynamic()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"✅ [RMBG-Loader] Model ready on device: {device}")
        self.loaded = True

    @torch.no_grad()
    def infer_masks(self, images: List[torch.Tensor], process_res: int, sensitivity: float) -> List[Image.Image]:
        self.load()
        device = next(self.model.parameters()).device
        tfm = transforms.Compose([
            transforms.Resize((process_res, process_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        orig_sizes: List[Tuple[int, int]] = []
        input_tensors: List[torch.Tensor] = []
        total = len(images)
        S4ToolLogger.info("RMBG", "Inference started: {total} image(s), res={process_res}, sensitivity={sensitivity:.2f}")
        for idx, img in enumerate(images, start=1):
            pil_img = _tensor_to_pil(img)
            orig_sizes.append(pil_img.size)
            input_tensors.append(tfm(pil_img).unsqueeze(0))
            S4ToolLogger.info("RMBG", "Preprocess {idx}/{total} done (size={pil_img.size})")
        input_batch = torch.cat(input_tensors, dim=0).to(device)

        print("🟢 [RMBG] Forward pass...")
        outputs = self.model(input_batch)
        if isinstance(outputs, list) and len(outputs) > 0:
            results = outputs[-1]
        elif isinstance(outputs, dict) and "logits" in outputs:
            results = outputs["logits"]
        elif isinstance(outputs, torch.Tensor):
            results = outputs
        else:
            # generic pick first tensor-like output
            results = None
            if isinstance(outputs, dict):
                for v in outputs.values():
                    if isinstance(v, torch.Tensor):
                        results = v
                        break
            if results is None:
                raise RuntimeError("Unrecognized RMBG-2.0 output format")

        results = results.sigmoid().detach().cpu()
        out_masks: List[Image.Image] = []
        for idx, (result, (w, h)) in enumerate(zip(results, orig_sizes), start=1):
            result = result.squeeze()
            # Single sensitivity amplification (avoid double boost)
            gain = 1.0 + (1.0 - float(sensitivity))
            result = torch.clamp(result * gain, 0.0, 1.0)
            result = F.interpolate(result.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze()
            out_masks.append(_tensor_to_pil(result))
            S4ToolLogger.info("RMBG", "Postprocess {idx}/{total} done (target={w}x{h})")
        print("✅ [RMBG] Inference completed")
        return out_masks


# ----------------------
# ComfyUI Node: 💀Image RMBG
# ----------------------
class ImageRMBG:
    """
    Enhanced RMBG-2.0 background removal with advanced post-processing options.
    Supports multiple output formats, quality optimization, and detailed edge handling.
    """
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "image": "Input image to be processed for background removal",
            "sensitivity": "Mask detection sensitivity (0.0=aggressive, 1.0=conservative)",
            "process_res": "Processing resolution - higher = better quality but slower",
            "output_format": "Output format: RGBA=transparent background, Foreground=black background, Mask=mask preview, Original+Background=replace background with color",
            "edge_method": "Edge refinement technique for smoother transitions",
            "feather_radius": "Soft feathering radius for natural edges (0=sharp)",
            "mask_blur": "Gaussian blur radius for mask smoothing",
            "mask_offset": "Expand (+) or contract (-) mask boundaries",
            "quality_mode": "Processing quality vs speed trade-off",
            "batch_size": "Process multiple images simultaneously for efficiency",
            "alpha_matting": "Advanced alpha matting for hair and fine details",
            "background_color": "Background color: Used by 'Original + Background' and 'Foreground Only' formats. RGBA format always outputs transparent background.",
            "invert_output": "Invert the mask output (foreground<->background)"
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "process_res": ([256, 512, 1024, 1536, 2048], {"default": 1024, "tooltip": tooltips["process_res"]}),
                "output_format": (["RGBA (Transparent)", "Foreground Only", "Mask Only", "Original + Background"], 
                                {"default": "RGBA (Transparent)", "tooltip": tooltips["output_format"]}),
                "edge_method": (["None", "Gaussian", "Guided Filter", "Alpha Matting", "Smart Blur"], 
                              {"default": "Gaussian", "tooltip": tooltips["edge_method"]}),
                "feather_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": tooltips["feather_radius"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -32, "max": 32, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "quality_mode": (["Fast", "Balanced", "High Quality", "Ultra"], 
                                {"default": "Balanced", "tooltip": tooltips["quality_mode"]}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": tooltips["batch_size"]}),
                "alpha_matting": ("BOOLEAN", {"default": False, "tooltip": tooltips["alpha_matting"]}),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False, "tooltip": tooltips["background_color"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "process"
    CATEGORY = "💀S4Tool"
    OUTPUT_NODE = False

    def __init__(self) -> None:
        self.loader = RMBG20Loader()

    def process(self, image, sensitivity=1.0, process_res=1024, output_format="RGBA (Transparent)", 
               edge_method="Gaussian", feather_radius=0.0, mask_blur=0, mask_offset=0, 
               quality_mode="Balanced", batch_size=1, alpha_matting=False, 
               background_color="", invert_output=False):
        import time
        start_time = time.time()
        
        try:
            # Normalize input to a list of tensors
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    images = [image]
                else:
                    images = [img for img in image]
            else:
                images = list(image)

            # Get quality settings
            quality_settings = _get_quality_settings(quality_mode)
            
            # Process in batches for efficiency
            batch_size = min(int(batch_size), len(images))
            processed_images = []
            processed_masks = []
            
            total_images = len(images)
            processing_info = []
            
            print(f"🟢 [RMBG Enhanced] Processing {total_images} image(s) with {quality_mode} quality")
            
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]
                
                # Inference with RMBG-2.0
                masks: List[Image.Image] = self.loader.infer_masks(batch_images, int(process_res), float(sensitivity))
                
                for idx, (img_tensor, mask_pil) in enumerate(zip(batch_images, masks)):
                    img_pil = _tensor_to_pil(img_tensor)
                    original_size = img_pil.size
                    
                    # Apply edge refinement methods
                    if edge_method == "Gaussian" and mask_blur > 0:
                        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=int(mask_blur)))
                    elif edge_method == "Guided Filter":
                        mask_pil = _guided_filter_refinement(img_pil, mask_pil)
                    elif edge_method == "Alpha Matting" or alpha_matting:
                        mask_pil = _alpha_matting_refinement(img_pil, mask_pil)
                    elif edge_method == "Smart Blur":
                        # Smart blur based on mask confidence
                        mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
                        confidence = np.std(mask_array)
                        blur_radius = max(1, int(3 - confidence * 3))
                        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    
                    # Apply feathering
                    if feather_radius > 0:
                        mask_pil = _apply_feathering(mask_pil, feather_radius)
                    
                    # Apply mask offset (expand/contract)
                    if mask_offset != 0:
                        mask_pil = _dilate_erode_mask(mask_pil, int(mask_offset))
                    
                    # Additional mask blur if specified separately
                    if edge_method != "Gaussian" and mask_blur > 0:
                        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=int(mask_blur)))
                    
                    # Invert mask if requested
                    if invert_output:
                        mask_pil = Image.fromarray(255 - np.array(mask_pil))
                    
                    # Generate output based on format - each format serves a specific purpose
                    if output_format == "RGBA (Transparent)":
                        # Standard transparent background PNG - most common use case
                        result_img = _compose_rgba(img_pil, mask_pil)
                        
                    elif output_format == "Foreground Only":
                        # Foreground with black background - useful for further compositing
                        img_array = np.array(img_pil.convert('RGB'))
                        mask_array = np.array(mask_pil) / 255.0
                        # Apply mask to RGB, background becomes black
                        fg_array = img_array * mask_array[..., None]
                        result_img = Image.fromarray(fg_array.astype(np.uint8))
                        
                    elif output_format == "Mask Only":
                        # Pure mask for analysis or further processing
                        result_img = mask_pil.convert('RGB')
                        
                    elif output_format == "Original + Background":
                        # Original image with custom background color - useful for product photos
                        if background_color and background_color.strip():
                            # Composite with custom background
                            result_img = _compose_rgba(img_pil, mask_pil)
                            result_img = _apply_background_color(result_img, mask_pil, background_color)
                        else:
                            # If no background color specified, default to white
                            result_img = _compose_rgba(img_pil, mask_pil)
                            result_img = _apply_background_color(result_img, mask_pil, "#FFFFFF")
                            
                    else:
                        # Fallback to RGBA
                        result_img = _compose_rgba(img_pil, mask_pil)
                    
                    # Apply background color only to Foreground Only format when background_color is specified
                    if background_color and background_color.strip() and output_format == "Foreground Only":
                        # Replace black background with custom color for Foreground Only format
                        result_img = _apply_background_color(result_img, mask_pil, background_color)
                    
                    processed_images.append(_pil_to_tensor(result_img))
                    processed_masks.append(_pil_to_tensor(mask_pil))
                    
                    # Collect processing info
                    processing_info.append(f"Image {batch_start + idx + 1}: {original_size[0]}x{original_size[1]}")
            
            # Prepare final output
            result_images = torch.cat(processed_images, dim=0)
            result_masks = torch.cat(processed_masks, dim=0)
            
            # Create processing summary
            end_time = time.time()
            processing_time = end_time - start_time
            
            info_summary = f"RMBG-2.0 Enhanced Processing Summary:\n"
            info_summary += f"- Total Images: {total_images}\n"
            info_summary += f"- Processing Resolution: {process_res}\n"
            info_summary += f"- Quality Mode: {quality_mode}\n"
            info_summary += f"- Edge Method: {edge_method}\n"
            info_summary += f"- Output Format: {output_format}\n"
            info_summary += f"- Processing Time: {processing_time:.2f}s\n"
            info_summary += f"- Average Time per Image: {processing_time/total_images:.2f}s\n"
            if background_color:
                info_summary += f"- Background Color: {background_color}\n"
            
            print(f"✅ [RMBG Enhanced] Completed in {processing_time:.2f}s")
            
            return (result_images, result_masks, info_summary)
        except Exception as e:
            S4ToolLogger.info("ImageRMBG", "Error: {e}")
            # Construct an empty mask with a safe shape
            try:
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:  # (B, H, W, C) or (B, C, H, W)
                        if image.shape[-1] in (1, 3, 4):
                            b, h, w = image.shape[0], image.shape[1], image.shape[2]
                        elif image.shape[1] in (1, 3, 4):
                            b, h, w = image.shape[0], image.shape[2], image.shape[3]
                        else:
                            b, h, w = image.shape[0], image.shape[1], image.shape[2]
                    elif image.dim() == 3:  # (H, W, C) or (C, H, W)
                        if image.shape[-1] in (1, 3, 4):
                            b, h, w = 1, image.shape[0], image.shape[1]
                        elif image.shape[0] in (1, 3, 4):
                            b, h, w = 1, image.shape[1], image.shape[2]
                        else:
                            b, h, w = 1, image.shape[0], image.shape[1]
                    else:
                        b, h, w = 1, 256, 256
                else:
                    b, h, w = 1, 256, 256
                empty_mask = torch.zeros((b, h, w), dtype=torch.float32)
            except Exception:
                empty_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
            error_info = f"RMBG-2.0 Processing Failed: {str(e)}"
            return (image, empty_mask, error_info)


