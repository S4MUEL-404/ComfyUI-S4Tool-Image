"""
S4SAM2 Image - Segment Anything Model 2 for ComfyUI
Integrated with S4Tool-Image Production Quality Suite

Combines GroundingDINO text detection with SAM2 segmentation
for superior text-prompt driven object segmentation

Author: S4MUEL
Homepage: https://s4muel.com
GitHub: https://github.com/S4MUEL-404/ComfyUI-S4Tool-Image
"""
import os
import sys
import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
import glob
from typing import Tuple, List, Dict, Optional
import importlib
import scipy.ndimage as ndimage

# Import dependency manager from parent module
from ..dependency_manager import require_dependency, S4ToolLogger

# Add current path and parent path to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import GroundingDINO modules (same as original imageSAM.py)
from GroundingDINO.datasets import transforms as T
from GroundingDINO.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from GroundingDINO.util.slconfig import SLConfig as local_groundingdino_SLConfig
from GroundingDINO.models import build_model as local_groundingdino_build_model

# Import SAM2 modules
from SAM2.load_model import load_model as load_sam2_model

# Check dependencies at import
require_dependency('yaml', 'SAM2 Configuration')
require_dependency('segment_anything', 'SAM Core Functionality')
require_dependency('timm', 'Vision Transformer Support')
require_dependency('addict', 'Configuration Management')

# SAM2 model configuration
sam2_model_dir_name = "sam2"
sam2_model_list = {
    # SAM2.1 Series (Latest, Recommended)
    "sam2.1_hiera_tiny (150MB)": {
        "model_url": "sam2.1_hiera_tiny.safetensors",
        "config": "sam2.1_hiera_t.yaml"
    },
    "sam2.1_hiera_small (200MB)": {
        "model_url": "sam2.1_hiera_small.safetensors",
        "config": "sam2.1_hiera_s.yaml"
    },
    "sam2.1_hiera_base_plus (250MB)": {
        "model_url": "sam2.1_hiera_base_plus.safetensors",
        "config": "sam2.1_hiera_b+.yaml"
    },
    "sam2.1_hiera_large (350MB)": {
        "model_url": "sam2.1_hiera_large.safetensors",
        "config": "sam2.1_hiera_l.yaml"
    },
    # SAM2 Original Series
    "sam2_hiera_tiny (150MB)": {
        "model_url": "sam2_hiera_tiny.safetensors",
        "config": "sam2_hiera_t.yaml"
    },
    "sam2_hiera_small (200MB)": {
        "model_url": "sam2_hiera_small.safetensors",
        "config": "sam2_hiera_s.yaml"
    },
    "sam2_hiera_base_plus (250MB)": {
        "model_url": "sam2_hiera_base_plus.safetensors",
        "config": "sam2_hiera_b+.yaml"
    },
    "sam2_hiera_large (350MB)": {
        "model_url": "sam2_hiera_large.safetensors",
        "config": "sam2_hiera_l.yaml"
    },
}

# GroundingDINO configuration (same as original)
groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    """Get BERT model path for GroundingDINO"""
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        S4ToolLogger.info("ImageSAM2", "Using models/bert-base-uncased")
        return comfy_bert_model_base
    return 'bert-base-uncased'

def get_local_filepath(url, dirname, local_file_name=None):
    """Download or get local file path - works with any ComfyUI installation"""
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    # Get models directory from ComfyUI configuration (works for any installation path)
    try:
        models_base = folder_paths.models_dir
    except Exception as e:
        S4ToolLogger.error("ImageSAM2", f"Cannot access ComfyUI models directory: {e}")
        raise RuntimeError(f"ComfyUI models directory not accessible: {e}")
    
    folder = os.path.join(models_base, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        S4ToolLogger.info("ImageSAM2", f"Created models directory: {folder}")

    destination = os.path.join(folder, local_file_name)
    
    # Check if file already exists
    if os.path.exists(destination):
        S4ToolLogger.info("ImageSAM2", f"Using existing model: {destination}")
        return destination
    
    # Download if file doesn't exist
    S4ToolLogger.info("ImageSAM2", f"Downloading {url} to {destination}")
    try:
        download_url_to_file(url, destination)
        if os.path.exists(destination):
            S4ToolLogger.success("ImageSAM2", f"Successfully downloaded: {destination}")
            return destination
        else:
            raise FileNotFoundError(f"Download completed but file not found: {destination}")
    except Exception as e:
        S4ToolLogger.error("ImageSAM2", f"Failed to download {url}: {str(e)}")
        raise RuntimeError(f"Model download failed: {str(e)}")

def load_sam2_model_wrapper(model_name):
    """Load SAM2 model from configuration"""
    S4ToolLogger.info("ImageSAM2", f"Loading SAM2 model: {model_name}")
    
    model_config = sam2_model_list[model_name]
    model_filename = model_config["model_url"]
    config_filename = model_config["config"]
    
    # Get model path (download if needed via HuggingFace)
    download_path = os.path.join(folder_paths.models_dir, sam2_model_dir_name)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    model_path = os.path.join(download_path, model_filename)
    
    # Auto-download from HuggingFace if not exists
    if not os.path.exists(model_path):
        S4ToolLogger.info("ImageSAM2", f"Downloading SAM2 model from HuggingFace...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Kijai/sam2-safetensors",
            allow_patterns=[f"*{model_filename}*"],
            local_dir=download_path,
            local_dir_use_symlinks=False
        )
    
    # Get config path
    config_dir = os.path.join(parent_dir, "SAM2", "sam2_configs")
    config_path = os.path.join(config_dir, config_filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    
    # Determine dtype and device
    device = comfy.model_management.get_torch_device()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load SAM2 model
    sam2_model = load_sam2_model(
        model_path,
        config_path,
        segmentor='single_image',
        dtype=dtype,
        device=device
    )
    
    S4ToolLogger.success("ImageSAM2", f"SAM2 model loaded successfully: {model_name}")
    return sam2_model

def load_groundingdino_model(model_name):
    """Load GroundingDINO model from configuration"""
    S4ToolLogger.info("ImageSAM2", f"Loading GroundingDINO model: {model_name}")
    
    # Get and ensure config file exists
    config_path = get_local_filepath(
        groundingdino_model_list[model_name]["config_url"],
        groundingdino_model_dir_name
    )
    
    # Load model configuration
    dino_model_args = local_groundingdino_SLConfig.fromfile(config_path)
    
    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    # Build model
    dino = local_groundingdino_build_model(dino_model_args)
    
    # Get and ensure model file exists
    model_path = get_local_filepath(
        groundingdino_model_list[model_name]["model_url"],
        groundingdino_model_dir_name,
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    
    # Move to device and set to eval mode
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    
    S4ToolLogger.success("ImageSAM2", f"GroundingDINO model loaded successfully: {model_name}")
    return dino

def groundingdino_predict(dino_model, image, prompt, threshold, nms_threshold=0.8, box_padding=0, min_box_size=0):
    """GroundingDINO prediction with enhanced filtering"""
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        scores_filt = logits_filt.max(dim=1)[0]  # confidence scores
        return boxes_filt.cpu(), scores_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt, scores_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    
    if boxes_filt.size(0) == 0:
        return boxes_filt
    
    H, W = image.size[1], image.size[0]
    
    # Convert boxes from normalized to absolute coordinates
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # Apply NMS (Non-Maximum Suppression) if threshold < 1.0
    if nms_threshold < 1.0 and boxes_filt.size(0) > 1:
        from torchvision.ops import nms
        keep_indices = nms(boxes_filt, scores_filt, nms_threshold)
        boxes_filt = boxes_filt[keep_indices]
        S4ToolLogger.debug("ImageSAM2", f"NMS: kept {len(keep_indices)} out of {boxes_filt.size(0)} boxes")
    
    # Filter by minimum box size if enabled
    if min_box_size > 0:
        box_widths = boxes_filt[:, 2] - boxes_filt[:, 0]
        box_heights = boxes_filt[:, 3] - boxes_filt[:, 1]
        box_areas = box_widths * box_heights
        size_mask = box_areas >= (min_box_size * min_box_size)
        boxes_filt = boxes_filt[size_mask]
        S4ToolLogger.debug("ImageSAM2", f"Min size filter: kept {torch.sum(size_mask).item()} boxes")
    
    # Apply box padding (expand or shrink boxes)
    if box_padding != 0:
        boxes_filt[:, 0] = torch.clamp(boxes_filt[:, 0] - box_padding, min=0, max=W)  # x1
        boxes_filt[:, 1] = torch.clamp(boxes_filt[:, 1] - box_padding, min=0, max=H)  # y1
        boxes_filt[:, 2] = torch.clamp(boxes_filt[:, 2] + box_padding, min=0, max=W)  # x2
        boxes_filt[:, 3] = torch.clamp(boxes_filt[:, 3] + box_padding, min=0, max=H)  # y2
    
    return boxes_filt

def sam2_segment(sam2_model, image, boxes, quality_threshold=0.0, refinement_level="none", dtype=torch.float16, device=None):
    """SAM2 segmentation with enhanced features"""
    if boxes.shape[0] == 0:
        return None, None
    
    # Convert image to numpy array
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    
    sam_device = device if device is not None else comfy.model_management.get_torch_device()
    
    # Use autocast context to match dtype
    autocast_condition = not comfy.model_management.is_device_mps(sam_device)
    from contextlib import nullcontext
    
    with torch.autocast(comfy.model_management.get_autocast_device(sam_device), dtype=dtype) if autocast_condition else nullcontext():
        # Set image for SAM2
        sam2_model.set_image(image_np_rgb)
        
        # Convert boxes to proper format
        # boxes is torch.Tensor in XYXY format, need to convert to numpy
        boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
        
        # Predict for each box
        all_masks = []
        all_scores = []
        all_logits = []
        
        for box_np in boxes_np:
            # SAM2 predict expects single box in XYXY format as numpy array
            masks_single, scores_single, logits_single = sam2_model.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,  # Single box as numpy array in absolute coordinates
                multimask_output=False,
                normalize_coords=True  # Let SAM2 normalize the coords
            )
            all_masks.append(masks_single)
            all_scores.append(scores_single)
            all_logits.append(logits_single)
        
        # Stack results
        if len(all_masks) == 0:
            return None, None
        
        masks = np.concatenate(all_masks, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        
        # Debug: check mask content from SAM2
        S4ToolLogger.debug("ImageSAM2", f"SAM2 output - masks shape: {masks.shape}, dtype: {masks.dtype}")
        S4ToolLogger.debug("ImageSAM2", f"SAM2 output - mask values: min={masks.min()}, max={masks.max()}, unique={len(np.unique(masks))}")
        S4ToolLogger.debug("ImageSAM2", f"SAM2 output - scores: {scores}")
        
        # Refinement if enabled (simplified - use logits as mask input)
        if refinement_level != "none":
            iterations = {"light": 1, "standard": 1, "aggressive": 2}.get(refinement_level, 0)
            for iteration in range(iterations):
                refined_masks = []
                refined_scores = []
                
                for idx, box_np in enumerate(boxes_np):
                    # Use previous logits as mask input for refinement
                    prev_logits = all_logits[idx]
                    
                    masks_single, scores_single, logits_single = sam2_model.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_np,
                        mask_input=prev_logits,  # Use previous logits
                        multimask_output=False,
                        normalize_coords=True
                    )
                    refined_masks.append(masks_single)
                    refined_scores.append(scores_single)
                    all_logits[idx] = logits_single  # Update for next iteration
                
                if len(refined_masks) > 0:
                    masks = np.concatenate(refined_masks, axis=0)
                    scores = np.concatenate(refined_scores, axis=0)
    
    # Already in numpy format from predict()
    
    # Filter by quality threshold
    if quality_threshold > 0:
        high_quality_indices = scores >= quality_threshold
        if not np.any(high_quality_indices):
            S4ToolLogger.warning("ImageSAM2", f"No masks above quality threshold {quality_threshold}")
            return None, None
        masks = masks[high_quality_indices]
        boxes = boxes[high_quality_indices]
        scores = scores[high_quality_indices]
    
    return masks, boxes

def process_mask_enhancements(mask, fill_holes=True, remove_small_regions=0, smooth_edges=False, edge_expansion=0):
    """Apply post-processing enhancements to mask"""
    # Fill holes
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    
    # Remove small regions
    if remove_small_regions > 0:
        labeled, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        mask_sizes = sizes < remove_small_regions
        remove_pixels = mask_sizes[labeled]
        mask[remove_pixels] = 0
    
    # Edge expansion/contraction (do before smoothing for better results)
    if edge_expansion != 0:
        if edge_expansion > 0:
            # Dilation
            mask = ndimage.binary_dilation(mask, iterations=abs(edge_expansion))
        else:
            # Erosion
            mask = ndimage.binary_erosion(mask, iterations=abs(edge_expansion))
        mask = mask.astype(np.uint8)
    
    # Smooth edges - light processing here, main smoothing is in image application
    if smooth_edges:
        # Light morphological smoothing to remove tiny jaggies
        mask = ndimage.binary_closing(mask, iterations=1)
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = mask.astype(np.uint8)
    
    return mask

def create_tensor_output(image_np, masks, boxes_filt, fill_holes, remove_small_regions, smooth_edges, edge_expansion, smooth_strength=1.0):
    """Create tensor output from masks with enhancements"""
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    
    # masks shape is (N, H, W) where N is number of masks
    for idx in range(masks.shape[0]):
        # Get single mask (H, W)
        mask_enhanced = masks[idx]
        
        # Debug: check mask content
        S4ToolLogger.debug("ImageSAM2", f"Mask {idx}: shape={mask_enhanced.shape}, dtype={mask_enhanced.dtype}, unique_values={np.unique(mask_enhanced)}, sum={np.sum(mask_enhanced)}")
        
        # Ensure mask is boolean or uint8
        if mask_enhanced.dtype == bool:
            mask_enhanced = mask_enhanced.astype(np.uint8)
        elif mask_enhanced.dtype == float:
            mask_enhanced = (mask_enhanced > 0.5).astype(np.uint8)
        
        mask_enhanced = process_mask_enhancements(
            mask_enhanced,
            fill_holes=fill_holes,
            remove_small_regions=remove_small_regions,
            smooth_edges=smooth_edges,
            edge_expansion=edge_expansion
        )
        
        # Debug: check mask after enhancement
        S4ToolLogger.debug("ImageSAM2", f"Mask {idx} after enhancement: unique_values={np.unique(mask_enhanced)}, sum={np.sum(mask_enhanced)}")
        
        # Create output image
        image_np_copy = copy.deepcopy(image_np)
        
        # Apply mask with smooth edges if enabled
        if smooth_edges and smooth_strength > 0:
            # Create soft alpha mask with adjustable strength
            # smooth_strength: 0 = sharp, 1 = natural, 2-5 = progressively softer
            mask_float = mask_enhanced.astype(float)
            soft_mask = ndimage.gaussian_filter(mask_float, sigma=smooth_strength)
            # Normalize to 0-1 range
            soft_mask = np.clip(soft_mask, 0, 1)
            
            # Apply soft mask to alpha channel for smooth edges
            image_np_copy[:, :, 3] = (soft_mask * 255).astype(np.uint8)
            
            # Also apply to RGB to avoid hard edges
            for c in range(3):
                image_np_copy[:, :, c] = (image_np_copy[:, :, c] * soft_mask).astype(np.uint8)
        else:
            # Hard edge mode: use binary mask
            mask_bool = mask_enhanced.astype(bool)
            # Set non-masked (background) areas to transparent [0, 0, 0, 0]
            image_np_copy[~mask_bool] = [0, 0, 0, 0]
        
        S4ToolLogger.debug("ImageSAM2", f"Output image {idx}: non-zero pixels={np.count_nonzero(image_np_copy)}")
        
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy)
        )
        output_masks.append(output_mask)
        output_images.append(output_image)
    
    return (output_images, output_masks)

def split_image_mask(image):
    """Split image and mask"""
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)

class ImageSAM2:
    """SAM2 node combining GroundingDINO and SAM2 for text-prompt driven segmentation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model_name": (list(sam2_model_list.keys()), {"default": "sam2.1_hiera_small (200MB)"}),
                "grounding_dino_model_name": (list(groundingdino_model_list.keys()), {"default": "GroundingDINO_SwinT_OGC (694MB)"}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"default": "object"}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "quality_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Filter out low-quality segmentations (0 = no filtering)"
                }),
                "refinement_level": (["none", "light", "standard", "aggressive"], {
                    "default": "standard",
                    "tooltip": "Mask refinement quality: none (fastest) to aggressive (best quality)"
                }),
                "edge_expansion": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Expand (positive) or shrink (negative) mask edges"
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill small holes in masks"
                }),
                "remove_small_regions": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Remove regions smaller than N pixels (0 = disabled)"
                }),
                "smooth_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth mask edges"
                }),
                "smooth_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Edge smoothing strength (0 = sharp, 5 = very soft)"
                }),
                "max_detections": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of detection results to output (0 = unlimited)"
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Non-Maximum Suppression threshold for removing overlapping boxes (1.0 = disabled)"
                }),
                "box_padding": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Expand (positive) or shrink (negative) detection boxes in pixels"
                }),
                "min_box_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "tooltip": "Filter out detection boxes smaller than N pixels (0 = disabled)"
                }),
            }
        }
    
    CATEGORY = "💀S4Tool"
    FUNCTION = "segment"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented_image", "mask")
    
    def __init__(self):
        self._sam2_model = None
        self._sam2_model_name = None
        self._dino_model = None
        self._dino_model_name = None
    
    def segment(self, sam2_model_name, grounding_dino_model_name, image, prompt, threshold,
                quality_threshold, refinement_level, edge_expansion, fill_holes, 
                remove_small_regions, smooth_edges, smooth_strength, max_detections,
                nms_threshold, box_padding, min_box_size):
        """Main segmentation function"""
        try:
            S4ToolLogger.info("ImageSAM2", f"Starting SAM2 segmentation with prompt: '{prompt}' threshold: {threshold}")
            
            # Load SAM2 model if needed
            if self._sam2_model is None or self._sam2_model_name != sam2_model_name:
                self._sam2_model = load_sam2_model_wrapper(sam2_model_name)
                self._sam2_model_name = sam2_model_name
            
            # Load GroundingDINO model if needed
            if self._dino_model is None or self._dino_model_name != grounding_dino_model_name:
                self._dino_model = load_groundingdino_model(grounding_dino_model_name)
                self._dino_model_name = grounding_dino_model_name
            
            res_images = []
            res_masks = []
            
            for idx, item in enumerate(image):
                S4ToolLogger.debug("ImageSAM2", f"Processing image {idx + 1}/{len(image)}")
                
                item = Image.fromarray(
                    np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                
                # GroundingDINO prediction with enhanced filtering
                boxes = groundingdino_predict(
                    self._dino_model,
                    item,
                    prompt,
                    threshold,
                    nms_threshold=nms_threshold,
                    box_padding=box_padding,
                    min_box_size=min_box_size
                )
                
                if boxes.shape[0] == 0:
                    S4ToolLogger.warning("ImageSAM2", f"No objects detected for image {idx + 1}")
                    continue
                
                S4ToolLogger.info("ImageSAM2", f"Found {boxes.shape[0]} objects in image {idx + 1}")
                
                # SAM2 segmentation with enhancements
                # Get dtype and device from cached model info
                sam_device = comfy.model_management.get_torch_device()
                sam_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                masks, filtered_boxes = sam2_segment(
                    self._sam2_model,
                    item,
                    boxes,
                    quality_threshold=quality_threshold,
                    refinement_level=refinement_level,
                    dtype=sam_dtype,
                    device=sam_device
                )
                
                if masks is None:
                    S4ToolLogger.warning("ImageSAM2", f"No valid masks for image {idx + 1}")
                    continue
                
                # Apply max_detections limit after quality filtering
                if max_detections > 0 and masks.shape[0] > max_detections:
                    S4ToolLogger.info("ImageSAM2", f"Limiting output from {masks.shape[0]} to {max_detections} highest quality masks")
                    masks = masks[:max_detections]
                    filtered_boxes = filtered_boxes[:max_detections]
                
                # Create output with post-processing
                result = create_tensor_output(
                    np.array(item),
                    masks,
                    filtered_boxes,
                    fill_holes=fill_holes,
                    remove_small_regions=remove_small_regions,
                    smooth_edges=smooth_edges,
                    edge_expansion=edge_expansion,
                    smooth_strength=smooth_strength
                )
                
                if result:
                    images, masks_out = result
                    res_images.extend(images)
                    res_masks.extend(masks_out)
            
            if len(res_images) == 0:
                S4ToolLogger.warning("ImageSAM2", "No valid segmentations found, returning empty mask")
                _, height, width, _ = image.size()
                empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
                return (empty_mask, empty_mask)
            
            S4ToolLogger.success("ImageSAM2", f"SAM2 segmentation completed successfully, {len(res_images)} segments found")
            return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))
            
        except Exception as e:
            S4ToolLogger.error("ImageSAM2", f"SAM2 segmentation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
            return (empty_mask, empty_mask)