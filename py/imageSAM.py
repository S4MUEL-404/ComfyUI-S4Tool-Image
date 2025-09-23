"""
S4SAM Image - Unified Segment Anything Model for ComfyUI
Integrated with S4Tool-Image Production Quality Suite

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

# Import dependency manager from parent module
from ..dependency_manager import require_dependency, S4ToolLogger

# Add current path and parent path to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import SAM-HQ and GroundingDINO modules using sys.path (same as original)
# Use importlib for SAM-HQ because directory name has hyphen
import importlib.util

# Load SAM-HQ modules dynamically
sam_hq_dir = os.path.join(parent_dir, 'SAM-HQ')

# Load predictor module
predictor_spec = importlib.util.spec_from_file_location(
    "sam_hq_predictor", 
    os.path.join(sam_hq_dir, 'predictor.py')
)
predictor_module = importlib.util.module_from_spec(predictor_spec)
predictor_spec.loader.exec_module(predictor_module)
SamPredictorHQ = predictor_module.SamPredictorHQ

# Load build_sam_hq module
build_sam_spec = importlib.util.spec_from_file_location(
    "build_sam_hq", 
    os.path.join(sam_hq_dir, 'build_sam_hq.py')
)
build_sam_module = importlib.util.module_from_spec(build_sam_spec)
build_sam_spec.loader.exec_module(build_sam_module)
sam_model_registry = build_sam_module.sam_model_registry
from GroundingDINO.datasets import transforms as T
from GroundingDINO.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from GroundingDINO.util.slconfig import SLConfig as local_groundingdino_SLConfig
from GroundingDINO.models import build_model as local_groundingdino_build_model

# Check SAM dependencies at import
require_dependency('segment_anything', 'SAM Core Functionality')
require_dependency('timm', 'Vision Transformer Support')
require_dependency('addict', 'Configuration Management')

# Model configuration
sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

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
        S4ToolLogger.info("ImageSAM", "Using models/bert-base-uncased")
        return comfy_bert_model_base
    return 'bert-base-uncased'

def get_local_filepath(url, dirname, local_file_name=None):
    """Download or get local file path"""
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        S4ToolLogger.info("ImageSAM", f"Using existing model: {destination}")
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        S4ToolLogger.info("ImageSAM", f"Downloading {url} to {destination}")
        download_url_to_file(url, destination)
    return destination

def load_sam_model(model_name):
    """Load SAM model from configuration"""
    S4ToolLogger.info("ImageSAM", f"Loading SAM model: {model_name}")
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    S4ToolLogger.success("ImageSAM", f"SAM model loaded successfully: {model_name}")
    return sam

def load_groundingdino_model(model_name):
    """Load GroundingDINO model from configuration"""
    S4ToolLogger.info("ImageSAM", f"Loading GroundingDINO model: {model_name}")
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    S4ToolLogger.success("ImageSAM", f"GroundingDINO model loaded successfully: {model_name}")
    return dino

def groundingdino_predict(dino_model, image, prompt, threshold):
    """GroundingDINO prediction"""
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
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt

def sam_segment(sam_model, image, boxes):
    """SAM segmentation"""
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # Check if it's HQ SAM
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)

def create_tensor_output(image_np, masks, boxes_filt):
    """Create tensor output from masks"""
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
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

class ImageSAM:
    """Unified SAM node combining SAM and GroundingDINO functionality"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model_name": (list(sam_model_list.keys()), {"default": "sam_vit_b (375MB)"}),
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
            }
        }
    
    CATEGORY = "💀S4Tool"
    FUNCTION = "segment"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented_image", "mask")
    
    def __init__(self):
        self._sam_model = None
        self._sam_model_name = None
        self._dino_model = None
        self._dino_model_name = None
    
    def segment(self, sam_model_name, grounding_dino_model_name, image, prompt, threshold):
        """Main segmentation function"""
        try:
            S4ToolLogger.info("ImageSAM", f"Starting segmentation with prompt: '{prompt}' threshold: {threshold}")
            
            # Load SAM model if needed
            if self._sam_model is None or self._sam_model_name != sam_model_name:
                self._sam_model = load_sam_model(sam_model_name)
                self._sam_model_name = sam_model_name
            
            # Load GroundingDINO model if needed
            if self._dino_model is None or self._dino_model_name != grounding_dino_model_name:
                self._dino_model = load_groundingdino_model(grounding_dino_model_name)
                self._dino_model_name = grounding_dino_model_name
            
            res_images = []
            res_masks = []
            
            for idx, item in enumerate(image):
                S4ToolLogger.debug("ImageSAM", f"Processing image {idx + 1}/{len(image)}")
                
                item = Image.fromarray(
                    np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                
                # GroundingDINO prediction
                boxes = groundingdino_predict(
                    self._dino_model,
                    item,
                    prompt,
                    threshold
                )
                
                if boxes.shape[0] == 0:
                    S4ToolLogger.warning("ImageSAM", f"No objects detected for image {idx + 1}")
                    continue
                
                S4ToolLogger.info("ImageSAM", f"Found {boxes.shape[0]} objects in image {idx + 1}")
                
                # SAM segmentation
                result = sam_segment(
                    self._sam_model,
                    item,
                    boxes
                )
                
                if result:
                    images, masks = result
                    res_images.extend(images)
                    res_masks.extend(masks)
            
            if len(res_images) == 0:
                S4ToolLogger.warning("ImageSAM", "No valid segmentations found, returning empty mask")
                _, height, width, _ = image.size()
                empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
                return (empty_mask, empty_mask)
            
            S4ToolLogger.success("ImageSAM", f"Segmentation completed successfully, {len(res_images)} segments found")
            return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))
            
        except Exception as e:
            S4ToolLogger.error("ImageSAM", f"Segmentation failed: {str(e)}")
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)