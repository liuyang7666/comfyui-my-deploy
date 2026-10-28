"""One SAM3 image encoding per image; all text prompts share the same features."""
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

from .core import REGIONS

PROMPTS = {'skin': ('exposed human skin',), 'sky': ('sky',),
           'vegetation': ('plants',), 'grass': ('grass',), 'trees': ('trees',), 'water': ('water',),
           'architecture': ('buildings',), 'person': ('person',)}
MODEL = 'sam3.1_multiplex_fp16'
SEGMENTATION_VERSION = 'shared-image-features-colour-selection-v5'


def encode_masks(images, processor, device, interrupt=lambda: None):
    context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    results = []
    with torch.inference_mode(), context:
        for image in images:
            interrupt()
            pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
            state = processor.set_image(pil)
            processor.set_confidence_threshold(.5)
            masks = {}
            for name in PROMPTS:
                union = np.zeros(image.shape[:2], np.float32)
                for prompt in PROMPTS[name]:
                    interrupt()
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt, state)
                    # SAM3's per-pixel sigmoid is not calibrated object confidence.
                    # Its accepted binary mask + inward erosion gives representative interiors.
                    selected = state.get('masks')
                    if selected is not None and selected.numel():
                        merged = selected.float().reshape(-1, *image.shape[:2]).amax(dim=0)
                        union = np.maximum(union, merged.cpu().numpy())
                masks[name] = union
            masks['vegetation'] *= 1 - np.maximum(masks['grass'], masks['trees'])
            results.append(masks)
            del state
    return results


def segment_images(images):
    import folder_paths
    import nodes
    import comfy.model_management as mm

    cls = nodes.NODE_CLASS_MAPPINGS.get('SAM3Segment')
    if cls is None:
        raise RuntimeError('需要已加载的 RMBG SAM3Segment 节点。')
    if folder_paths.get_full_path('sam3', MODEL + '.safetensors') is None:
        raise RuntimeError('未找到本地 sam3.1_multiplex_fp16.safetensors；不会自动下载权重。')
    segmenter = cls()
    processor, device = segmenter._load_processor(MODEL, 'Auto')
    try:
        return encode_masks(images, processor, device, mm.throw_exception_if_processing_interrupted)
    finally:
        segmenter.processor_cache.clear()
