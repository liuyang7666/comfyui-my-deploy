"""Local semantic hair protection; processes the whole image, no face detection."""
from pathlib import Path
import threading
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_model=None
_processor=None
_lock=threading.Lock()

def hair_mask(image):
    global _model,_processor
    with _lock:
        if _model is None:
            import folder_paths
            from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
            directories=[]
            if 'face_parsing' in folder_paths.folder_names_and_paths:
                directories.extend(folder_paths.get_folder_paths('face_parsing'))
            directories.append(str(Path(folder_paths.models_dir)/'face_parsing'))
            path=next((Path(p) for p in directories if (Path(p)/'model.safetensors').is_file()),None)
            if path is None:
                raise RuntimeError('自动头发保护需要本地face_parsing模型；未找到模型，请接入保护蒙版或安装模型。')
            _processor=SegformerImageProcessor.from_pretrained(str(path),local_files_only=True)
            _model=AutoModelForSemanticSegmentation.from_pretrained(str(path),local_files_only=True).eval()
        results=[]
        for rgb in image.detach().float().cpu().numpy():
            im=Image.fromarray((rgb.clip(0,1)*255).astype(np.uint8))
            inputs=_processor(images=im,return_tensors='pt')
            with torch.inference_mode():
                logits=_model(**inputs).logits
                probabilities=logits.softmax(1)
                # Include uncertain hair boundary pixels, with no blanket dilation.
                hair=probabilities[:,13:14]
                hair=F.interpolate(hair,size=rgb.shape[:2],mode='bilinear',align_corners=False)[0,0]
                results.append((hair>=.20).float())
        return torch.stack(results).to(image.device)
