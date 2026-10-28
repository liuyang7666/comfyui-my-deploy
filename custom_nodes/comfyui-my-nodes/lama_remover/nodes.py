"""MY package integration of the ComfyUI-RMBG LaMa object-removal node.

The processing implementation is derived from AILab-AI/ComfyUI-RMBG v3.2.0.
The model is shared from ComfyUI/models/RMBG/Lama/big-lama.pt.
"""

import gc
import os
import shutil
import traceback

import folder_paths
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

import comfy.model_management as model_management


DEVICE = model_management.get_torch_device()


def pil2comfy(image):
    img_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0:
        w += 8 - w % 8
    if h % 8 != 0:
        h += 8 - h % 8

    fill_color = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill_color)
    padded.paste(image, (0, 0))
    return padded


def cropimage(image, w, h):
    return image.crop((0, 0, w, h))


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def clean_vram():
    gc.collect()
    model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


class MYLamaRemover:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "images": "要进行 LaMa 修复的图像。",
            "masks": "移除区域遮罩：白色为移除，黑色为保留。本节点不自动识别人。",
            "removal_strength": "控制移除区域的阈值范围；数值越高，作用区域可能越大。",
            "edge_smoothness": "控制遮罩边缘平滑程度；数值越高，过渡越平滑。",
        }
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": tooltips["images"]}),
                "masks": ("MASK", {"tooltip": tooltips["masks"]}),
                "removal_strength": (
                    "INT",
                    {
                        "default": 230,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "display": "slider",
                        "tooltip": tooltips["removal_strength"],
                    },
                ),
                "edge_smoothness": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 20,
                        "step": 1,
                        "display": "slider",
                        "tooltip": tooltips["edge_smoothness"],
                    },
                ),
            },
            "optional": {
                "unload_model": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "执行后从显存卸载模型。"},
                ),
            },
        }

    CATEGORY = "MY/移除路人"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_object"

    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "Lama")
        self.model_path = os.path.join(self.cache_dir, "big-lama.pt")
        self.to_pil = transforms.ToPILImage()

    def load_model(self):
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            self.download_model()

        try:
            # Loading from a binary stream avoids PyTorch's Windows Unicode-path
            # fopen failure when the ComfyUI install path contains Chinese text.
            with open(self.model_path, "rb") as model_file:
                self.model = torch.jit.load(model_file, map_location=self.device)
        except Exception as e:
            print(f"Can't use comfy device: {str(e)}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            with open(self.model_path, "rb") as model_file:
                self.model = torch.jit.load(model_file, map_location=self.device)

        self.model.eval()
        self.model.to(self.device)

    def download_model(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        print("Downloading Big-Lama model...")
        try:
            downloaded_path = hf_hub_download(
                repo_id="1038lab/Lama",
                filename="big-lama.pt",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            if os.path.dirname(downloaded_path) != self.cache_dir:
                target_path = os.path.join(self.cache_dir, "big-lama.pt")
                shutil.move(downloaded_path, target_path)
            print("Model downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error downloading Big-Lama model: {str(e)}") from e

    def process_with_model(self, img_tensor, mask_tensor):
        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        with torch.no_grad():
            result = self.model(img_tensor, mask_tensor)
            result_cpu = result.cpu()
            del img_tensor
            del mask_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result_cpu

    def remove_object(self, images, masks, removal_strength, edge_smoothness, unload_model=False):
        try:
            self.load_model()
            results = []

            for image, mask in zip(images, masks):
                ori_image = tensor2pil(image)
                w, h = ori_image.size
                p_image = pad_image(ori_image)

                mask_np = mask.cpu().numpy()
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                p_mask = pad_image(mask_pil, is_mask=True)

                if p_mask.size != p_image.size:
                    try:
                        p_mask = p_mask.resize(p_image.size, Image.LANCZOS)
                    except AttributeError:
                        p_mask = p_mask.resize(p_image.size, Image.ANTIALIAS)

                p_mask = ImageOps.invert(p_mask)
                p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
                gray = p_mask.point(lambda x: 0 if x > removal_strength else 255)

                img_tensor = (
                    torch.FloatTensor(np.array(p_image))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                mask_tensor = (
                    torch.FloatTensor(np.array(gray)).unsqueeze(0).unsqueeze(0) / 255.0
                )

                result = self.process_with_model(img_tensor, mask_tensor)
                result_img = self.to_pil(result.squeeze())

                if result_img.width > w or result_img.height > h:
                    result_img = cropimage(result_img, w, h)

                results.append(pil2comfy(result_img))
                del result
                gc.collect()

            return (torch.cat(results, dim=0),)

        except Exception as e:
            print(traceback.format_exc())
            raise RuntimeError(f"Error in object removal: {str(e)}") from e
        finally:
            if unload_model:
                self.model = None
            clean_vram()


NODE_CLASS_MAPPINGS = {"MYLamaRemover": MYLamaRemover}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MYLamaRemover": "移除路人（LaMa，遮罩驱动）",
}
