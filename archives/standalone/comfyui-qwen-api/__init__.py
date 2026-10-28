from .nodes import QwenAPI
from .image_nodes import QwenImageAPI

NODE_CLASS_MAPPINGS = {"QwenAPI": QwenAPI, "QwenImageAPI": QwenImageAPI}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenAPI": "千问 API · 对话 / 图片理解",
    "QwenImageAPI": "千问 Image 3.0 / Pro · 生图",
}
