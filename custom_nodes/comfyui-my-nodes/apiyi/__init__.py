from .nodes import APIYiGPT
from .fallback import APIYiGPTFallback
from .chat import APIYiVisionChat

NODE_CLASS_MAPPINGS = {
    "APIYiGPT": APIYiGPT,
    "APIYiGPTFallback": APIYiGPTFallback,
    "APIYiVisionChat": APIYiVisionChat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "APIYiGPT": "API易GPT",
    "APIYiGPTFallback": "API易GPT（官方备用）",
    "APIYiVisionChat": "API易多模态聊天（识图）",
}
WEB_DIRECTORY = "./web"
