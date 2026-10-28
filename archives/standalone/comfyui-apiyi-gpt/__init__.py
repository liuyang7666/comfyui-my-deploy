from .nodes import APIYiGPT
from .fallback import APIYiGPTFallback

NODE_CLASS_MAPPINGS = {"APIYiGPT": APIYiGPT, "APIYiGPTFallback": APIYiGPTFallback}
NODE_DISPLAY_NAME_MAPPINGS = {"APIYiGPT": "API易GPT", "APIYiGPTFallback": "API易GPT（官方备用）"}
WEB_DIRECTORY = "./web"
