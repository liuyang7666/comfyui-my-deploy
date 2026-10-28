from .nodes import ConcurrentImageCollect, ConcurrentImageSubmit

NODE_CLASS_MAPPINGS = {"ConcurrentImageSubmit": ConcurrentImageSubmit, "ConcurrentImageCollect": ConcurrentImageCollect}
NODE_DISPLAY_NAME_MAPPINGS = {"ConcurrentImageSubmit": "分块并发提交", "ConcurrentImageCollect": "分块并发取回（按序）"}
WEB_DIRECTORY = "./web"
