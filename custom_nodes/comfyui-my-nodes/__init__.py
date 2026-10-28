"""MY: API易、千问API、自动分块、图像对齐、遮罩处理与LaMa移除。"""
from .apiyi import NODE_CLASS_MAPPINGS as _apiyi, NODE_DISPLAY_NAME_MAPPINGS as _apiyi_names
from .image25 import NODE_CLASS_MAPPINGS as _tiles, NODE_DISPLAY_NAME_MAPPINGS as _tile_names
from .qwen import NODE_CLASS_MAPPINGS as _qwen, NODE_DISPLAY_NAME_MAPPINGS as _qwen_names
from .pixel_drift import NODE_CLASS_MAPPINGS as _align, NODE_DISPLAY_NAME_MAPPINGS as _align_names
from .mask_crop import NODE_CLASS_MAPPINGS as _mask_crop, NODE_DISPLAY_NAME_MAPPINGS as _mask_crop_names
from .mask_seam import NODE_CLASS_MAPPINGS as _mask_seam, NODE_DISPLAY_NAME_MAPPINGS as _mask_seam_names
from .mask_router import NODE_CLASS_MAPPINGS as _mask_router, NODE_DISPLAY_NAME_MAPPINGS as _mask_router_names
from .lama_remover import NODE_CLASS_MAPPINGS as _lama, NODE_DISPLAY_NAME_MAPPINGS as _lama_names

NODE_CLASS_MAPPINGS = {**_apiyi, **_tiles, **_qwen, **_align, **_mask_crop, **_mask_seam, **_mask_router, **_lama}
NODE_DISPLAY_NAME_MAPPINGS = {**_apiyi_names, **_tile_names, **_qwen_names, **_align_names, **_mask_crop_names, **_mask_seam_names, **_mask_router_names, **_lama_names}
WEB_DIRECTORY = "./web"
__version__ = "1.8.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .skin_blemish import NODE_CLASS_MAPPINGS as _skin_blemish, NODE_DISPLAY_NAME_MAPPINGS as _skin_blemish_names
NODE_CLASS_MAPPINGS.update(_skin_blemish)
NODE_DISPLAY_NAME_MAPPINGS.update(_skin_blemish_names)


from .dropdown_select import NODE_CLASS_MAPPINGS as _dropdown, NODE_DISPLAY_NAME_MAPPINGS as _dropdown_names
NODE_CLASS_MAPPINGS.update(_dropdown)
NODE_DISPLAY_NAME_MAPPINGS.update(_dropdown_names)
