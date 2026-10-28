from .nodes import PixelAlignedResize, PixelAlignedRestore

NODE_CLASS_MAPPINGS = {
    "PixelAlignedResize": PixelAlignedResize,
    "PixelAlignedRestore": PixelAlignedRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAlignedResize": "对齐缩放（长边·16/32）",
    "PixelAlignedRestore": "对齐还原（编辑后）",
}
