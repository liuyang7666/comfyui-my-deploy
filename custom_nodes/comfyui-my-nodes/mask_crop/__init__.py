from .nodes import MaskCropByMultiple

NODE_CLASS_MAPPINGS = {"MYImageCropByMaskMultiple": MaskCropByMultiple}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MYImageCropByMaskMultiple": "遮罩裁剪（倍数）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
