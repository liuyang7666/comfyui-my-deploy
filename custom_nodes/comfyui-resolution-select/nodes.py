import torch.nn.functional as F


RESOLUTIONS = {f"{k}K": k * 1024 for k in (1, 2, 4, 6, 8, 12)}


def target_size(width, height, resolution):
    long_edge = RESOLUTIONS[resolution]
    if width >= height:
        short_edge = max(16, ((height * long_edge + width * 8) // (width * 16)) * 16)
        return long_edge, short_edge
    short_edge = max(16, ((width * long_edge + height * 8) // (height * 16)) * 16)
    return short_edge, long_edge


class ResolutionSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "resolution": (list(RESOLUTIONS), {
                "default": "2K",
                "tooltip": "长边：1K=1024、2K=2048、4K=4096、6K=6144、8K=8192、12K=12288 像素。短边按原比例取最近的16倍数，最小16；不裁剪、不补边。",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("图像", "宽度", "高度")
    FUNCTION = "resize"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "按所选K值缩放长边，宽高均为16倍数，尽量保持原比例。使用浮点双三次抗锯齿缩放，支持图像批次。"
    SEARCH_ALIASES = ["分辨率选择", "resolution select", "1K", "2K", "4K", "6K", "8K", "12K"]

    def resize(self, image, resolution):
        height, width = image.shape[1:3]
        new_width, new_height = target_size(width, height, resolution)
        if (width, height) == (new_width, new_height):
            return image, new_width, new_height
        result = F.interpolate(
            image.movedim(-1, 1), size=(new_height, new_width),
            mode="bicubic", align_corners=False, antialias=True,
        )
        return result.clamp_(0, 1).movedim(1, -1), new_width, new_height
