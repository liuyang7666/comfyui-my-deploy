"""千问 2.1 图像补边：提供全中文节点，并兼容旧版英文节点。"""

import torch.nn.functional as F


def _pad_to_multiple(image, multiple, mode):
    """仅向右、向下补边，不改变裁剪图已有像素及左上角位置。"""
    _, height, width, _ = image.shape
    pad_right = (-width) % multiple
    pad_bottom = (-height) % multiple
    if not pad_right and not pad_bottom:
        return (image,)
    tensor = image.movedim(-1, 1)
    if mode == "reflect" and (pad_right >= width or pad_bottom >= height):
        # PyTorch 镜像补边不能大于或等于原维度，回退到复制边缘。
        mode = "replicate"
    result = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)
    return (result.movedim(1, -1),)


class PadImageToMultipleZH:
    """全中文节点：变量名、枚举和节点菜单都显示中文。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "输入图像": ("IMAGE", {"tooltip": "逐个人物裁剪图，或其他需要补边的图片"}),
            "补边倍数": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1,
                       "tooltip": "Qwen Image 2.1 推荐使用 32"}),
            "补边模式": (["复制边缘", "镜像补边", "补黑边"], {"default": "复制边缘",
                       "tooltip": "复制边缘最适合人物裁剪图；不拉伸原图"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("补边后图像",)
    FUNCTION = "pad"
    CATEGORY = "图像处理/千问2.1"
    DESCRIPTION = "向图像右侧及底部补边到指定倍数，尺寸满足 32 倍数，不缩放、不裁切人物。"

    def pad(self, 输入图像, 补边倍数=32, 补边模式="复制边缘"):
        mode = {"复制边缘": "replicate", "镜像补边": "reflect", "补黑边": "constant"}[补边模式]
        return _pad_to_multiple(输入图像, 补边倍数, mode)

NODE_CLASS_MAPPINGS = {
    "PadImageToMultipleZH": PadImageToMultipleZH,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageToMultipleZH": "图像自动补边",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
