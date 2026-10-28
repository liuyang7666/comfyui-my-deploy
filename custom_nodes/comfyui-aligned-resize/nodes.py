import torch
import torch.nn.functional as F


ALIGNMENT_TYPE = "PIXEL_ALIGNED_RESIZE_INFO"


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def resample_uniform(image, width, height, numerator, denominator):
    """Resample on one uniform pixel-center grid, independent of rounded sizes."""
    if numerator == denominator and image.shape[1:3] == (height, width):
        return image

    pixels = image.movedim(-1, 1)
    # PyTorch's antialiased bicubic kernel requires float32/float64.
    if pixels.dtype in (torch.float16, torch.bfloat16):
        pixels = pixels.float()
    scale = numerator / denominator
    targets = {2: height, 3: width}
    # Shrink the longer axis first; enlarge the shorter axis first. Padding a
    # very thin image then never creates a huge two-dimensional source canvas.
    axes = sorted((2, 3), key=lambda axis: pixels.shape[axis], reverse=scale < 1)
    for axis in axes:
        length = targets[axis]
        if numerator == denominator:
            extra = max(0, length - pixels.shape[axis])
        else:
            # An extra source pixel covers floating-point floor at an exact
            # output boundary. Replication also covers the fractional last row.
            extra = max(0, ceil_div(length * denominator, numerator) + 1 - pixels.shape[axis])
        if extra:
            padding = (0, extra, 0, 0) if axis == 3 else (0, 0, 0, extra)
            pixels = F.pad(pixels, padding, mode="replicate")
        if numerator != denominator:
            factors = (1.0, scale) if axis == 3 else (scale, 1.0)
            pixels = F.interpolate(
                pixels, scale_factor=factors, mode="bicubic",
                align_corners=False, recompute_scale_factor=False, antialias=True,
            )
        pixels = pixels[:, :, :, :length] if axis == 3 else pixels[:, :, :length, :]

    return pixels.clamp(0, 1).movedim(1, -1).to(dtype=image.dtype).contiguous()


class PixelAlignedResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "long_edge": ("INT", {
                "default": 2048, "min": 32, "max": 32768, "step": 1,
                "tooltip": "目标长边像素：常用2048、1536、1024，也可输入其他整数。",
            }),
            "short_multiple": (["32", "16"], {
                "default": "32",
                "tooltip": "短边向上补齐到该倍数；在右侧/底部复制边缘补边，不拉伸、不裁掉原图内容。",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", ALIGNMENT_TYPE)
    RETURN_NAMES = ("缩放图像", "缩放前宽", "缩放前高", "对齐信息")
    OUTPUT_TOOLTIPS = (
        "保持统一比例的图像，短边含少量补边。",
        "本节点输入图像的实际宽度，不读取文件EXIF。",
        "本节点输入图像的实际高度，不读取文件EXIF。",
        "连接到对齐还原节点，保存输入尺寸、补边画布及精确比例。",
    )
    FUNCTION = "resize"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "长边缩放到指定像素，短边补齐16/32倍数，保留统一缩放比例及输入宽高。"
        "编辑后用配套对齐还原节点恢复坐标；不要仅将含补边的图像直接拉回输入宽高。"
    )
    SEARCH_ALIASES = ["对齐缩放", "无偏移缩放", "aligned resize", "2048", "1536", "1024"]

    def resize(self, image, long_edge=2048, short_multiple="32"):
        multiple = int(short_multiple)
        if multiple not in (16, 32) or long_edge < multiple:
            raise ValueError("短边倍数须为16或32，目标长边不能小于短边倍数。")
        height, width = image.shape[1:3]
        source_long = max(width, height)
        numerator, denominator = long_edge, source_long
        short_limit = (long_edge // multiple) * multiple
        # A near-square image at a custom non-multiple edge needs a little
        # padding on the long axis too, or its rounded short edge would be longer.
        if min(width, height) * numerator > short_limit * denominator:
            numerator, denominator = short_limit, min(width, height)
        content_width = ceil_div(width * numerator, denominator)
        content_height = ceil_div(height * numerator, denominator)
        if width >= height:
            canvas_width = long_edge
            canvas_height = ceil_div(content_height, multiple) * multiple
        else:
            canvas_width = ceil_div(content_width, multiple) * multiple
            canvas_height = long_edge

        result = resample_uniform(image, content_width, content_height, numerator, denominator)
        pad_right, pad_bottom = canvas_width - content_width, canvas_height - content_height
        if pad_right or pad_bottom:
            result = F.pad(
                result.movedim(-1, 1), (0, pad_right, 0, pad_bottom), mode="replicate",
            ).movedim(1, -1).contiguous()
        info = {
            "version": 1,
            "input_width": width, "input_height": height,
            "canvas_width": canvas_width, "canvas_height": canvas_height,
            "scale_numerator": numerator, "scale_denominator": denominator,
        }
        return result, width, height, info


class PixelAlignedRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {
                "tooltip": "编辑后的图像，须与缩放节点输出保持相同画布尺寸及构图。",
            }),
            "alignment": (ALIGNMENT_TYPE, {
                "tooltip": "连接同一次对齐缩放的“对齐信息”。还原时自动移除补边的坐标范围。",
            }),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("还原图像",)
    FUNCTION = "restore"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "按保存的统一比例、像素中心和输入宽高还原，排除右/下补边。"
        "保证缩放坐标映射一致；不能无损恢复缩小丢失的细节，不能纠正AI编辑导致的结构位移。"
    )
    SEARCH_ALIASES = ["对齐还原", "恢复尺寸", "aligned restore"]

    def restore(self, image, alignment):
        if alignment["version"] != 1:
            raise ValueError("不支持的对齐信息版本，请重新运行对齐缩放节点。")
        expected = (alignment["canvas_height"], alignment["canvas_width"])
        if image.shape[1:3] != expected:
            raise ValueError(
                f"编辑图像为{image.shape[2]}×{image.shape[1]}，应保持"
                f"{expected[1]}×{expected[0]}。请关闭编辑节点的自动裁剪/改尺寸后再还原。"
            )
        content_width = ceil_div(
            alignment["input_width"] * alignment["scale_numerator"], alignment["scale_denominator"],
        )
        content_height = ceil_div(
            alignment["input_height"] * alignment["scale_numerator"], alignment["scale_denominator"],
        )
        image = image[:, :content_height, :content_width, :]
        return (resample_uniform(
            image, alignment["input_width"], alignment["input_height"],
            alignment["scale_denominator"], alignment["scale_numerator"],
        ),)
