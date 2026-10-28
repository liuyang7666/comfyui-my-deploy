"""Build a narrow mask around the boundary of an input mask."""

import math

import torch
import torch.nn.functional as F


class RemoveMaskSeam:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seam_width": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "接缝遮罩总宽度（像素）。偶数宽度在原遮罩内外各分一半；例如6像素为内缩3、外扩3。",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
                "bboxes": (
                    "BBOX",
                    {"tooltip": "XYWH边界框，可直接连接遮罩裁剪节点的边界框输出。"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("seam_mask",)
    FUNCTION = "create_mask"
    CATEGORY = "MY/遮罩处理"
    DESCRIPTION = "由输入遮罩或XYWH边界框生成接缝遮罩，适合连接到 Lama 等修复节点。"

    @staticmethod
    def _dilate(binary_mask, radius):
        if radius == 0:
            return binary_mask
        padded = F.pad(binary_mask, (radius, radius, radius, radius), value=0.0)
        return F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1)

    @staticmethod
    def _erode(binary_mask, radius):
        if radius == 0:
            return binary_mask
        inverse = 1.0 - binary_mask
        # Treat pixels beyond the image as background in the original mask.
        padded = F.pad(inverse, (radius, radius, radius, radius), value=1.0)
        return 1.0 - F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1)

    @staticmethod
    def _normalize_bboxes(bboxes, batch):
        if hasattr(bboxes, "tolist"):
            bboxes = bboxes.tolist()
        if not isinstance(bboxes, (list, tuple)):
            raise ValueError("bboxes must contain XYWH boundary boxes")

        if len(bboxes) == 4 and all(not isinstance(value, (list, tuple)) for value in bboxes):
            boxes = [bboxes]
        else:
            boxes = list(bboxes)

        normalized = []
        for index, box in enumerate(boxes):
            if hasattr(box, "tolist"):
                box = box.tolist()
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"Boundary box {index} must be an XYWH tuple of four numbers")
            try:
                x, y, width, height = (float(value) for value in box)
            except (TypeError, ValueError):
                raise ValueError(f"Boundary box {index} must contain numeric XYWH values") from None
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"Boundary box {index} contains a non-finite value")
            if width < 0 or height < 0:
                raise ValueError(f"Boundary box {index} width and height must be non-negative")
            normalized.append((x, y, width, height))

        if len(normalized) == 1 and batch > 1:
            normalized *= batch
        elif len(normalized) != batch:
            raise ValueError(f"BBOX batch must contain 1 or {batch} boxes, got {len(normalized)}")
        return normalized

    @classmethod
    def _boxes_to_mask(cls, bboxes, batch, height, width, device):
        boxes = cls._normalize_bboxes(bboxes, batch)
        result = torch.zeros((batch, height, width), dtype=torch.float32, device=device)
        for index, (x, y, box_width, box_height) in enumerate(boxes):
            left = max(0, min(width, math.floor(x)))
            top = max(0, min(height, math.floor(y)))
            right = max(0, min(width, math.ceil(x + box_width)))
            bottom = max(0, min(height, math.ceil(y + box_height)))
            if right > left and bottom > top:
                result[index, top:bottom, left:right] = 1.0
        return result

    def create_mask(self, image, seam_width=6, mask=None, bboxes=None):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with shape [B,H,W,C], got {tuple(image.shape)}")

        image_batch, image_h, image_w, _ = image.shape
        if image_batch < 1 or image_h < 1 or image_w < 1:
            raise ValueError("Image batch and spatial dimensions must be non-empty")
        if mask is None and bboxes is None:
            raise ValueError("Connect a MASK or BBOX input")

        seam_width = int(seam_width)
        if seam_width < 0 or seam_width > 1024:
            raise ValueError("seam_width must be between 0 and 1024")

        device = mask.device if mask is not None else image.device
        combined = None
        if mask is not None:
            if mask.ndim != 3:
                raise ValueError(f"Expected MASK with shape [B,H,W], got {tuple(mask.shape)}")
            mask_batch, mask_h, mask_w = mask.shape
            if (image_h, image_w) != (mask_h, mask_w):
                raise ValueError(
                    f"Image and mask spatial sizes differ: {(image_h, image_w)} vs {(mask_h, mask_w)}"
                )
            if mask_batch not in (1, image_batch):
                raise ValueError(f"Mask batch must be 1 or match image batch ({image_batch}), got {mask_batch}")
            if mask_batch == 1 and image_batch > 1:
                mask = mask.expand(image_batch, -1, -1)
            combined = (mask.to(device=device, dtype=torch.float32) > 0.5).to(dtype=torch.float32)

        if bboxes is not None:
            bbox_mask = self._boxes_to_mask(bboxes, image_batch, image_h, image_w, device)
            combined = bbox_mask if combined is None else torch.maximum(combined, bbox_mask)

        if seam_width == 0:
            return (torch.zeros_like(combined),)

        binary = combined.unsqueeze(1)
        outward_radius = (seam_width + 1) // 2
        inward_radius = seam_width // 2
        expanded = self._dilate(binary, outward_radius)
        contracted = self._erode(binary, inward_radius)
        seam = (expanded - contracted).clamp_(0.0, 1.0)
        return (seam[:, 0],)


NODE_CLASS_MAPPINGS = {"MYMaskSeam": RemoveMaskSeam}
NODE_DISPLAY_NAME_MAPPINGS = {"MYMaskSeam": "移除遮罩接缝"}

