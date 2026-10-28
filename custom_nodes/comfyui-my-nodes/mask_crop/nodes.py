"""MY mask crop node with the KJ crop controls and aligned output sizes."""

import math

import torch
import torch.nn.functional as F


class MaskCropByMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_size_mult": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
                "bbox_smooth_alpha": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "裁剪输出的宽和高都向上取整为此整数倍。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "INT", "INT")
    RETURN_NAMES = ("original_images", "cropped_images", "bboxes", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "MY/图像裁剪"
    DESCRIPTION = (
        "按遮罩计算裁剪框，保留原图、裁剪图、XYWH边界框和裁剪尺寸输出。"
        "裁剪尺寸会按倍数向上对齐。"
    )

    @staticmethod
    def _round_up(value, multiple):
        value = max(1, int(value))
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _center_crop_or_pad(image, target_h, target_w):
        height, width = image.shape[-2:]
        pad_left = max(0, (target_w - width) // 2)
        pad_right = max(0, target_w - width - pad_left)
        pad_top = max(0, (target_h - height) // 2)
        pad_bottom = max(0, target_h - height - pad_top)
        if pad_left or pad_right or pad_top or pad_bottom:
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

        height, width = image.shape[-2:]
        top = max(0, (height - target_h) // 2)
        left = max(0, (width - target_w) // 2)
        return image[..., top : top + target_h, left : left + target_w]

    def crop(self, masks, original_images, crop_size_mult=3.0, bbox_smooth_alpha=0.5, multiple=16):
        if original_images.ndim != 4:
            raise ValueError(
                f"Expected IMAGE with shape [B,H,W,C], got {tuple(original_images.shape)}"
            )
        if masks.ndim != 3:
            raise ValueError(f"Expected MASK with shape [B,H,W], got {tuple(masks.shape)}")

        batch, image_h, image_w, _ = original_images.shape
        mask_batch, mask_h, mask_w = masks.shape
        if batch < 1:
            raise ValueError("Cannot crop an empty image batch")
        if image_h < 1 or image_w < 1:
            raise ValueError("Cannot crop an empty image")
        if (mask_h, mask_w) != (image_h, image_w):
            raise ValueError(
                f"Image and mask spatial sizes differ: {(image_h, image_w)} vs {(mask_h, mask_w)}"
            )
        if mask_batch not in (1, batch):
            raise ValueError(f"Mask batch must be 1 or match image batch ({batch}), got {mask_batch}")

        multiple = int(multiple)
        crop_size_mult = float(crop_size_mult)
        bbox_smooth_alpha = float(bbox_smooth_alpha)
        if multiple < 1:
            raise ValueError("multiple must be a positive integer")
        if not math.isfinite(crop_size_mult) or not 0.0 <= crop_size_mult <= 10.0:
            raise ValueError("crop_size_mult must be between 0 and 10")
        if not math.isfinite(bbox_smooth_alpha) or not 0.0 <= bbox_smooth_alpha <= 1.0:
            raise ValueError("bbox_smooth_alpha must be between 0 and 1")

        smooth_centers = []
        smooth_sizes = []
        previous_center = None
        previous_size = None

        for index in range(batch):
            current_mask = masks[0 if mask_batch == 1 else index] > 0
            ys, xs = torch.where(current_mask)
            if ys.numel() == 0:
                raise ValueError(f"Mask {index} is empty; cannot determine a crop")

            x0 = int(xs.min().item())
            x1 = int(xs.max().item()) + 1
            y0 = int(ys.min().item())
            y1 = int(ys.max().item()) + 1
            center = (round(float(xs.float().mean().item())), round(float(ys.float().mean().item())))
            bbox_w, bbox_h = x1 - x0, y1 - y0
            requested_size = (
                max(1, round(bbox_w * crop_size_mult)),
                max(1, round(bbox_h * crop_size_mult)),
            )

            if previous_center is None:
                size = requested_size
            else:
                alpha = bbox_smooth_alpha
                center = (
                    round(alpha * center[0] + (1.0 - alpha) * previous_center[0]),
                    round(alpha * center[1] + (1.0 - alpha) * previous_center[1]),
                )
                size = (
                    max(1, round(alpha * requested_size[0] + (1.0 - alpha) * previous_size[0])),
                    max(1, round(alpha * requested_size[1] + (1.0 - alpha) * previous_size[1])),
                )

            smooth_centers.append(center)
            smooth_sizes.append(size)
            previous_center = center
            previous_size = size

        target_w = self._round_up(max(size[0] for size in smooth_sizes), multiple)
        target_h = self._round_up(max(size[1] for size in smooth_sizes), multiple)
        aspect_ratio = target_w / target_h

        cropped_images = []
        bounding_boxes = []
        for index, center in enumerate(smooth_centers):
            crop_x0 = center[0] - target_w // 2
            crop_y0 = center[1] - target_h // 2
            crop_x1 = crop_x0 + target_w
            crop_y1 = crop_y0 + target_h

            x0 = max(0, crop_x0)
            y0 = max(0, crop_y0)
            x1 = min(image_w, crop_x1)
            y1 = min(image_h, crop_y1)
            bounding_boxes.append((x0, y0, x1 - x0, y1 - y0))

            crop = original_images[index : index + 1, y0:y1, x0:x1, :].movedim(-1, 1)
            resize_h = min(crop.shape[-2], target_h)
            resize_w = max(1, round(resize_h * aspect_ratio))
            resized = F.interpolate(
                crop,
                size=(resize_h, resize_w),
                mode="bilinear",
                align_corners=False,
            )
            resized = self._center_crop_or_pad(resized, target_h, target_w)
            cropped_images.append(resized.movedim(1, -1))

        return (
            original_images,
            torch.cat(cropped_images, dim=0),
            bounding_boxes,
            target_w,
            target_h,
        )
