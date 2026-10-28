"""Route complete mask components by maximum local thickness, in source pixels."""

import math

import cv2
import numpy as np
import torch
from comfy_api.latest import io as IO


def large_components(alpha, threshold, tile_size=512):
    support = (alpha > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    large = np.zeros(count, dtype=bool)
    pending = (np.minimum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) > threshold)
    pending[0] = False
    if not pending.any():
        return large[labels]

    core = alpha >= 0.5
    has_core = np.bincount(labels[core], minlength=count) > 0
    # A faint, independent mask still needs geometry; do not discard it.
    geometry = (core | ((support > 0) & ~has_core[labels])).astype(np.uint8)
    height, width = alpha.shape
    halo = math.ceil(threshold / 2) + 1
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            tile_labels = labels[top:bottom, left:right]
            if not pending[tile_labels].any():
                continue
            y0, y1 = max(0, top - halo), min(height, bottom + halo)
            x0, x1 = max(0, left - halo), min(width, right + halo)
            region = geometry[y0:y1, x0:x1]
            doubled = np.repeat(np.repeat(region, 2, axis=0), 2, axis=1)
            doubled = np.pad(doubled, 1)
            distance = cv2.distanceTransform(doubled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            y, x = 1 + 2 * (top - y0), 1 + 2 * (left - x0)
            distance = distance[y:y + 2 * (bottom - top), x:x + 2 * (right - left)]
            # At 2x, distance already equals diameter in original pixels.
            # This measures both 3px and 4px straight strips correctly.
            thick = distance.reshape(bottom - top, 2, right - left, 2).max(axis=(1, 3)) > threshold
            found = np.unique(tile_labels[thick])
            found = found[found != 0]
            large[found] = True
            pending[found] = False
        if not pending.any():
            break
    return large[labels]


class MaskWidthRouter(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MYMaskWidthRouter",
            display_name="遮罩分流（宽度）",
            category="MY/遮罩处理",
            description="按每个独立区域最粗处的局部宽度整块分流。大于阈值输出大遮罩，等于或小于输出小遮罩；细长电线按粗细判断。保留原灰度、软边和批次位置。",
            search_aliases=["遮罩分流", "大遮罩", "小遮罩", "mask width", "lama"],
            is_input_list=True,
            inputs=[
                IO.Mask.Input("mask", display_name="遮罩", tooltip="单张遮罩、MASK批次或同尺寸遮罩列表；每张内的8连通区域分别判断。"),
                IO.Int.Input("pixel_threshold", display_name="像素阈值", default=32, min=0, max=16384, step=1,
                             tooltip="原遮罩分辨率下的局部宽度阈值，不是面积。最粗处超过此值，整个区域进入大遮罩；等于阈值进入小遮罩。"),
            ],
            outputs=[
                IO.Mask.Output("large_mask", display_name="大遮罩"),
                IO.Mask.Output("small_mask", display_name="小遮罩"),
            ],
        )

    @classmethod
    def execute(cls, mask, pixel_threshold):
        large_batches, small_batches = [], []
        if len(pixel_threshold) not in (1, len(mask)):
            raise ValueError("像素阈值列表须为一个值，或与遮罩列表长度相同。")
        shape = None
        for index, batch in enumerate(mask):
            if batch.ndim == 2:
                batch = batch.unsqueeze(0)
            if batch.ndim != 3 or min(batch.shape) < 1:
                raise ValueError("遮罩须为非空的 [B,H,W] 或 [H,W] 张量。")
            if shape is not None and batch.shape[1:] != shape:
                raise ValueError("遮罩列表中的宽高必须相同；请先对齐到相同画布。")
            shape = batch.shape[1:]
            threshold = int(pixel_threshold[0 if len(pixel_threshold) == 1 else index])
            if threshold < 0:
                raise ValueError("像素阈值不能小于0。")
            frames = batch.detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(frames).all() or (frames < 0).any() or (frames > 1).any():
                raise ValueError("遮罩值必须是0至1之间的有限数值。")
            large_batch, small_batch = torch.zeros_like(batch), torch.zeros_like(batch)
            for frame_index, alpha in enumerate(frames):
                selected = torch.from_numpy(large_components(alpha, threshold)).to(batch.device)
                large_batch[frame_index] = torch.where(selected, batch[frame_index], 0)
                small_batch[frame_index] = torch.where(selected, 0, batch[frame_index])
            large_batches.append(large_batch)
            small_batches.append(small_batch)
        if not large_batches:
            raise ValueError("请接入至少一张遮罩。")
        return IO.NodeOutput(torch.cat(large_batches), torch.cat(small_batches))
