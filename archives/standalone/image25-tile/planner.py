import math

import torch
import torch.nn.functional as F


MAX_PIXELS = 8294400
MIN_PIXELS = 655360


def align16(value):
    return math.ceil(value / 16) * 16


def axis_size(length, count, overlap):
    core = math.ceil(length / count)
    margin = math.ceil(core * overlap) if count > 1 else 0
    return align16(core + margin), margin


def axis_overlap(length, count, tile):
    if count == 1:
        return 0
    # margin/core <= 4%; spend alignment slack on overlap before padding.
    return min(tile - math.ceil(length / count), tile // 26)


def tile_size(width, height, cols, rows, overlap):
    tile_w, _ = axis_size(width, cols, overlap)
    tile_h, _ = axis_size(height, rows, overlap)
    if tile_w > 3 * tile_h:
        tile_h = align16(tile_w / 3)
    elif tile_h > 3 * tile_w:
        tile_w = align16(tile_h / 3)
    while tile_w * tile_h < MIN_PIXELS:
        if tile_w <= tile_h:
            tile_w += 16
        else:
            tile_h += 16
    return tile_w, tile_h, axis_overlap(width, cols, tile_w), axis_overlap(height, rows, tile_h)


def plan_tiles(width, height, max_long_edge=3840, overlap=0.03):
    if min(width, height) < 1:
        raise ValueError("图像宽高必须大于 0。")
    if not 816 <= max_long_edge <= 3840 or not math.isfinite(overlap) or not 0 <= overlap <= 0.5:
        raise ValueError("长边上限应在 816–3840，重叠比例应在 0–0.5。")
    limit = int(max_long_edge) // 16 * 16
    preferred = min(0.04, max(0.02, overlap))
    min_cols = max(1, math.ceil(width / limit))
    min_rows = max(1, math.ceil(height / limit))
    while min_cols <= 256 and axis_size(width, min_cols, 0.02)[0] > limit:
        min_cols += 1
    while min_rows <= 256 and axis_size(height, min_rows, 0.02)[0] > limit:
        min_rows += 1
    best, best_score = None, (256 * 256 + 1,)
    for cols in range(min_cols, 257):
        if cols * min_rows > best_score[0]:
            break
        first_row = max(min_rows, math.ceil(width * height / (MAX_PIXELS * cols)))
        for rows in range(first_row, min(256, best_score[0] // cols) + 1):
            feasible = False
            for ratio in sorted({0.02, preferred, 0.04}):
                tile_w, tile_h, overlap_x, overlap_y = tile_size(width, height, cols, rows, ratio)
                if max(tile_w, tile_h) > limit or tile_w * tile_h > MAX_PIXELS:
                    continue
                feasible = True
                core_w, core_h = tile_w - overlap_x, tile_h - overlap_y
                padding = cols * core_w * rows * core_h - width * height
                deviation = (abs(overlap_x / core_w - preferred) if cols > 1 else 0) + (abs(overlap_y / core_h - preferred) if rows > 1 else 0)
                score = (cols * rows, padding, deviation, abs(cols - rows))
                if score >= best_score:
                    continue
                best_score = score
                best = {
                    "cols": cols, "rows": rows, "total": cols * rows,
                    "pad_right": cols * core_w - width, "pad_bottom": rows * core_h - height,
                    "overlap_x": overlap_x, "overlap_y": overlap_y,
                    "tile_width": tile_w, "tile_height": tile_h,
                }
            if feasible:
                break
    if best is not None:
        return best
    raise ValueError("分块数量超过现有 ImageTile+ 每轴 256 块的限制。")


def split_tiles(image, plan):
    if plan["pad_right"] or plan["pad_bottom"]:
        image = F.pad(image.movedim(-1, 1), (0, plan["pad_right"], 0, plan["pad_bottom"]), mode="replicate").movedim(1, -1)
    core_w = plan["tile_width"] - plan["overlap_x"]
    core_h = plan["tile_height"] - plan["overlap_y"]
    tiles = []
    for row in range(plan["rows"]):
        y = row * core_h - (plan["overlap_y"] if row > 0 else 0)
        for col in range(plan["cols"]):
            x = col * core_w - (plan["overlap_x"] if col > 0 else 0)
            tiles.append(image[:, y:y + plan["tile_height"], x:x + plan["tile_width"], :])
    return torch.cat(tiles, dim=0)


class Image25TilePlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "max_long_edge": ("INT", {"default": 3840, "min": 816, "max": 3840, "step": 16}),
            "overlap": ("FLOAT", {"default": 0.03, "min": 0, "max": 0.5, "step": 0.01,
                "tooltip": "自动在2%–4%内选择整数重叠；此值为偏好，旧工作流的5%自动适配。"}),
        }}

    RETURN_TYPES = ("INT",) * 9 + ("IMAGE",)
    RETURN_NAMES = ("列数", "行数", "数量", "右补齐", "下补齐", "重叠X", "重叠Y", "块宽", "块高", "分块图像")
    FUNCTION = "execute"
    CATEGORY = "image/tiling"
    DESCRIPTION = "总块数最少优先，自动2%–4%重叠；每块宽高为16倍数、最长边≤3840、面积655360–8294400、比例≤3:1。分块图像为按行排列的IMAGE批次，已含重叠和边缘补齐，不缩放。合成后裁回输入原尺寸。"

    def execute(self, image, max_long_edge=3840, overlap=0.03):
        height, width = image.shape[1:3]
        plan = plan_tiles(width, height, max_long_edge, overlap)
        values = tuple(plan[key] for key in ("cols", "rows", "total", "pad_right", "pad_bottom", "overlap_x", "overlap_y", "tile_width", "tile_height"))
        return values + (split_tiles(image, plan),)
