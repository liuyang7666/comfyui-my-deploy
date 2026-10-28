"""Global SIFT registration; estimate on previews and warp the original once."""

from time import perf_counter

import cv2
import numpy as np
import torch


class PixelDriftFixNode:
    DETECTION_MAX_SIDE = 1536

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_image": ("IMAGE", {"tooltip": "作为位置、透视和输出尺寸基准的原图。"}),
            "edited_image": ("IMAGE", {"tooltip": "需要对齐的编辑后图像；两图须保留足够相同的画面特征。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("fixed_image",)
    FUNCTION = "fix_pixel_drift"
    CATEGORY = "MY/图像对齐"
    DESCRIPTION = (
        "按画面特征自动校正整体位移、旋转、缩放与透视。大图在缩小副本上匹配，"
        "按原图尺寸输出。匹配失败会原样返回编辑图并警告；请检查结果。内部按8位处理。"
    )

    @staticmethod
    def _detection_image(rgb, max_side):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        scale = min(1.0, max_side / max(height, width))
        if scale == 1.0:
            return gray, np.eye(3, dtype=np.float64)
        small_w = max(1, round(width * scale))
        small_h = max(1, round(height * scale))
        gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # OpenCV resize samples pixel centres. Include both the rounded x/y
        # scales and the half-pixel offsets when returning to native coordinates.
        sx, sy = small_w / width, small_h / height
        native_to_small = np.array([
            [sx, 0.0, (sx - 1.0) / 2.0],
            [0.0, sy, (sy - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ])
        return gray, native_to_small

    def fix_pixel_drift(self, source_image, edited_image, method=None, max_mesh_points=None):
        # Accept obsolete API arguments without exposing widgets or running mesh.
        outputs = []
        for index in range(min(len(source_image), len(edited_image))):
            started = perf_counter()
            original = (source_image[index].detach().cpu().numpy() * 255).astype(np.uint8)
            edited = (edited_image[index].detach().cpu().numpy() * 255).astype(np.uint8)
            height, width = original.shape[:2]
            # Preserve the original small-image path. Both large inputs get
            # independent previews, supporting different sizes/aspect ratios.
            use_preview = min(max(original.shape[:2]), max(edited.shape[:2])) > self.DETECTION_MAX_SIDE
            limit = self.DETECTION_MAX_SIDE if use_preview else max(*original.shape[:2], *edited.shape[:2])
            gray_orig, scale_orig = self._detection_image(original, limit)
            gray_edit, scale_edit = self._detection_image(edited, limit)
            sift = cv2.SIFT_create(
                nfeatures=0, nOctaveLayers=3, contrastThreshold=0.01,
                edgeThreshold=20, sigma=1.6,
            )
            kp_orig, des_orig = sift.detectAndCompute(gray_orig, None)
            kp_edit, des_edit = sift.detectAndCompute(gray_edit, None)
            detected = perf_counter()

            def fallback(reason):
                print(f"[MY 自动对齐] 批次{index}：{reason}，原样输出编辑图，未完成对齐。")
                outputs.append(edited_image[index].detach().cpu())

            if des_orig is None or des_edit is None or min(len(kp_orig), len(kp_edit)) < 10:
                fallback("特征点不足")
                continue
            matches = cv2.BFMatcher().knnMatch(des_edit, des_orig, k=2)
            good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.80 * pair[1].distance]
            matched = perf_counter()
            if len(good) < 10:
                fallback(f"有效匹配不足（{len(good)}）")
                continue
            pts_edit = np.float32([kp_edit[m.queryIdx].pt for m in good])
            pts_orig = np.float32([kp_orig[m.trainIdx].pt for m in good])
            matrix, inliers = cv2.findHomography(pts_edit, pts_orig, cv2.RANSAC, 5.0)
            if matrix is None or inliers is None or np.count_nonzero(inliers) < 10:
                fallback("相同特征不足或透视矩阵计算失败")
                continue
            matrix = np.linalg.inv(scale_orig) @ matrix @ scale_edit
            final = cv2.warpPerspective(edited, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            outputs.append(torch.from_numpy(final).float().div_(255.0))
            print(
                f"[MY 自动对齐] 批次{index}：检测{detected-started:.2f}s，"
                f"匹配{matched-detected:.2f}s，总计{perf_counter()-started:.2f}s；"
                f"特征{len(kp_orig)}/{len(kp_edit)}，内点{np.count_nonzero(inliers)}；"
                f"检测尺寸{gray_orig.shape[1]}×{gray_orig.shape[0]}/"
                f"{gray_edit.shape[1]}×{gray_edit.shape[0]}，输出{width}×{height}。"
            )
        return (torch.stack(outputs, dim=0),)
