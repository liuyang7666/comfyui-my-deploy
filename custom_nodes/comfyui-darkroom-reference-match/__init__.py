import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .core import render_numpy, solve
from .darkroom import load_color_core, load_gpu_core
from .features import analyze_images, local_model_signature
from .image_io import read_srgb, save16
from .render import render

CATEGORY = 'AKURATE/Darkroom/自动参考匹配'


class DarkroomSemanticReferenceMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'reference': ('IMAGE',),
                'strength': ('INT', {'default': 100, 'min': 0, 'max': 100, 'step': 1,
                                     'display': 'slider', 'tooltip': '0保留原图，100完整颜色调整；在对应蒙版内选颜色范围，保留原始亮度，不迁移参考曝光。'})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('图像',)
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'v0.5.0。在肤色、天空、植物等对应蒙版内选择连续颜色范围；仅改颜色，保留逐像素原始亮度。检测、蒙版、统计及颜色方案缓存复用。'

    def run(self, image, reference, strength=100):
        import folder_paths
        started = time.monotonic()
        if image.ndim != 4 or reference.ndim != 4 or image.shape[-1] != 3 or reference.shape[-1] != 3:
            raise ValueError('需要B,H,W,3的已显色sRGB图像。')
        if reference.shape[0] not in (1, image.shape[0]):
            raise ValueError('参考批次应为1或与目标批次一致。')
        if not np.isfinite(strength) or not 0 <= strength <= 100:
            raise ValueError('强度必须在0–100之间。')
        if strength == 0:
            self.last_run = {'version': '0.5.0', 'strength': 0, 'identity': True,
                             'image_encoder_calls': 0, 'statistics_rebuilt': 0, 'seconds': 0.}
            return (image,)
        sources = [x.detach().cpu().float().numpy() for x in image]
        refs = [x.detach().cpu().float().numpy() for x in reference]
        for array in sources + refs:
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError('需要有限的0–1 sRGB；RAW/HDR请先显式转换。')
        pairs = {i: 0 if len(refs) == 1 else i for i in range(len(sources))}
        active = [i for i, j in pairs.items() if not np.array_equal(sources[i], refs[j])]
        if not active:
            self.last_run = {'version': '0.5.0', 'strength': strength, 'identity': True, 'image_encoder_calls': 0,
                             'seconds': round(time.monotonic() - started, 3)}
            return (image,)
        color, dependency = load_color_core()
        gpu = load_gpu_core(color)
        used_refs = sorted({pairs[i] for i in active})
        needed = [sources[i] for i in active] + [refs[j] for j in used_refs]
        directory = Path(folder_paths.get_user_directory()) / 'default' / 'darkroom_reference_cache'
        profiles, cache = analyze_images(needed, color, directory, local_model_signature(), dependency['sha256'])
        source_profiles = dict(zip(active, profiles[:len(active)]))
        reference_profiles = dict(zip(used_refs, profiles[len(active):]))
        device = gpu.device()
        outputs, render_seconds, plans, plan_cache = [], 0., [], {}
        for i, source in enumerate(sources):
            if i not in source_profiles:
                outputs.append(torch.from_numpy(source))
                continue
            profile = source_profiles[i]
            reference_profile = reference_profiles[pairs[i]]
            key = (id(profile), id(reference_profile), strength)
            if key not in plan_cache:
                plan_cache[key] = solve(profile, reference_profile, strength)
            plan = plan_cache[key]
            render_started = time.monotonic()
            if device.type == 'cpu':
                result = render_numpy(source, plan, profile['application'], color)
            else:
                try:
                    result = render(source, plan, profile['application'], gpu, device)
                except torch.cuda.OutOfMemoryError:
                    logging.warning('[Darkroom reference] GPU memory exhausted; rendering on CPU.')
                    result = render_numpy(source, plan, profile['application'], color)
            render_seconds += time.monotonic() - render_started
            outputs.append(torch.from_numpy(result))
            plans.append({'batch': i, 'regional_sample_counts': plan['support'], 'order': plan['order'],
                          'regions': plan['regions'], 'brightness_policy': plan['brightness_policy'],
                          'application_masks': [n for n in plan['ranges'] if np.any(profile['application'][n])],
                          'solver_cache_hit': plan['plan_cache_hit']})
        self.last_run = {'version': '0.5.0', 'strength': strength, **cache, 'plan_solves': len(plan_cache), 'render_seconds': round(render_seconds, 3),
                         'seconds': round(time.monotonic() - started, 3), 'device': str(device), 'images': plans}
        logging.info('[Darkroom reference] %s', json.dumps(self.last_run, ensure_ascii=True))
        return (torch.stack(outputs).to(device=image.device, dtype=image.dtype),)


def input_path(image):
    import folder_paths
    path = Path(folder_paths.get_annotated_filepath(image)).resolve()
    allowed = [Path(folder_paths.get_input_directory()).resolve(), Path(folder_paths.get_output_directory()).resolve()]
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError('只允许读取ComfyUI input/output内的图像。')
    return path


class DarkroomPrecisionLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root = Path(folder_paths.get_input_directory())
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                       if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'))
        return {'required': {'image': (files, {'image_upload': True})}}

    RETURN_TYPES = ('IMAGE', 'DR_IMAGE_META')
    RETURN_NAMES = ('高精度图像', '原图元数据')
    FUNCTION = 'run'
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, image):
        return hashlib.sha256(input_path(image).read_bytes()).hexdigest()

    def run(self, image):
        rgb, meta = read_srgb(input_path(image))
        return torch.from_numpy(rgb).unsqueeze(0), {'images': [meta]}


class DarkroomSaveImage16:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',),
                'filename_prefix': ('STRING', {'default': 'Darkroom参考匹配/result'}),
                'format': (['PNG16', 'TIFF16'],)}, 'optional': {'metadata': ('DR_IMAGE_META',)}}

    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, images, filename_prefix='Darkroom参考匹配/result', format='PNG16', metadata=None):
        import folder_paths
        root = Path(folder_paths.get_output_directory()).resolve()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(root), images.shape[2], images.shape[1])
        folder = Path(folder).resolve()
        if not folder.is_relative_to(root):
            raise ValueError('保存路径必须在ComfyUI output内。')
        if format not in ('PNG16', 'TIFF16'):
            raise ValueError('保存格式不受支持。')
        entries = metadata['images'] if metadata else [{} for item in images]
        if len(entries) not in (1, len(images)):
            raise ValueError('保存元数据批次不匹配。')
        saved = []
        suffix = '.png' if format == 'PNG16' else '.tif'
        for i, item in enumerate(images):
            leaf = filename.replace('%batch_num%', str(i))
            path = folder / (leaf + '_' + str(counter).zfill(5) + '_' + suffix)
            while path.exists():
                counter += 1
                path = folder / (leaf + '_' + str(counter).zfill(5) + '_' + suffix)
            if not path.resolve().is_relative_to(root):
                raise ValueError('保存文件必须在ComfyUI output内。')
            save16(path, item.detach().cpu().float().numpy(), entries[min(i, len(entries) - 1)])
            saved.append({'filename': path.name, 'subfolder': subfolder, 'type': 'output'})
            counter += 1
        return {'ui': {'images': saved}}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (DarkroomSemanticReferenceMatch,
                      DarkroomPrecisionLoadImage, DarkroomSaveImage16)}
NODE_DISPLAY_NAME_MAPPINGS = {'DarkroomSemanticReferenceMatch': 'Darkroom 自动风格匹配',
    'DarkroomPrecisionLoadImage': 'Darkroom 高精度读图', 'DarkroomSaveImage16': 'Darkroom 无损16位保存'}
