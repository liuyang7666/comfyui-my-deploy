"""Precision-preserving sRGB decoding and lossless 16-bit output."""
import base64
import io
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms


def srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def orient(array, orientation):
    if orientation == 2:
        return array[:, ::-1]
    if orientation == 3:
        return array[::-1, ::-1]
    if orientation == 4:
        return array[::-1]
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return np.swapaxes(array, 0, 1)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(array)
    return array


def read_srgb(path):
    path = Path(path)
    with Image.open(path) as pil:
        mode = pil.mode
        if mode in ('CMYK', 'LAB'):
            raise ValueError('请先把 CMYK/Lab 原件以高精度转换为 sRGB，再进行参考匹配。')
        icc = pil.info.get('icc_profile')
        if icc:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            description = ImageCms.getProfileDescription(profile).strip()
            if 'srgb' not in description.lower().replace(' ', ''):
                raise ValueError('当前入口只处理 sRGB，检测到 ICC：' + description + '。请先高精度转 sRGB，避免误读或降位深。')
        else:
            description = 'untagged_assumed_sRGB'
        exif = pil.getexif()
        orientation = int(exif.get(274, 1))
        exif[274] = 1
        exif_bytes = exif.tobytes() if exif else b''
        # Palette images need their palette, not cv2's grayscale index values.
        if mode == 'P':
            rgb = np.array(pil.convert('RGB'), dtype=np.float32) / 255
            depth = 8
        else:
            data = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if data is None or data.dtype not in (np.uint8, np.uint16):
                raise ValueError('仅支持已显色的 8/16 位 RGB PNG/TIFF/JPEG/WebP；不隐式显色 RAW/HDR。')
            if data.ndim == 2:
                data = np.repeat(data[..., None], 3, axis=-1)
            if data.shape[-1] == 4 and np.any(data[..., 3] != np.iinfo(data.dtype).max):
                raise ValueError('当前调色入口不丢弃透明度；请先提供已合成的 RGB 摄影图。')
            if data.shape[-1] not in (3, 4):
                raise ValueError('图像通道不受支持。')
            depth = 16 if data.dtype == np.uint16 else 8
            rgb = data[..., :3][..., ::-1].astype(np.float32) / np.iinfo(data.dtype).max
    rgb = np.ascontiguousarray(orient(rgb, orientation), dtype=np.float32)
    return rgb, {'source_file': str(path), 'source_bit_depth': depth,
                 'shape': list(rgb.shape), 'working_space': 'sRGB',
                 'source_profile_description': description,
                 'icc_base64': base64.b64encode(icc or srgb_profile()).decode(),
                 'exif_base64': base64.b64encode(exif_bytes).decode()}


def png_chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def encode_png16(rgb, metadata=None):
    quantized = np.rint(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
    ok, encoded = cv2.imencode('.png', quantized[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not ok:
        raise RuntimeError('16 位 PNG 编码失败。')
    metadata = metadata or {}
    icc = base64.b64decode(metadata['icc_base64']) if metadata.get('icc_base64') else srgb_profile()
    exif = base64.b64decode(metadata.get('exif_base64', ''))
    # OpenCV has no ICC embedding API; insert standard PNG chunks after IHDR.
    chunks = png_chunk(b'iCCP', b'sRGB\0\0' + zlib.compress(icc))
    if exif:
        chunks += png_chunk(b'eXIf', exif[6:] if exif.startswith(b'Exif\0\0') else exif)
    raw = encoded.tobytes()
    return raw[:33] + chunks + raw[33:]


def save16(path, rgb, metadata=None, report=None):
    path = Path(path)
    metadata = metadata or {}
    if metadata.get('shape') and metadata['shape'] != list(rgb.shape):
        raise ValueError('保存元数据的原图尺寸与结果不同，请连接目标图的元数据。')
    if path.suffix.lower() == '.png':
        with path.open('xb') as stream:
            stream.write(encode_png16(rgb, metadata))
    else:
        import tifffile
        icc = base64.b64decode(metadata['icc_base64']) if metadata.get('icc_base64') else srgb_profile()
        with path.open('xb') as stream:
            tifffile.imwrite(stream, np.rint(np.clip(rgb, 0, 1) * 65535).astype(np.uint16),
                              photometric='rgb', compression='deflate', metadata=None,
                              extratags=[(34675, 'B', len(icc), icc, False)])
    sidecar = path.with_suffix(path.suffix + '.json')
    sidecar.write_text(json.dumps({'source_metadata': metadata, 'processing_report': report,
                                  'output_bit_depth': 16, 'lossless': True}, ensure_ascii=False, indent=2), encoding='utf-8')
