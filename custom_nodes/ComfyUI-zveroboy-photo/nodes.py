import io
import os
import random
import base64
import hashlib
import json
from pathlib import Path
import re
import torch
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance, ImageOps
from PIL.TiffImagePlugin import ImageFileDirectory_v2, IFDRational
from PIL import TiffTags
import piexif
import folder_paths

# --- Поддержка HEIC (iPhone) ---
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

# --- Поддержка RAW (Камеры) ---
try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

# --- Импорт пресетов ---
from .presets import CAMERA_PRESETS
from .photo_metadata import prepare_metadata

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _rational_from_str(s: str):
    try:
        s = str(s).strip()
        if "/" in s:
            n, d = s.split("/", 1)
            return (int(n), int(d))
        v = float(s)
        if v.is_integer():
            return (int(v * 10), 10)
        return (int(v * 100), 100)
    except Exception:
        return (0, 1)

def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    if img_t.is_cuda:
        img_t = img_t.detach().cpu()
    if img_t.ndim == 4:
        if img_t.shape[0] == 1:
            img_t = img_t[0]
        else:
            raise ValueError("tensor_to_pil expects single image (HWC) or batch size 1.")
    img_t = torch.clamp(img_t, 0.0, 1.0)
    arr = (img_t.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)

def encode_exif_to_string(exif_bytes: bytes) -> str:
    return base64.b64encode(exif_bytes).decode("utf-8")

def decode_exif_from_string(exif_str: str) -> bytes:
    if not exif_str:
        return b""
    try:
        return base64.b64decode(exif_str)
    except Exception:
        return b""

def build_exif_bytes(
    preset_name: str,
    artist: str,
    software: str,
    copyright_text: str,
    body_serial: str,
    lens_serial: str,
    focal_length_mm: str,
    fnumber: str,
    exposure_1_over_s: str,
    iso: int,
    exposure_bias_ev: str,
    white_balance: int,
    datetime_original: str,
    lens_model_override: str = "",
    make_override: str = "",
    model_override: str = "",
):
    preset = CAMERA_PRESETS.get(preset_name, CAMERA_PRESETS["Canon"])
    make = make_override or preset.get("Make", "")
    model = model_override or preset.get("Model", "")
    lens_model = lens_model_override or preset.get("LensModel", "")

    zeroth = {
        piexif.ImageIFD.Make: make.encode("utf-8"),
        piexif.ImageIFD.Model: model.encode("utf-8"),
        piexif.ImageIFD.Software: (software or preset.get("Software", "")).encode("utf-8"),
        piexif.ImageIFD.Artist: (artist or preset.get("Artist", "")).encode("utf-8"),
        piexif.ImageIFD.Copyright: (copyright_text or "").encode("utf-8"),
        piexif.ImageIFD.XResolution: (300, 1),
        piexif.ImageIFD.YResolution: (300, 1),
        piexif.ImageIFD.ResolutionUnit: 2,
    }

    dt_clean = datetime_original.strip()
    if not dt_clean:
        dt_clean = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    else:
        try:
            dt_obj = datetime.strptime(dt_clean, "%Y-%m-%d %H:%M:%S")
            dt_clean = dt_obj.strftime("%Y:%m:%d %H:%M:%S")
        except:
            pass

    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: dt_clean.encode("ascii", errors="ignore"),
        piexif.ExifIFD.ExposureTime: _rational_from_str("1/" + str(exposure_1_over_s)),
        piexif.ExifIFD.FNumber: _rational_from_str(str(fnumber)),
        piexif.ExifIFD.ISOSpeedRatings: int(iso),
        piexif.ExifIFD.FocalLength: _rational_from_str(str(focal_length_mm)),
        piexif.ExifIFD.LensModel: (lens_model or "").encode("utf-8"),
        piexif.ExifIFD.BodySerialNumber: (body_serial or "").encode("utf-8"),
        piexif.ExifIFD.LensSerialNumber: (lens_serial or "").encode("utf-8"),
        piexif.ExifIFD.ExposureBiasValue: _rational_from_str(str(exposure_bias_ev)),
        piexif.ExifIFD.WhiteBalance: int(white_balance),
    }

    exif_dict = {"0th": zeroth, "Exif": exif_ifd, "GPS": {}, "1st": {}, "Interop": {}}
    return piexif.dump(exif_dict)

# =============================================================================
# NODE: Photo Add Noise
# =============================================================================

class PhotoAddNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "noise_level": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "zveroboy/photo"

    def apply(self, images, noise_level):
        out_list = []
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i])
            arr = np.array(pil, dtype=np.float32)
            noise = np.random.normal(0, 255 * noise_level, arr.shape)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                noise[:, :, 2] *= 0.8
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            pil_noisy = Image.fromarray(arr, mode="RGB")
            out_list.append(pil_to_tensor(pil_noisy))
        return (torch.stack(out_list, dim=0),)

# =============================================================================
# NODE: Photo Add Grain
# =============================================================================

class PhotoAddGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grain_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "zveroboy/photo"

    def apply(self, images, grain_strength, grain_size):
        out_list = []
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i]).convert("L")
            w, h = pil.size
            grain_arr = np.random.normal(128, 50 * grain_strength, (h, w)).astype(np.float32)
            grain_img = Image.fromarray(np.clip(grain_arr, 0, 255).astype(np.uint8), mode="L")
            if grain_size > 1:
                gw, gh = w // grain_size, h // grain_size
                grain_small = grain_img.resize((gw, gh), resample=Image.NEAREST)
                grain_img = grain_small.resize((w, h), resample=Image.NEAREST)
            original_rgb = tensor_to_pil(images[i]).convert("RGB")
            orig_arr = np.array(original_rgb, dtype=np.float32)
            grain_arr = np.array(grain_img, dtype=np.float32)
            grain_norm = (grain_arr - 128) / 128.0
            for c in range(3):
                channel = orig_arr[:, :, c] / 255.0
                mixed = np.where(channel < 0.5,
                                 2 * channel * (0.5 + 0.5 * grain_norm),
                                 1 - 2 * (1 - channel) * (0.5 - 0.5 * grain_norm))
                orig_arr[:, :, c] = np.clip(mixed * 255, 0, 255)
            final_pil = Image.fromarray(orig_arr.astype(np.uint8), mode="RGB")
            out_list.append(pil_to_tensor(final_pil))
        return (torch.stack(out_list, dim=0),)

# =============================================================================
# NODE: Photo Add EXIF
# =============================================================================

class PhotoAddExif:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (list(CAMERA_PRESETS.keys()), {"default": "Canon"}),
                "artist": ("STRING", {"default": "VM"}),
                "software": ("STRING", {"default": ""}),
                "copyright": ("STRING", {"default": ""}),
                "body_serial": ("STRING", {"default": ""}),
                "lens_serial": ("STRING", {"default": ""}),
                "focal_length_mm": ("STRING", {"default": "50"}),
                "fnumber": ("STRING", {"default": "4.0"}),
                "exposure_1_over_s": ("STRING", {"default": "125"}),
                "iso": ("INT", {"default": 400, "min": 50, "max": 204800}),
                "exposure_bias_ev": ("STRING", {"default": "0"}),
                "white_balance": ("INT", {"default": 0, "min": 0, "max": 1}),
                "datetime_original": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "EXIF_DATA")
    FUNCTION = "apply"
    CATEGORY = "zveroboy/photo"

    def apply(self, images, preset, artist, software, copyright, body_serial, lens_serial,
              focal_length_mm, fnumber, exposure_1_over_s, iso, exposure_bias_ev,
              white_balance, datetime_original):
        if not body_serial.strip():
            body_serial = str(random.randint(1000000, 99999999))
        if not lens_serial.strip():
            lens_serial = str(random.randint(100000000, 999999999))
        exif_bytes = build_exif_bytes(
            preset_name=preset,
            artist=artist,
            software=software,
            copyright_text=copyright,
            body_serial=body_serial,
            lens_serial=lens_serial,
            focal_length_mm=focal_length_mm,
            fnumber=fnumber,
            exposure_1_over_s=exposure_1_over_s,
            iso=iso,
            exposure_bias_ev=exposure_bias_ev,
            white_balance=white_balance,
            datetime_original=datetime_original,
        )
        exif_str = encode_exif_to_string(exif_bytes)
        return (images, exif_str)

# =============================================================================
# NODE: Photo Load RAW
# =============================================================================

PHOTO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif",
    ".3fr", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dcs", ".dng", ".erf", ".fff", ".gpr",
    ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef", ".raf",
    ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}


def photo_paths(raw_file, include_subfolders=False, start_index=0, max_images=0):
    value = raw_file.strip()
    if not value:
        raise ValueError("请点击“选择照片”或“选择文件夹”，也可以填写完整路径。")
    if value.startswith("["):
        values = json.loads(value)
        if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
            raise ValueError("照片选择内容应为文件路径列表，请重新选择照片。")
    else:
        values = [value]
    paths = []
    for item in values:
        item = item.strip()
        if len(item) >= 2 and (item[0], item[-1]) in (("\"", "\""), ("'", "'"), ("“", "”")):
            item = item[1:-1].strip()
        path = Path(item)
        if not path.is_absolute():
            path = Path(folder_paths.base_path) / path
        if not path.exists():
            raise FileNotFoundError(f"找不到照片或文件夹，请检查完整路径：{path}")
        if path.is_dir():
            entries = path.rglob("*") if include_subfolders else path.iterdir()
            paths.extend(file.resolve() for file in entries if file.is_file() and file.suffix.lower() in PHOTO_EXTENSIONS)
        else:
            paths.append(path.resolve())
    paths = sorted(set(paths), key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path).casefold())])
    paths = paths[start_index:]
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise ValueError("没有可读取的照片。请检查文件夹、是否包含子文件夹以及起始序号。")
    return paths


class PhotoLoadRaw:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_file": ("STRING", {"multiline": False, "default": "", "tooltip": "点击按钮选择照片或文件夹，也可粘贴完整路径。多选文件时由按钮自动填写路径列表。"}),
            },
            "optional": {
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "0表示全部；照片较多时可配合起始序号分批读取，减少内存占用。"}),
            },
        }
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "EXIF_DATA", "EXIF_TEXT")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load"
    CATEGORY = "zveroboy/photo"

    @classmethod
    def IS_CHANGED(cls, raw_file, include_subfolders=False, start_index=0, max_images=0):
        files = photo_paths(raw_file, include_subfolders, start_index, max_images)
        signature = [(str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files]
        return hashlib.sha256(json.dumps(signature).encode("utf-8")).hexdigest()

    def load(self, raw_file, include_subfolders=False, start_index=0, max_images=0):
        images, exifs, summaries = [], [], []
        for path in photo_paths(raw_file, include_subfolders, start_index, max_images):
            image, exif, summary = self.load_one(str(path))
            images.append(image)
            exifs.append(exif)
            summaries.append(summary)
        return {"ui": {"exif_text": summaries}, "result": (images, exifs, summaries)}

    def load_one(self, raw_file):
        img_tensor = None
        embedded = b""
        is_raw = False
        try:
            with Image.open(raw_file) as source:
                metadata = source.getexif()
                if metadata:
                    embedded = metadata.tobytes()
                img_tensor = pil_to_tensor(ImageOps.exif_transpose(source).convert("RGB")).unsqueeze(0)
        except Exception as e:
            print(f"Pillow failed (trying rawpy): {e}")
            if RAW_SUPPORT:
                try:
                    with rawpy.imread(raw_file) as raw:
                        rgb = raw.postprocess()
                        img_pil = Image.fromarray(rgb)
                        img_tensor = pil_to_tensor(img_pil).unsqueeze(0)
                        is_raw = True
                except Exception as e2:
                    raise RuntimeError(f"RAW照片解码失败（{raw_file}）：{e2}") from e2
            else:
                raise RuntimeError("rawpy not installed and Pillow failed.")
        height, width = img_tensor.shape[1:3]
        exif_bytes, summary = prepare_metadata(raw_file, width, height, embedded, is_raw)
        return img_tensor, encode_exif_to_string(exif_bytes), summary

# =============================================================================
# NODE: Photo Save JPG
# =============================================================================

class PhotoSaveJpg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "exif_data": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "zveroboy_photo"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "zveroboy/photo"

    def save(self, images, exif_data, filename_prefix, quality):
        output_dir = folder_paths.get_output_directory()
        exif_bytes = decode_exif_from_string(exif_data)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir, filename, counter, _, _ = folder_paths.get_save_image_path(f"{filename_prefix}_{timestamp}", output_dir)
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i])
            filepath = os.path.join(output_dir, f"{filename}_{counter:05d}_.jpg")
            pil.save(filepath, format="JPEG", quality=quality, exif=exif_bytes, optimize=True)
            counter += 1
        return ()

# =============================================================================
# NODE: Photo Save RAW
# =============================================================================

class PhotoSaveRaw:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "exif_data": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "zveroboy_raw"}),
                "format": (["TIFF", "DNG"], {"default": "TIFF"}),
                "preset": (list(CAMERA_PRESETS.keys()), {"default": "Canon"}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "zveroboy/photo"

    def save(self, images, exif_data, filename_prefix, format, preset):
        output_dir = folder_paths.get_output_directory()
        exif_bytes = decode_exif_from_string(exif_data)
        if not exif_bytes or len(exif_bytes) == 0:
            exif_bytes = None
        preset_data = CAMERA_PRESETS.get(preset, CAMERA_PRESETS["Canon"])
        color_matrix1 = preset_data.get("ColorMatrix1")
        color_matrix2 = preset_data.get("ColorMatrix2")
        neutral = preset_data.get("AsShotNeutral")
        illum1 = preset_data.get("CalibrationIlluminant1", 21)
        illum2 = preset_data.get("CalibrationIlluminant2", 17)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir, filename, counter, _, _ = folder_paths.get_save_image_path(f"{filename_prefix}_{timestamp}", output_dir)
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i])
            ext = "tif" if format == "TIFF" else "dng"
            filepath = os.path.join(output_dir, f"{filename}_{counter:05d}_.{ext}")
            info = ImageFileDirectory_v2()
            if exif_bytes:
                exif = Image.Exif()
                exif.load(exif_bytes)
                for tag in exif:
                    value = exif.get_ifd(tag) if tag in TiffTags.TAGS_V2_GROUPS else exif[tag]
                    if isinstance(value, str) and TiffTags.lookup(tag).type == TiffTags.ASCII:
                        value = value.encode("latin-1")
                    info[tag] = value
            if format == "DNG":
                # The input is rendered 8-bit RGB, not original sensor RAW.
                info[50706] = bytes((1, 4, 0, 0))
                info.tagtype[50706] = TiffTags.BYTE
                info[50707] = bytes((1, 1, 0, 0))
                info.tagtype[50707] = TiffTags.BYTE
                info[50708] = preset_data["Model"]
                for tag, values in ((50721, color_matrix1), (50722, color_matrix2)):
                    info[tag] = tuple(IFDRational(round(value * 10000), 10000) for value in values)
                    info.tagtype[tag] = TiffTags.SIGNED_RATIONAL
                info[50728] = tuple(IFDRational(round(value * 10000), 10000) for value in neutral)
                info.tagtype[50728] = TiffTags.RATIONAL
                info[50778], info[50779] = illum1, illum2
                info.tagtype[50778] = info.tagtype[50779] = TiffTags.SHORT
            pil.save(filepath, format="TIFF", tiffinfo=info, compression="raw")
            counter += 1
        return ()

# =============================================================================
# NODE: Photo Advanced Noise
# =============================================================================

class PhotoAdvancedNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "noise_strength": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.5, "step": 0.001}),
                "color_correlation": ("BOOLEAN", {"default": True, "label": "Simulate Bayer Pattern"}),
                "add_grain_layer": ("BOOLEAN", {"default": False, "label": "Add Film Grain Layer"}),
                "grain_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "jpeg_compression": ("INT", {"default": 95, "min": 70, "max": 100, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "zveroboy/photo"

    def apply(self, images, noise_strength, color_correlation, add_grain_layer, grain_strength, grain_size, jpeg_compression):
        out_list = []
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i])
            arr = np.array(pil, dtype=np.float32)
            shot_noise = np.random.normal(0, 1, arr.shape) * (arr ** 0.5) * (255 * noise_strength * 0.5)
            read_noise = np.random.normal(0, 255 * noise_strength * 0.3, arr.shape)
            if color_correlation:
                shot_noise[:, :, 0] *= 1.2
                shot_noise[:, :, 1] *= 0.9
                shot_noise[:, :, 2] *= 1.4
            arr_noisy = np.clip(arr + shot_noise + read_noise, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr_noisy, mode="RGB")
            if add_grain_layer:
                w, h = pil.size
                grain_arr = np.random.normal(128, 50 * grain_strength, (h, w)).astype(np.float32)
                grain_img = Image.fromarray(np.clip(grain_arr, 0, 255).astype(np.uint8), mode="L")
                if grain_size > 1:
                    gw, gh = w // grain_size, h // grain_size
                    grain_small = grain_img.resize((gw, gh), resample=Image.NEAREST)
                    grain_img = grain_small.resize((w, h), resample=Image.NEAREST)
                pil = Image.blend(pil, grain_img.convert("RGB"), alpha=min(grain_strength * 0.5, 0.3))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(jpeg_compression), subsampling=0)
            buf.seek(0)
            pil = Image.open(buf).convert("RGB")
            out_list.append(pil_to_tensor(pil))
        return (torch.stack(out_list, dim=0),)

# =============================================================================
# NODE: Realism7 Noise+EXIF
# =============================================================================

class Realism7NoiseExif:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (list(CAMERA_PRESETS.keys()), {"default": "Canon"}),
                "noise_level": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.20, "step": 0.01}),
                "jpeg_quality_first": ("INT", {"default": 88, "min": 70, "max": 98, "step": 1}),
                "jpeg_quality_final": ("INT", {"default": 95, "min": 70, "max": 98, "step": 1}),
                "random_color_jitter": ("BOOLEAN", {"default": True}),
                "jitter_strength": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.20, "step": 0.01}),
                "artist": ("STRING", {"default": "VM"}),
                "software": ("STRING", {"default": ""}),
                "copyright": ("STRING", {"default": ""}),
                "body_serial": ("STRING", {"default": ""}),
                "lens_serial": ("STRING", {"default": ""}),
                "focal_length_mm": ("STRING", {"default": "50"}),
                "fnumber": ("STRING", {"default": "4.0"}),
                "exposure_1_over_s": ("STRING", {"default": "125"}),
                "iso": ("INT", {"default": 400, "min": 50, "max": 204800, "step": 1}),
                "exposure_bias_ev": ("STRING", {"default": "0"}),
                "white_balance": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "datetime_original": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "zveroboy/photo"

    def apply(self, images, preset, noise_level, jpeg_quality_first, jpeg_quality_final,
              random_color_jitter, jitter_strength, artist, software, copyright,
              body_serial, lens_serial, focal_length_mm, fnumber, exposure_1_over_s,
              iso, exposure_bias_ev, white_balance, datetime_original):
        if not body_serial.strip():
            body_serial = str(random.randint(1000000, 99999999))
        if not lens_serial.strip():
            lens_serial = str(random.randint(100000000, 999999999))
        if not datetime_original.strip():
            datetime_original = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif_bytes = build_exif_bytes(
            preset_name=preset,
            artist=artist,
            software=software,
            copyright_text=copyright,
            body_serial=body_serial,
            lens_serial=lens_serial,
            focal_length_mm=focal_length_mm,
            fnumber=fnumber,
            exposure_1_over_s=exposure_1_over_s,
            iso=iso,
            exposure_bias_ev=exposure_bias_ev,
            white_balance=white_balance,
            datetime_original=datetime_original,
        )
        out_list = []
        for i in range(images.shape[0]):
            pil = tensor_to_pil(images[i])
            if random_color_jitter and jitter_strength > 0:
                s = float(jitter_strength)
                pil = ImageEnhance.Brightness(pil).enhance(random.uniform(1.0 - s, 1.0 + s))
                pil = ImageEnhance.Color(pil).enhance(random.uniform(1.0 - (s * 0.8), 1.0 + (s * 0.8)))
                pil = ImageEnhance.Contrast(pil).enhance(random.uniform(1.0 - (s * 0.7), 1.0 + (s * 0.7)))
            arr = np.array(pil, dtype=np.float32)
            noise = np.random.normal(0, 255 * noise_level, arr.shape)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                noise[:, :, 2] *= 0.8
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            buf1 = io.BytesIO()
            pil.save(buf1, format="JPEG", quality=int(jpeg_quality_first), optimize=True, subsampling=0)
            buf1.seek(0)
            img_jpeg = Image.open(buf1).convert("RGB")
            buf2 = io.BytesIO()
            img_jpeg.save(buf2, format="JPEG", quality=int(jpeg_quality_final), exif=exif_bytes, subsampling=0)
            buf2.seek(0)
            final = Image.open(buf2).convert("RGB")
            out_list.append(pil_to_tensor(final))
        out = torch.stack(out_list, dim=0)
        return (out,)

# =============================================================================
# MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "PhotoAddNoise": PhotoAddNoise,
    "PhotoAddGrain": PhotoAddGrain,
    "PhotoAddExif": PhotoAddExif,
    "PhotoLoadRaw": PhotoLoadRaw,
    "PhotoSaveJpg": PhotoSaveJpg,
    "PhotoSaveRaw": PhotoSaveRaw,
    "PhotoAdvancedNoise": PhotoAdvancedNoise,
    "Realism7NoiseExif": Realism7NoiseExif,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoAddNoise": "Photo Add Noise (AI Protection)",
    "PhotoAddGrain": "Photo Add Grain (Aesthetic)",
    "PhotoAddExif": "Photo Add EXIF (Camera Data)",
    "PhotoLoadRaw": "Photo Load RAW/HEIC (iPhone)",
    "PhotoSaveJpg": "Photo Save JPG (with EXIF)",
    "PhotoSaveRaw": "Photo Save RAW (TIFF/DNG)",
    "PhotoAdvancedNoise": "Photo Advanced Noise (Sensor Sim)",
    "Realism7NoiseExif": "Realism7 Noise + EXIF (All-in-One)",
}
