from fractions import Fraction
import os
from pathlib import Path
import subprocess

import piexif


def extract_raw_exif(path):
    executable = Path(__file__).parent / "vendor/exiftool/exiftool.exe"
    if not executable.is_file():
        raise FileNotFoundError("缺少本地ExifTool，无法读取RAW拍摄信息。")
    result = subprocess.run(
        [str(executable), "-charset", "filename=UTF8", "-o", "-.exif", "-EXIF:all", "-@", "-"],
        input=(str(path) + "\n").encode("utf-8"), capture_output=True,
        timeout=30, env=dict(os.environ, PERL_BADLANG="0"), creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def value_text(value):
    if value is None:
        return "未记录"
    if isinstance(value, bytes):
        return value.rstrip(b"\0").decode("utf-8", errors="replace")
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, int) for item in value):
        return f"{float(Fraction(*value)):g}" if value[1] else "未记录"
    return str(value)


def prepare_metadata(path, width, height, embedded=b"", is_raw=False):
    prefix = f"文件：{Path(path).name}\n图像尺寸：{width} × {height}"
    try:
        raw_bytes = extract_raw_exif(path) if is_raw else embedded
        if not raw_bytes:
            return b"", prefix + "\nEXIF：源文件未包含可读取的拍摄信息。"
        data = piexif.load(raw_bytes)
        main, exif = data["0th"], data["Exif"]
        if not main and not exif and not data["GPS"]:
            return b"", prefix + "\nEXIF：源文件未包含可读取的拍摄信息。"
        exposure = exif.get(piexif.ExifIFD.ExposureTime)
        shutter = str(Fraction(*exposure)) + " 秒" if exposure and exposure[1] else "未记录"
        lines = [prefix]
        fields = [
            ("相机品牌", main.get(piexif.ImageIFD.Make)),
            ("相机型号", main.get(piexif.ImageIFD.Model)),
            ("镜头", exif.get(piexif.ExifIFD.LensModel)),
            ("焦距（毫米）", exif.get(piexif.ExifIFD.FocalLength)),
            ("光圈 f/", exif.get(piexif.ExifIFD.FNumber)),
            ("快门", shutter),
            ("ISO", exif.get(piexif.ExifIFD.ISOSpeedRatings)),
            ("曝光补偿（EV）", exif.get(piexif.ExifIFD.ExposureBiasValue)),
            ("拍摄时间", exif.get(piexif.ExifIFD.DateTimeOriginal)),
            ("作者", main.get(piexif.ImageIFD.Artist)),
            ("处理软件", main.get(piexif.ImageIFD.Software)),
        ]
        lines.extend(f"{label}：{value_text(value)}" for label, value in fields)
        # Pixels are already rendered in display orientation by the image decoder.
        main[piexif.ImageIFD.Orientation] = 1
        main[piexif.ImageIFD.ImageWidth] = width
        main[piexif.ImageIFD.ImageLength] = height
        exif[piexif.ExifIFD.PixelXDimension] = width
        exif[piexif.ExifIFD.PixelYDimension] = height
        data["1st"], data["thumbnail"] = {}, None
        return piexif.dump(data), "\n".join(lines)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, piexif.InvalidImageDataError) as error:
        return b"", prefix + f"\nEXIF读取失败：{error}"
