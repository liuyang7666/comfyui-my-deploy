import base64
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

import folder_paths
import comfy.model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io as IO

ROOT = Path(__file__).parent
LOGGER = logging.getLogger(__name__)
SIZES = [
    "1280x1280", "848x1280", "1280x848", "960x1280", "1280x960",
    "1024x1280", "1280x1024", "720x1280", "1280x720", "1280x544",
    "2048x2048", "1360x2048", "2048x1360", "1536x2048", "2048x1536",
    "1632x2048", "2048x1632", "1152x2048", "2048x1152", "2048x864",
    "2880x2880", "2336x3520", "3520x2336", "2480x3312", "3312x2480",
    "2560x3216", "3216x2560", "2160x3840", "3840x2160", "3840x1632",
]
BACKGROUNDS = {"自动": "auto", "不透明": "opaque", "透明": "transparent"}
QUALITIES = {"低": "low", "中": "medium", "高": "high", "超高": "xhigh", "最高": "max"}
MB = 1024 * 1024


def settings():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["base_url"] = os.environ.get("APIYI_BASE_URL", config["base_url"]).rstrip("/")
    if urlsplit(config["base_url"]).scheme != "https":
        raise ValueError("APIYI_BASE_URL 必须使用 HTTPS。")
    config["timeout"] = max(360, float(config["timeout"]))
    if config["response_format"] not in ("url", "b64_json"):
        raise ValueError("response_format 只能是 url 或 b64_json。")
    return config


def compress_upload(raw, target=1.5 * MB, force=False):
    if len(raw) <= target and not force:
        return raw
    try:
        with Image.open(io.BytesIO(raw)) as original:
            fmt = original.format
            image = ImageOps.exif_transpose(original).copy()
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        for _ in range(16):
            output = io.BytesIO()
            image.save(output, format=fmt, quality=90, optimize=True)
            candidate = output.getvalue()
            if len(candidate) <= target:
                return candidate
            image.thumbnail((max(1, int(image.width * 0.8)), max(1, int(image.height * 0.8))), Image.Resampling.LANCZOS)
        return candidate if len(candidate) < len(raw) else raw
    except (OSError, ValueError, KeyError):
        LOGGER.warning("API易GPT：参考图压缩失败，将使用该图原始数据继续请求。")
        return raw


def mask_upload(mask, source_size, upload_size):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("蒙版须为单张 MASK，不能接入多张蒙版批次。")
    if tuple(mask.shape[1:]) != (source_size[1], source_size[0]):
        raise ValueError("蒙版宽高必须与第一张输入图像一致。")
    weights = mask.detach().cpu().float().numpy()[0]
    if not np.isfinite(weights).all():
        raise ValueError("蒙版包含无效数值。")
    weights = weights.clip(0, 1)
    if upload_size != source_size:
        weights = np.asarray(Image.fromarray(weights).resize(upload_size, Image.Resampling.BILINEAR))
    rgba = np.zeros((upload_size[1], upload_size[0], 4), dtype=np.uint8)
    # API mask transparency denotes the white/edit area of a ComfyUI MASK.
    rgba[:, :, 3] = ((1 - weights) * 255).round().astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgba).save(output, format="PNG")
    return ("mask", ("mask.png", output.getvalue(), "image/png"))


def prepare_uploads(raw_images, mask=None):
    if mask is not None and not raw_images:
        raise ValueError("接入蒙版时必须同时接入要编辑的图像。")
    images = [compress_upload(raw) for raw in raw_images]
    if sum(map(len, images)) > 6 * MB:
        target = min(1.5 * MB, 6 * MB / len(images))
        images = [compress_upload(raw, target, force=len(raw) > target) for raw in images]
    if any(len(raw) > 10 * MB for raw in images) or sum(map(len, images)) > 6 * MB:
        LOGGER.warning("API易GPT：压缩回退后参考图仍超过上传建议大小；按要求保留原图继续上传。")
    files = []
    for index, raw in enumerate(images, 1):
        with Image.open(io.BytesIO(raw)) as image:
            fmt = image.format
        extension, mime = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"), "WEBP": ("webp", "image/webp")}[fmt]
        files.append(("image", (f"image_{index}.{extension}", raw, mime)))
    if mask is not None:
        with Image.open(io.BytesIO(raw_images[0])) as source, Image.open(io.BytesIO(images[0])) as uploaded:
            files.append(mask_upload(mask, source.size, uploaded.size))
    return files


def image_bytes(tensor):
    pixels = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def error_message(response, key):
    try:
        body = response.json()
        error = body.get("error", body)
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
    except ValueError:
        message = response.text[:800]
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[密钥已隐藏]", str(message).replace(key, "[密钥已隐藏]"))[:1000]


def is_saturated(message):
    message = message.lower()
    if any(word in message for word in ("must be", "invalid", "parameter", "quality", "不支持", "参数", "余额", "quota", "额度")):
        return False
    return any(word in message for word in ("饱和", "负载", "overload", "saturat", "capacity", "rate limit", "too many requests"))


def request_image(config, payload, files=None):
    key = os.environ.get("APIYI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 APIYI_API_KEY。请运行节点目录中的 配置密钥.ps1，然后重新启动绘世/ComfyUI。")
    endpoint = "/images/edits" if files else "/images/generations"
    for attempt in range(int(config["saturation_retries"]) + 1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        try:
            response = requests.post(
                config["base_url"] + endpoint,
                headers={"Authorization": "Bearer " + key},
                **({"data": payload, "files": files} if files else {"json": payload}),
                timeout=config["timeout"], allow_redirects=False,
            )
        except requests.RequestException:
            if config.get("fallback_mode"):
                raise RuntimeError("API易连接中断或超时，原请求结果未知。") from None
            raise RuntimeError("API易连接中断或超时；未重新发起生图，避免重复计费。请在 API易调用日志检查本次状态。") from None
        if response.status_code >= 300:
            message = error_message(response, key)
            if response.status_code == 429 and is_saturated(message) and attempt < config["saturation_retries"]:
                delay = min(60, 5 * 2 ** attempt)
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = min(60, max(delay, int(retry_after)))
                LOGGER.warning("API易GPT：上游饱和，%s 秒后重试。", delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"API易 HTTP {response.status_code}：{message}。状态：https://docs.apiyi.com/live")
        try:
            result = response.json()
        except ValueError:
            if config.get("fallback_mode"):
                raise RuntimeError("API易响应不是 JSON。") from None
            raise RuntimeError("API易响应不是 JSON；未重试，请检查调用日志。") from None
        if not isinstance(result, dict) or not result.get("data"):
            if config.get("fallback_mode"):
                raise RuntimeError("API易未返回图片。")
            raise RuntimeError("API易未返回图片；未重试，请检查调用日志及是否已扣费。")
        if config.get("fallback_mode") and (not isinstance(result["data"], list) or len(result["data"]) != 1
                or not isinstance(result["data"][0], dict)
                or not (result["data"][0].get("b64_json") or result["data"][0].get("url"))):
            raise RuntimeError("API易未返回有效的单张图片结果。")
        return result, response.headers.get("x-request-id") or response.headers.get("request-id")


def result_bytes(item, timeout):
    if item.get("b64_json"):
        encoded = item["b64_json"]
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        return base64.b64decode(encoded)
    url = item.get("url", "")
    if urlsplit(url).scheme != "https":
        raise RuntimeError("API易未返回有效的 base64 图片或 HTTPS 图片链接。")
    # CDN 请求不携带 API 密钥；只重试下载，不会重新生图。
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError("图片已生成但 CDN 下载失败；链接保存在本次 JSON 记录中，请在失效前下载，未重新生图。") from None


def account_query(config, path, params=None):
    token = os.environ.get("APIYI_ACCOUNT_TOKEN", "").strip()
    if not token:
        return None, "未配置 APIYI_ACCOUNT_TOKEN"
    address = urlsplit(config["base_url"])
    try:
        response = requests.get(
            f"{address.scheme}://{address.netloc}{path}",
            headers={"Authorization": token, "Accept": "application/json"},
            params=params, timeout=20, allow_redirects=False,
        )
        if response.status_code == 403:
            scope = "个人资料·读取" if path == "/api/user/self" else "用量与个人记录·读取"
            return None, f"权限不足，请勾选{scope}"
        if response.status_code == 401:
            return None, "查询令牌无效或已过期"
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            return None, "查询接口未返回成功状态"
        return body.get("data"), None
    except (requests.RequestException, ValueError):
        return None, "查询失败或超时，图片已保留"


def query_billing(config, request_id, started_at):
    billing = {"cost_usd": None, "balance_usd": None}
    if request_id:
        now = int(time.time())
        rows, error = account_query(config, "/api/log/self", {
            "request_id": request_id, "type": 2, "p": 0, "pageSize": 1000,
            "start_timestamp": max(now - 86400, int(started_at) - 60), "end_timestamp": now + 60,
        })
        if isinstance(rows, dict):
            rows = rows.get("items") or rows.get("logs") or []
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0].get("quota"), (int, float)):
            # 只保存必要字段；API易原始日志可能包含未带 sk- 前缀的密钥。
            row = rows[0]
            billing.update(cost_usd=row["quota"] / 500000, quota=row["quota"],
                           log_request_id=row.get("request_id"), source="API易消费日志实扣")
        else:
            billing["cost_status"] = error or "账单尚未入库或无法唯一匹配；请稍后查询"
    else:
        billing["cost_status"] = "响应未提供 request_id，无法精确查账"
    account, error = account_query(config, "/api/user/self")
    if isinstance(account, dict) and isinstance(account.get("quota"), (int, float)):
        billing.update(balance_usd=account["quota"] / 500000, balance_quota=account["quota"],
                       balance_checked_at=int(time.time()))
    else:
        billing["balance_status"] = error or "余额接口未返回额度"
    return billing


def billing_line(index, billing):
    cost = f"实扣 ${billing['cost_usd']:.6f}" if billing.get("cost_usd") is not None else "费用：" + billing.get("cost_status", "待查询")
    balance = f"余额 ${billing['balance_usd']:.6f}" if billing.get("balance_usd") is not None else "余额：" + billing.get("balance_status", "待查询")
    return f"第 {index} 次：{cost}\n{balance}"


def model_inputs(model):
    qualities = list(QUALITIES) if model != "gpt-image-2-vip" else list(QUALITIES)[:3]
    return [
        IO.Combo.Input("size", display_name="尺寸", options=["自动"] + SIZES + ["自定义"], default="自动",
                       tooltip="自定义时读取自定义宽高；VIP 可能自动调整尺寸，以实际返回为准。"),
        IO.Combo.Input("background", display_name="背景", options=list(BACKGROUNDS), default="自动"),
        IO.Combo.Input("quality", display_name="质量", options=qualities, default="低",
                       tooltip="2.5 的高≈2-vip 的中；2.5 的最高≈2-vip 的高。"),
        IO.Autogrow.Input("images", optional=True,
            template=IO.Autogrow.TemplateNames(IO.Image.Input("image"), names=[f"图像_{i}" for i in range(1, 17)], min=0),
            tooltip="不接图为文生图；接入图片为编辑/融合。IMAGE 张量已无原文件格式，上传前统一编码 PNG。"),
    ]


class APIYiGPT(IO.ComfyNode):
    @classmethod
    def generate_image(cls, config, payload, files, seed):
        return request_image(config, payload, files)

    @classmethod
    def define_schema(cls):
        config = settings()
        models = config["models"]
        if config["default_model"] in models:
            models = [config["default_model"]] + [model for model in models if model != config["default_model"]]
        return IO.Schema(
            node_id="APIYiGPT", display_name="API易GPT", category="MY/API易",
            description="VIP 文生图/图片编辑，支持可选蒙版输入。图片自动保存至 output/APIYI-GPT。VIP 接受蒙版但不保证精确局部重绘。",
            search_aliases=["APIYI", "GPT Image", "API易GPT"], is_output_node=True,
            inputs=[
                IO.String.Input("prompt", display_name="提示词", multiline=True, default=""),
                IO.DynamicCombo.Input("model", display_name="模型",
                    options=[IO.DynamicCombo.Option(model, model_inputs(model)) for model in models]),
                IO.Int.Input("n", display_name="数量", default=1, min=1, max=8, step=1,
                    display_mode=IO.NumberDisplay.number, tooltip="每张独立请求、独立计费；不会向 VIP 接口发送 n。"),
                IO.Int.Input("seed", display_name="种子", default=0, min=0, max=2147483647,
                    display_mode=IO.NumberDisplay.number, control_after_generate=True,
                    tooltip="仅控制 ComfyUI 缓存/重新执行；VIP 无固定种子复现参数，不向 API 传 seed。"),
                IO.Int.Input("custom_width", display_name="自定义宽度", optional=True, default=1024,
                    min=1, max=2147483647, step=1, display_mode=IO.NumberDisplay.number,
                    tooltip="单位像素，仅尺寸选自定义时生效。建议用 16 的倍数；表外尺寸可能被接口改写。"),
                IO.Int.Input("custom_height", display_name="自定义高度", optional=True, default=1024,
                    min=1, max=2147483647, step=1, display_mode=IO.NumberDisplay.number,
                    tooltip="单位像素，仅尺寸选自定义时生效。建议用 16 的倍数；过小尺寸可能被接口放大。"),
                IO.Mask.Input("mask", display_name="蒙版", optional=True,
                    tooltip="白色编辑、黑色保留；尺寸须与第一张图像一致。VIP可能改动蒙版外区域；官方备用的蒙版编辑仅支持单张输入图像。"),
            ],
            outputs=[IO.Image.Output(display_name="图像", is_output_list=True)],
        )

    @classmethod
    def execute(cls, prompt, model, n=1, seed=0, custom_width=1024, custom_height=1024, mask=None):
        config = settings()
        model_id = model["model"]
        if model_id not in config["models"]:
            raise ValueError("模型不在 config.json 配置列表中。")
        size = model["size"]
        if size == "自定义":
            if type(custom_width) is not int or type(custom_height) is not int or min(custom_width, custom_height) < 1:
                raise ValueError("自定义宽度和高度必须是大于 0 的整数像素。")
            size = f"{custom_width}x{custom_height}"
        elif size not in ["自动", "auto"] + SIZES:
            raise ValueError("请选择预设尺寸、自动或自定义。")
        quality = QUALITIES.get(model["quality"], model["quality"])
        if quality not in (list(QUALITIES.values()) if model_id != "gpt-image-2-vip" else ["low", "medium", "high"]):
            raise ValueError("当前模型不支持此质量档位，请重新选择。")
        if not prompt.strip():
            raise ValueError("请填写提示词。")
        payload = {"model": model_id, "prompt": prompt, "size": "auto" if size == "自动" else size,
                   "quality": quality, "background": BACKGROUNDS.get(model["background"], model["background"]),
                   "response_format": config["response_format"]}
        raw_images = []
        image_map = model.get("images") or {}
        for name in sorted(image_map, key=lambda name: int(name.rsplit("_", 1)[-1])):
            batch = image_map[name]
            if batch is not None:
                raw_images.extend(image_bytes(tensor) for tensor in (batch if batch.ndim == 4 else batch.unsqueeze(0)))
        files = prepare_uploads(raw_images, mask) if raw_images or mask is not None else None
        mask_info = None
        if mask is not None:
            with Image.open(io.BytesIO(files[-1][1][1])) as uploaded_mask:
                mask_info = {"white_means": "edit", "applies_to": "image_1", "upload_size": list(uploaded_mask.size)}
        directory = Path(folder_paths.get_output_directory()) / "APIYI-GPT"
        directory.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        tensors, previews, bills, dimensions = [], [], [], []
        progress = ProgressBar(n)
        for index in range(n):
            record_path = directory / f"{run_id}_{index + 1:02d}.json"
            try:
                started_at = int(time.time())
                started = time.monotonic()
                result, request_id = cls.generate_image(config, payload, files, seed)
                elapsed = round(time.monotonic() - started, 2)
                provider = result.get("provider", "apiyi")
                apiyi_model = result.get("apiyi_model", payload["model"]) if provider == "apiyi" else None
                vip = provider == "apiyi" and apiyi_model.endswith("-vip")
                record = {"request": payload, "request_id": request_id, "started_at": started_at, "seconds": elapsed,
                          "provider": provider, "attempts": result.get("attempts", []),
                          "apiyi_model": apiyi_model, "apiyi_tier": result.get("apiyi_tier", "vip" if vip else None),
                          "official_model": result.get("official_model"),
                          "usage": result.get("usage"), "price_usd_estimate": 0.03 if vip else None,
                          "mask": mask_info,
                          "price_source": ("API易 VIP 官方标价；非账户账单实扣验证" if vip else
                                           "API易非 VIP；费用以消费日志实扣为准" if provider == "apiyi" else
                                           "ComfyUI 账户积分；实扣请查看官方账户记录"),
                          "images": [{"url": item["url"]} if item.get("url") else {"response_format": "b64_json"} for item in result["data"]]}
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                for item_index, item in enumerate(result["data"], 1):
                    raw = result_bytes(item, config["timeout"])
                    with Image.open(io.BytesIO(raw)) as generated:
                        image = ImageOps.exif_transpose(generated).convert("RGBA" if "A" in generated.getbands() or "transparency" in generated.info else "RGB")
                    filename = f"{run_id}_{index + 1:02d}_{item_index}.png"
                    image.save(directory / filename)
                    tensors.append(torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0))
                    previews.append({"filename": filename, "subfolder": "APIYI-GPT", "type": "output"})
                    actual_size = f"{image.width}x{image.height}"
                    adjusted = payload["size"] != "auto" and payload["size"] != actual_size
                    dimensions.append(f"第 {len(tensors)} 张：请求 {payload['size']} → 实际 {actual_size}" + ("（接口调整）" if adjusted else ""))
                    record["images"][item_index - 1].update({"filename": filename, "size": list(image.size), "size_adjusted": adjusted})
                    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                if provider == "official":
                    billing = {"cost_usd": None, "balance_usd": None,
                               "cost_status": "已切换 ComfyUI 官方节点，使用账户积分",
                               "balance_status": "请查看 ComfyUI 账户记录"}
                else:
                    billing = query_billing(config, request_id, started_at)
                if result.get("attempts") or provider == "official":
                    attempts = result.get("attempts", [])
                    vip_count = sum(attempt.get("tier", "vip") == "vip" for attempt in attempts)
                    non_vip_count = sum(attempt.get("tier") == "non_vip" for attempt in attempts)
                    route = f"API易 VIP 调用 {vip_count} 次"
                    if non_vip_count:
                        route += f" → 非 VIP 调用 {non_vip_count} 次"
                    route += " → ComfyUI 官方节点" if provider == "official" else "成功"
                    dimensions.append(f"第 {index + 1} 次路线：{route}")
                bills.append(billing)
                record["billing"] = billing
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                progress.update(1)
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}\n本批此前完成的图片和调用记录位于：{directory}；当前记录：{record_path.name}") from None
        lines = [billing_line(index, bill) for index, bill in enumerate(bills, 1)]
        if all(bill.get("cost_usd") is not None for bill in bills):
            lines.append(f"本批合计：${sum(bill['cost_usd'] for bill in bills):.6f} USD（实扣）")
        return IO.NodeOutput(tensors, ui={"images": previews, "apiyi_dimensions": dimensions, "apiyi_billing": lines})
