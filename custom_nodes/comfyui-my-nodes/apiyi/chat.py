"""APIYi multimodal chat node for image understanding."""

import base64
import io
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
from PIL import Image

import comfy.model_management
from comfy_api.latest import io as IO


ROOT = Path(__file__).parent
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-6-luna",
    "deepseek-v4-flash-vision-exp",
    "qwen3.8-flash",
]
MAX_IMAGES = 16
MAX_IMAGE_EDGE = 2048
REQUEST_TIMEOUT = (15, 240)


def api_base_url():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    base_url = os.environ.get("APIYI_BASE_URL", config["base_url"]).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("API易地址必须是无账号、查询参数和片段的 HTTPS 地址。")
    return base_url


def image_data_uri(tensor):
    pixels = tensor.detach().cpu().float().numpy()
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
        raise ValueError("图片张量应为 H×W×3/4。")
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
    pixels = (pixels.clip(0, 1) * 255).round().astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def image_parts(images):
    if images is None:
        return [], 0
    if not hasattr(images, "ndim"):
        raise ValueError("IMAGE 输入格式无效。")
    if images.ndim == 3:
        batch = [images]
    elif images.ndim == 4:
        batch = images
    else:
        raise ValueError("IMAGE 输入应为 H×W×C 或 B×H×W×C。")
    if len(batch) > MAX_IMAGES:
        raise ValueError(f"单次最多识别 {MAX_IMAGES} 张图片；请缩小批次后再运行。")
    parts = [{"type": "image_url", "image_url": {"url": image_data_uri(item)}} for item in batch]
    return parts, len(batch)


def response_error(response, api_key):
    try:
        body = response.json()
        error = body.get("error", body)
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        message = response.text[:800]
    message = str(message).replace(api_key, "[密钥已隐藏]")
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[密钥已隐藏]", message)
    return message[:1000]


def response_text(body):
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("API易响应缺少 choices；请检查模型权限和调用记录。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        answer = content
    elif isinstance(content, list):
        answer = "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("text")
        )
    else:
        answer = ""
    if not answer.strip():
        raise RuntimeError("API易未返回文本回答；请检查模型的 finish_reason 和调用记录。")
    return answer


class APIYiVisionChat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="APIYiVisionChat",
            display_name="API易多模态聊天（识图）",
            category="MY/API易",
            description="调用 API易视觉模型分析一张或多张图片并生成文字回答。图片缩放至长边 2048 后以 Base64 发送；不会自动重试。",
            search_aliases=["APIYI", "Vision Chat", "图像理解", "图片识别"],
            inputs=[
                IO.Combo.Input(
                    "model", display_name="模型", options=MODELS, default="gemini-3.8-flash",
                    tooltip="精选视觉模型：Gemini 3.8 Flash（日常均衡）、Gemini 3.5 Flash Lite（快速批量）、GPT-6.1 Sol（复杂分析）、GPT-6.1 Terra（视觉推理与性价比）、GPT-6 Luna（低成本概述，细小 OCR 可能漏字）、DeepSeek V4 Flash Vision（低图像 token）、Qwen 3.8 Flash（中文性价比）。",
                ),
                IO.String.Input(
                    "prompt", display_name="提示词", multiline=True,
                    default="请用中文描述图片内容，并准确回答我提出的问题。若有多张图，请按图 1、图 2……的顺序分析。",
                ),
                IO.String.Input("system_prompt", display_name="系统提示", multiline=True, optional=True, default=""),
                IO.Int.Input(
                    "max_tokens", display_name="最大回复 Token", default=2048,
                    min=64, max=8192, step=64,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="限制回复长度；提高上限可能增加输出费用。",
                ),
                IO.Image.Input(
                    "images", display_name="图片", optional=True,
                    tooltip="可连接单张或批次图片；批次中每张图都会作为同一条消息的图片输入，最多 16 张。上传前自动缩放至长边 2048。",
                ),
            ],
            outputs=[IO.String.Output(display_name="回答"), IO.String.Output(display_name="用量")],
        )

    @classmethod
    def execute(cls, model, prompt, system_prompt="", max_tokens=2048, images=None):
        if model not in MODELS:
            raise ValueError("请选择本节点提供的视觉模型。")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请填写提示词。")
        api_key = os.environ.get("APIYI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未设置 APIYI_API_KEY。请运行节点目录中的 配置密钥.ps1，然后重新启动 ComfyUI。")
        if type(max_tokens) is not int or not 64 <= max_tokens <= 8192:
            raise ValueError("最大回复 Token 必须在 64 到 8192 之间。")

        image_content, image_count = image_parts(images)
        user_content = [{"type": "text", "text": prompt}]
        user_content.extend(image_content)
        messages = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content if image_count else prompt})
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}

        comfy.model_management.throw_exception_if_processing_interrupted()
        try:
            response = requests.post(
                api_base_url() + "/chat/completions",
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException:
            raise RuntimeError("API易连接失败或超时；本节点不会自动重试，以免重复计费。") from None
        comfy.model_management.throw_exception_if_processing_interrupted()
        if response.status_code >= 300:
            raise RuntimeError(f"API易 HTTP {response.status_code}：{response_error(response, api_key)}")
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError("API易响应不是有效 JSON；本节点不会自动重试，请检查调用记录。") from None
        if not isinstance(body, dict):
            raise RuntimeError("API易响应格式无效。")

        answer = response_text(body)
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None:
            usage_text = f"模型：{model}；图片：{image_count} 张；API 未返回 token 用量"
            if image_count:
                usage_text += "（请在 API易调用记录确认图片已计入输入）"
        else:
            usage_text = (
                f"模型：{model}；图片：{image_count} 张；输入：{prompt_tokens} tokens；"
                f"输出：{completion_tokens if completion_tokens is not None else '未知'} tokens"
            )
            if total_tokens is not None:
                usage_text += f"；合计：{total_tokens} tokens"
            if image_count and prompt_tokens < 64 * image_count:
                usage_text += "；提示：输入 token 偏少，请确认所选模型确实接收了图片"
        return IO.NodeOutput(answer, usage_text)
