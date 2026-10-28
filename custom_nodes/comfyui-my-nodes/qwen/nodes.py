import base64
import ctypes
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
def credential_path():
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x001C, None, 0, buffer) != 0:
            raise RuntimeError("无法定位 Windows 本地应用数据目录。")
        directory = Path(buffer.value)
    else:
        directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return directory / "ComfyUI-QwenAPI" / "credentials.dpapi"


SECRET = credential_path()
CATALOG = json.loads((ROOT / "trial_models.json").read_text(encoding="utf-8"))
MODELS = ['qwen3.8-flash', 'deepseek-v4.1-flash', 'qwen3.7-plus', 'qwen3.8-max']
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"



class Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def credentials():
    if not SECRET.exists():
        return {}
    if os.name != "nt":
        raise RuntimeError("DPAPI 密钥只能由保存密钥的 Windows 用户读取。")
    encrypted = SECRET.read_bytes()
    buf = ctypes.create_string_buffer(encrypted)
    src = Blob(len(encrypted), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    dst = Blob()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise RuntimeError("无法解密密钥，请运行配置密钥.ps1重新保存。")
    try:
        return json.loads(ctypes.string_at(dst.data, dst.size).decode("utf-8"))
    finally:
        ctypes.windll.kernel32.LocalFree(dst.data)


def request_json(url, payload=None, key=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if payload and payload.get("stream"):
                parts = []
                reasoning_chars = 0
                finish_reason = None
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError("模型流式返回错误：" + str(chunk["error"].get("code", "unknown")))
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        reasoning_chars += len(delta.get("reasoning_content") or "")
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                        content = delta.get("content")
                        if content:
                            parts.append(content)
                answer = "".join(parts)
                if not answer:
                    raise RuntimeError(f"模型未返回回答：finish_reason={finish_reason}，思考字符数={reasoning_chars}；请求未自动重试。")
                return {"choices": [{"message": {"content": answer}}]}
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(body)
            code = error.get("error", error).get("code", "unknown")
        except (ValueError, AttributeError):
            code = "unknown"
        raise RuntimeError(f"千问 API HTTP {exc.code} ({code})。检查地域、密钥、模型权限和额度；请求未自动重试。") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("API 连接失败或超时；未自动重试，避免重复计费。") from None


class QwenAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (MODELS, {"default": "qwen3.8-flash"}),
            "prompt": ("STRING", {"multiline": True, "default": "请用中文回答；如果有图片，请描述图片的内容。"}),
        }, "optional": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回答",)
    FUNCTION = "run"
    CATEGORY = "MY/千问API"

    def run(self, model, prompt, images=None):
        secrets = credentials()
        key = os.environ.get("DASHSCOPE_API_KEY") or secrets.get("api_key")
        if not key:
            raise RuntimeError(f"未读取到千问 API 密钥。密钥文件：{SECRET}；请运行节点目录中的配置密钥.ps1。")
        capability = CATALOG[model]
        content = [{"type": "text", "text": prompt}]
        if images is not None:
            if not capability["vision"]:
                raise RuntimeError("所选模型仅支持文字，请选择支持图片的模型，例如 qwen3.8-flash。")
            for tensor in images:
                pixels = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(pixels).convert("RGB").save(buf, format="PNG")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
        messages = [{"role": "user", "content": content if images is not None else prompt}]
        payload = {"model": model, "messages": messages, "max_tokens": 16384 if capability["thinking_only"] else 2048, "temperature": 0.7, "stream": True}
        if capability["hybrid"] and not capability["thinking_only"]:
            payload["enable_thinking"] = False
        if capability["family"] in {"千问 Omni", "Qwen-Omni"}:
            payload["modalities"] = ["text"]
        response = request_json(URL, payload, key)
        return (response["choices"][0]["message"].get("content") or "",)
