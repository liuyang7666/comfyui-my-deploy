import base64
import io
import os
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageOps

from comfy_api.latest import io as comfy_io

from .nodes import credentials, request_json

IMAGE_URL = "https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation"
IMAGE_MODELS = ["qwen-image-3.0", "qwen-image-3.0-pro"]
class QwenImageAPI(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="QwenImageAPI",
            display_name="千问 Image 3.0 / Pro · 生图",
            category="MY/千问API",
            inputs=[
                comfy_io.Autogrow.Input("reference_images", optional=True,
                    template=comfy_io.Autogrow.TemplateNames(
                        comfy_io.Image.Input("reference_image"),
                        names=["参考图1", "参考图2", "参考图3"], min=0)),
                comfy_io.Combo.Input("model", options=IMAGE_MODELS, default="qwen-image-3.0-pro", display_name="模型"),
                comfy_io.String.Input("prompt", multiline=True, default="", display_name="正向提示词"),
                comfy_io.String.Input("negative_prompt", multiline=True, default="", display_name="负向提示词"),
                comfy_io.Int.Input("seed", default=0, min=0, max=2147483647, control_after_generate=True),
            ],
            outputs=[comfy_io.Image.Output(display_name="图像")],
        )

    @classmethod
    def execute(cls, model, prompt, negative_prompt, seed, reference_images=None):
        if model not in IMAGE_MODELS:
            raise RuntimeError("请选择支持的图像模型。")
        key = os.environ.get("DASHSCOPE_API_KEY") or credentials().get("api_key")
        if not key:
            raise RuntimeError("未读取到千问 API 密钥，请运行节点目录中的配置密钥.ps1。")
        parameters = {"n": 1, "prompt_extend": True, "watermark": False, "seed": seed}
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt
        content = []
        if reference_images is not None:
            for name in ["参考图1", "参考图2", "参考图3"]:
                batch = reference_images.get(name)
                if batch is None:
                    continue
                for tensor in batch:
                    pixels = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).convert("RGB").save(buffer, format="PNG")
                    content.append({"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
        content.append({"text": prompt})
        payload = {"model": model, "input": {"messages": [{"role": "user", "content": content}]}, "parameters": parameters}
        response = request_json(IMAGE_URL, payload, key, timeout=600)
        if response.get("code"):
            raise RuntimeError(f"千问生图失败：{response['code']}；request_id={response.get('request_id', 'unknown')}")
        images = [item["image"] for choice in response.get("output", {}).get("choices", [])
                  for item in choice.get("message", {}).get("content", []) if item.get("image")]
        if not images:
            raise RuntimeError(f"千问未返回图像；request_id={response.get('request_id', 'unknown')}。请求未自动重试。")
        image_url = images[0]
        if not image_url.startswith("https://"):
            raise RuntimeError("千问返回的图像地址不是 HTTPS。")
        try:
            with urllib.request.urlopen(image_url, timeout=120) as download:
                data = download.read()
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("图像已生成，但下载失败。请到千问控制台请求记录保存结果；未自动重新生图。") from None
        with Image.open(io.BytesIO(data)) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0
        return comfy_io.NodeOutput(torch.from_numpy(pixels.copy()).unsqueeze(0))
