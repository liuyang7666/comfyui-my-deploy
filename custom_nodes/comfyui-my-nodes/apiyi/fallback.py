"""Retry APIYi VIP, try non-VIP once, then use the installed official node."""
import asyncio
import base64
import os
import time

import comfy.model_management
from comfy_api.latest import io as IO
from comfy_api_nodes.nodes_openai import OpenAIGPTImageNodeV2

from . import nodes


OFFICIAL_MODELS = {
    "gpt-image-2.5-vip": "gpt-image-2.5-flare",
    "gpt-image-2.5-flare-vip": "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst-vip": "gpt-image-2.5-sunburst",
    "gpt-image-2-vip": "gpt-image-2",
}


def retry_wait(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        comfy.model_management.throw_exception_if_processing_interrupted()
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))


class APIYiGPTFallback(nodes.APIYiGPT):
    @classmethod
    async def execute(cls, prompt, model, n=1, seed=0, custom_width=1024, custom_height=1024, mask=None):
        # Keep blocking requests and the official asyncio.run off ComfyUI's event loop.
        return await asyncio.to_thread(super().execute, prompt, model, n, seed, custom_width, custom_height, mask=mask)

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "APIYiGPTFallback"
        schema.display_name = "API易GPT（官方备用）"
        schema.search_aliases = ["APIYI", "GPT Image", "API易GPT备用", "官方备用"]
        schema.description = "API易VIP首次失败后重试3次，再尝试对应非VIP模型1次，仍失败调用ComfyUI官方GPT Image节点。非VIP使用API易余额，官方使用ComfyUI积分；断线重试可能重复计费。"
        schema.is_api_node = True
        schema.hidden = [IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id]
        return schema

    @classmethod
    def generate_image(cls, config, payload, files, seed):
        attempts = []
        # Disable nested 429 retries: four VIP POSTs, then one non-VIP POST.
        single = dict(config, saturation_retries=0, fallback_mode=True)
        if os.environ.get("APIYI_API_KEY", "").strip():
            for attempt in range(4):
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    result, request_id = nodes.request_image(single, payload, files)
                except RuntimeError as exc:
                    attempts.append({"number": attempt + 1, "tier": "vip", "model": payload["model"], "error": str(exc)})
                    if attempt < 3:
                        delay = 5 * 2 ** attempt
                        nodes.LOGGER.warning("API易GPT备用：第 %s 次调用失败，%s 秒后重试（%s/3）。", attempt + 1, delay, attempt + 1)
                        retry_wait(delay)
                else:
                    attempts.append({"number": attempt + 1, "tier": "vip", "model": payload["model"], "succeeded": True})
                    result.update(provider="apiyi", attempts=attempts, apiyi_tier="vip", apiyi_model=payload["model"])
                    return result, request_id
            comfy.model_management.throw_exception_if_processing_interrupted()
            non_vip_payload = {name: value for name, value in payload.items() if name != "response_format"}
            non_vip_payload["model"] = OFFICIAL_MODELS[payload["model"]]
            non_vip_files = files
            if files and sum(field == "image" for field, _ in files) > 1:
                non_vip_files = [("image[]" if field == "image" else field, upload) for field, upload in files]
            nodes.LOGGER.warning("API易GPT备用：VIP失败，尝试非VIP模型 %s（仅1次）。", non_vip_payload["model"])
            attempt = {"number": 5, "tier": "non_vip", "model": non_vip_payload["model"]}
            try:
                result, request_id = nodes.request_image(single, non_vip_payload, non_vip_files)
            except RuntimeError as exc:
                attempt["error"] = str(exc)
                attempts.append(attempt)
            else:
                attempt["succeeded"] = True
                attempts.append(attempt)
                result.update(provider="apiyi", attempts=attempts, apiyi_tier="non_vip", apiyi_model=non_vip_payload["model"])
                return result, request_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        nodes.LOGGER.warning("API易GPT备用：切换 ComfyUI 官方节点。")
        try:
            result = cls.official_image(payload, files, seed)
        except Exception as exc:
            if isinstance(exc, comfy.model_management.InterruptProcessingException):
                raise
            reason = "；".join(f"{item['model']}：{item['error']}" for item in attempts[-2:]) if attempts else "未配置 APIYI_API_KEY"
            raise RuntimeError(f"API易失败：{reason}\nComfyUI官方备用失败：{exc}") from None
        result.update(provider="official", attempts=attempts, official_model=OFFICIAL_MODELS[payload["model"]])
        return result, None

    @classmethod
    def official_image(cls, payload, files, seed):
        token = cls.hidden.auth_token_comfy_org or ""
        key = cls.hidden.api_key_comfy_org or os.environ.get("COMFY_API_KEY", "")
        if not token and not key:
            raise RuntimeError("请先登录ComfyUI账户或设置COMFY_API_KEY。")
        hidden = {"AUTH_TOKEN_COMFY_ORG": token, "API_KEY_COMFY_ORG": key,
                  "UNIQUE_ID": cls.hidden.unique_id, "COMFY_USAGE_SOURCE": "comfyui-frontend"}
        official = OpenAIGPTImageNodeV2.PREPARE_CLASS_CLONE({"hidden_inputs": hidden})
        model = {"model": OFFICIAL_MODELS[payload["model"]], "size": payload["size"],
                 "quality": payload["quality"], "background": payload["background"]}
        if payload["size"] != "auto":
            width, height = map(int, payload["size"].split("x"))
            model.update(size="Custom", custom_width=width, custom_height=height)
        images = {}
        for field, (_, raw, _) in files or []:
            with nodes.Image.open(nodes.io.BytesIO(raw)) as image:
                if field == "mask":
                    alpha = nodes.np.asarray(image.getchannel("A"), dtype=nodes.np.float32) / 255
                    model["mask"] = nodes.torch.from_numpy(1 - alpha).unsqueeze(0)
                    continue
                pixels = nodes.np.asarray(image.convert("RGBA" if "A" in image.getbands() else "RGB"), dtype=nodes.np.float32) / 255
            images[f"image_{len(images) + 1}"] = nodes.torch.from_numpy(pixels.copy()).unsqueeze(0)
        model["images"] = images
        output = asyncio.run(official.execute(prompt=payload["prompt"], model=model, n=1, seed=seed))
        batch = output[0]
        if batch.shape[0] != 1:
            raise RuntimeError("官方备用返回数量不是1，停止以免输出错位。")
        return {"data": [{"b64_json": base64.b64encode(nodes.image_bytes(batch[0])).decode()}]}
