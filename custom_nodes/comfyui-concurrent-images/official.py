"""Execute the installed official node, in a headless worker process."""
import asyncio
import contextvars
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image

from engine import write_json

_load_lock = threading.Lock()
_node = None
_torch = None
_response_target = contextvars.ContextVar("official_response_target", default=None)


class ResponseJournal(logging.Handler):
    """Observe the official node's completed native log; do not replace its HTTP implementation."""
    def emit(self, record):
        target = _response_target.get()
        if not target or record.msg != "API log saved to: %s":
            return
        try:
            native = Path(record.args[0]).read_text(encoding="utf-8")
            if "Method: POST\n" not in native or "/proxy/openai/images/" not in native:
                return
            response = native.split(" RESPONSE ", 1)[1]
            if "\nStatus Code: 200\n" not in response or "\nContent:\n" not in response:
                return
            body, _ = json.JSONDecoder().raw_decode(response.split("\nContent:\n", 1)[1])
            if isinstance(body.get("data"), list) and len(body["data"]) == 1:
                write_json(target, body)
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            # Native logging is best-effort; a logging-format change cannot fail generation.
            return


class HeadlessProgress:
    def send_progress_text(self, *args, **kwargs):
        pass


def load_official(comfy_root):
    global _node, _torch
    with _load_lock:
        if _node is None:
            sys.path.insert(0, comfy_root)
            # The worker has no local model inference; keep it off the GPU.
            from comfy.cli_args import args
            args.cpu = True
            import torch
            from server import PromptServer
            from comfy_api_nodes.nodes_openai import OpenAIGPTImageNodeV2
            from comfy_api_nodes.util import request_logger
            if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
                PromptServer.instance = HeadlessProgress()
            _node, _torch = OpenAIGPTImageNodeV2, torch
            request_logger.logger.addHandler(ResponseJournal())
            request_logger.logger.setLevel(logging.DEBUG)
            request_logger.logger.propagate = False
    return _node


def generate(raw, options, index):
    token = os.environ.get("CONCURRENT_COMFY_TOKEN", "")
    key = os.environ.get("CONCURRENT_COMFY_KEY", "")
    if not token and not key:
        raise RuntimeError("官方备用需要登录ComfyUI账户或提供COMFY_API_KEY；重新登录后用原任务编号取回。")
    node = load_official(options["comfy_root"])
    if options.get("_native_temp"):
        import folder_paths
        folder_paths.set_temp_directory(options["_native_temp"])
    hidden = {"AUTH_TOKEN_COMFY_ORG": token, "API_KEY_COMFY_ORG": key,
              "UNIQUE_ID": str(index), "COMFY_USAGE_SOURCE": "comfyui-frontend"}
    clone = node.PREPARE_CLASS_CLONE({"hidden_inputs": hidden})
    with Image.open(io.BytesIO(raw)) as source:
        pixels = np.asarray(source.convert("RGBA" if "A" in source.getbands() else "RGB"), dtype=np.float32) / 255
        width, height = source.size
    tensor = _torch.from_numpy(pixels.copy()).unsqueeze(0)
    image_inputs = {"image_1": tensor}
    for ref_index, reference in enumerate(options.get("_references", []), 2):
        with Image.open(io.BytesIO(reference)) as source:
            ref_pixels = np.asarray(source.convert("RGBA" if "A" in source.getbands() else "RGB"), dtype=np.float32) / 255
        image_inputs[f"image_{ref_index}"] = _torch.from_numpy(ref_pixels.copy()).unsqueeze(0)
    model = {"model": options["fallback_model"], "size": "Custom", "custom_width": width,
             "custom_height": height, "quality": options["quality"], "background": "auto",
             "images": image_inputs}
    if options.get("_saved_response"):
        from comfy_api_nodes.apis.openai import OpenAIImageGenerationResponse
        from comfy_api_nodes.nodes_openai import validate_and_cast_response
        response = OpenAIImageGenerationResponse.model_validate(options["_saved_response"])
        value = asyncio.run(validate_and_cast_response(response, cls=clone))
    else:
        # n=1: one response belongs to exactly one immutable input index.
        context = _response_target.set(options.get("_response_file"))
        try:
            result = asyncio.run(clone.execute(prompt=options["prompt"], model=model, n=1, seed=options["seed"]))
            value = result[0]
        finally:
            _response_target.reset(context)
    if value.shape[0] != 1:
        raise RuntimeError("官方节点返回数量不是1，停止以防编号错位。")
    pixels = (value[0].detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()
