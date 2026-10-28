from functools import lru_cache
import importlib.util
import io
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

import folder_paths
import comfy.model_management
from comfy_api.latest import io as IO

from .engine import WorkerLock, create_job, job_dir, sha, snapshot, start_worker, summary, validate_apiyi_url, write_json


def store():
    return Path(folder_paths.get_output_directory()) / "ConcurrentImages"


@lru_cache(maxsize=1)
def apiyi_settings_reader():
    for directory in folder_paths.get_folder_paths("custom_nodes"):
        path = Path(directory) / "comfyui-apiyi-gpt/nodes.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_concurrent_local_apiyi", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.settings
    raise RuntimeError("未找到本地 comfyui-apiyi-gpt；请先安装本地API易节点。")


def local_apiyi_settings():
    return apiyi_settings_reader()()


def encode_images(batches):
    if batches is None:
        return []
    if isinstance(batches, torch.Tensor):
        batches = [batches]
    raw_images = []
    for batch in batches:
        if batch is None:
            continue
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        for tensor in batch:
            pixels = (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            raw_images.append(buffer.getvalue())
    return raw_images


def credentials(cls):
    return {"CONCURRENT_COMFY_TOKEN": cls.hidden.auth_token_comfy_org or "",
            "CONCURRENT_COMFY_KEY": cls.hidden.api_key_comfy_org or os.environ.get("COMFY_API_KEY", "")}


class ConcurrentImageSubmit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        config = local_apiyi_settings()
        models = list(config["models"])
        default_model = config["default_model"]
        return IO.Schema(node_id="ConcurrentImageSubmit", display_name="分块并发提交", category="API易/并发",
            is_input_list=True, is_output_node=True, is_api_node=True,
            description="输入整个IMAGE批次，每块独立并发生成。固定种子复用原任务；换种子新建。后台任务继续运行，取消ComfyUI等待不会取消已付费生成。",
            inputs=[IO.Image.Input("images", display_name="分块图像"),
                IO.String.Input("prompt", display_name="提示词", multiline=True, default=""),
                IO.Combo.Input("route", display_name="接口路线", options=["API易→官方备用", "仅官方节点"], default="API易→官方备用"),
                IO.Combo.Input("primary_model", display_name="API易模型", options=models, default=default_model),
                IO.Combo.Input("fallback_model", display_name="官方模型", options=["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2"], default="gpt-image-2.5-flare"),
                IO.Combo.Input("quality", display_name="质量", options=["low", "medium", "high", "xhigh", "max"], default="low"),
                IO.Int.Input("retries", display_name="每接口重试次数", default=2, min=0, max=10,
                    tooltip="首次调用之外的重试数；官方节点内部另有原生网络重试。断线后重试API易可能再次扣费。"),
                IO.Int.Input("seed", display_name="任务种子", default=0, min=0, max=2147483647,
                    tooltip="同图同参数同种子自动续取，不重发成功块；换种子才新建整批。"),
                IO.Autogrow.Input("references", display_name="逐块参考图", optional=True,
                    template=IO.Autogrow.TemplateNames(IO.Image.Input("image"), names=[f"参考图_{i}" for i in range(1, 101)], min=0),
                    tooltip="每路接参考图分块批次，张数必须等于待处理分块数；按相同编号一一配对。每次请求图1为当前分块，图2起为各路同编号参考图。")],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            outputs=[IO.String.Output(display_name="任务编号"), IO.String.Output(display_name="状态")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, images, references=None, **kwargs):
        options = {key: value[0] for key, value in kwargs.items()}
        config = local_apiyi_settings()
        options.update(primary_url=config["base_url"], request_timeout=900,
                       response_format=config["response_format"])
        if not options["prompt"].strip():
            raise ValueError("请填写提示词。")
        validate_apiyi_url(options["primary_url"])
        if options["route"] != "仅官方节点" and options["primary_model"] not in config["models"]:
            raise ValueError("请选择本地API易配置列表中的模型。")
        if options["fallback_model"] == "gpt-image-2" and options["quality"] in ("xhigh", "max"):
            raise ValueError("官方gpt-image-2不支持xhigh/max，请选2.5模型或降低质量。")
        options.update(comfy_root=str(Path(folder_paths.__file__).parent), retry_delay=5, download_retries=12)
        raw_images = encode_images(images)
        options["concurrency"] = len(raw_images)
        reference_groups = []
        for name in sorted(references or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            group = encode_images(references[name])
            if not group:
                continue
            if len(group) != len(raw_images):
                raise ValueError(f"{name}有{len(group)}张，但分块有{len(raw_images)}张；必须同数量按编号一一配对，未提交生成。")
            reference_groups.append(group)
        task_id = create_job(store(), raw_images, options, reference_groups)
        directory = job_dir(store(), task_id)
        start_worker(directory, credentials(cls))
        status = summary(directory)
        print("[分块并发] " + status)
        return IO.NodeOutput(task_id, status, ui={"text": [task_id, status]})


class ConcurrentImageCollect(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id="ConcurrentImageCollect", display_name="分块并发取回（按序）", category="API易/并发",
            is_output_node=True, is_api_node=True,
            description="按发送编号输出完整IMAGE批次。失败块不跳过；重启后输入原任务编号续取。任务文件在output/ConcurrentImages。",
            inputs=[IO.String.Input("task_id", display_name="任务编号", default=""),
                IO.Int.Input("wait_seconds", display_name="最长等待秒", default=3600, min=1, max=86400),
                IO.Combo.Input("size_policy", display_name="尺寸处理", options=["严格原块尺寸", "缩放回原块尺寸"], default="严格原块尺寸"),
                IO.Boolean.Input("retry_failed", display_name="重试失败块", default=False,
                    tooltip="明确开启后仅重试失败块，保留成功块。每次执行重新给予失败块重试额度，可能再次计费。")],
            hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
            outputs=[IO.Image.Output(display_name="按序分块批次"), IO.String.Output(display_name="状态")])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, task_id, wait_seconds=3600, size_policy="严格原块尺寸", retry_failed=False):
        directory = job_dir(store(), task_id.strip())
        manifest, states = snapshot(directory)
        if retry_failed:
            with WorkerLock(directory) as acquired:
                if not acquired:
                    raise RuntimeError("后台仍在运行，请等待后再重试失败块。")
                for item in states:
                    if item["state"] == "failed":
                        item.setdefault("previous_attempts", []).extend(item["attempts"])
                        item.update(state="pending", attempts=[], error=None)
                        # Keep attempt response filenames unique across manual retry cycles.
                        item["cycle"] = item.get("cycle", 0) + 1
                        write_json(directory / f"item-{item['index']:06d}.json", item)
        start_worker(directory, credentials(cls))
        deadline = time.monotonic() + wait_seconds
        next_restart_check = time.monotonic() + 10
        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            manifest, states = snapshot(directory)
            failed = [item for item in states if item["state"] == "failed"]
            active = [item for item in states if item["state"] not in ("succeeded", "failed")]
            if not active:
                if failed:
                    details = "; ".join(f"第{x['index']+1}块: {x['error']}" for x in failed)
                    raise RuntimeError(f"分块未齐全，停止拼接。{details}\n任务编号: {task_id}\n{directory}")
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"等待结束，后台任务和已完成结果保留。用原编号继续取回：{task_id}\n{summary(directory)}")
            if time.monotonic() >= next_restart_check:
                start_worker(directory, credentials(cls))
                next_restart_check = time.monotonic() + 10
            time.sleep(0.5)
        tensors, dimensions = [], []
        for index, (source, item) in enumerate(zip(manifest["inputs"], states)):
            if source["index"] != index or item["index"] != index:
                raise RuntimeError("编号校验失败，停止拼接。")
            raw = (directory / f"result-{index:06d}.png").read_bytes()
            if sha(raw) != item["result_sha256"]:
                raise RuntimeError(f"第{index+1}块文件校验失败，停止拼接。")
            with Image.open(io.BytesIO(raw)) as image:
                actual = list(image.size)
                if actual != source["size"]:
                    if size_policy == "严格原块尺寸":
                        raise RuntimeError(f"第{index+1}块尺寸{actual}与原块{source['size']}不同；可选择缩放回原块尺寸。")
                    image = image.resize(source["size"], Image.Resampling.LANCZOS)
                image = image.convert("RGBA" if "A" in image.getbands() or source["mode"] == "RGBA" else "RGB")
                tensors.append(torch.from_numpy(np.array(image, dtype=np.float32) / 255).unsqueeze(0))
                dimensions.append({"index": index, "returned": actual, "output": list(image.size)})
        if len({tuple(t.shape[1:3]) for t in tensors}) != 1:
            raise RuntimeError("原始分块尺寸不统一，不能组成IMAGE批次。请逐张源图分块提交。")
        if any(t.shape[-1] == 4 for t in tensors):
            tensors = [torch.cat((t, torch.ones_like(t[..., :1])), dim=-1) if t.shape[-1] == 3 else t for t in tensors]
        status = summary(directory)
        return IO.NodeOutput(torch.cat(tensors, dim=0), status, ui={"text": [status]})
