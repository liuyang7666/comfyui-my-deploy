"""Durable, indexed jobs. No ComfyUI imports: usable by an independent worker."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageOps

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def atomic_write(path, raw):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def job_dir(store, task_id):
    if not re.fullmatch(r"[a-f0-9]{32}", task_id):
        raise ValueError("任务编号必须是32位十六进制字符。")
    return Path(store) / task_id


class WorkerLock:
    """OS lock, automatically released even if the worker is killed."""
    def __init__(self, directory):
        self.path = Path(directory) / "worker.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            self.stream = None
        return self.stream is not None

    def __exit__(self, *_):
        if self.stream:
            self.stream.close()


def create_job(store, raw_images, options, reference_groups=None):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    inputs = []
    for index, raw in enumerate(raw_images):
        with Image.open(io.BytesIO(raw)) as image:
            inputs.append({"index": index, "sha256": sha(raw), "size": list(image.size),
                           "mode": "RGBA" if "A" in image.getbands() else "RGB"})
    if not inputs:
        raise ValueError("没有收到分块图像。")
    manifest = {"schema": 1, "options": options, "inputs": inputs}
    if reference_groups:
        records = []
        for group in reference_groups:
            if len(group) != len(inputs):
                raise ValueError("参考图列表必须与分块数量完全一致，未提交。")
            records.append([{"index": index, "sha256": sha(raw)} for index, raw in enumerate(group)])
        manifest["reference_groups"] = records
    task_id = sha(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())[:32]
    directory = job_dir(store, task_id)
    if directory.exists():
        return task_id
    staging = store / (".creating-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        write_json(staging / "job.json", dict(manifest, task_id=task_id, created=time.time()))
        for item, raw in zip(inputs, raw_images):
            atomic_write(staging / f"input-{item['index']:06d}.png", raw)
            write_json(staging / f"item-{item['index']:06d}.json", {
                "index": item["index"], "state": "pending", "attempts": [], "error": None})
        for group_index, group in enumerate(reference_groups or []):
            for index, raw in enumerate(group):
                atomic_write(staging / f"reference-{group_index:03d}-{index:06d}.png", raw)
        try:
            staging.rename(directory)
        except OSError:
            if not directory.exists():
                raise
    finally:
        # Only remove this invocation's private staging directory.
        if staging.exists() and staging.resolve().parent == store.resolve():
            shutil.rmtree(staging)
    return task_id


def snapshot(directory):
    manifest = read_json(Path(directory) / "job.json")
    states = [read_json(Path(directory) / f"item-{i:06d}.json") for i in range(len(manifest["inputs"]))]
    return manifest, states


def summary(directory):
    manifest, states = snapshot(directory)
    counts = {}
    for item in states:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    return json.dumps({"task_id": manifest["task_id"], "total": len(states), "states": counts,
                       "directory": str(directory)}, ensure_ascii=False)


def start_worker(directory, credentials=None):
    manifest, states = snapshot(directory)
    if all(item["state"] in ("succeeded", "failed") for item in states):
        return
    with WorkerLock(directory) as acquired:
        if not acquired:
            return
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (credentials or {}).items() if v})
    env["PYTHONUTF8"] = "1"
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS} if os.name == "nt" else {"start_new_session": True}
    # Credentials are inherited in memory, never placed in argv, workflow or job files.
    with (Path(directory) / "worker.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("worker.py")), str(Path(directory).resolve())],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                         cwd=str(Path(__file__).parent), close_fds=True, **kwargs)
    threading.Thread(target=process.wait, daemon=True).start()


def redact(message):
    text = str(message)
    for name in ("APIYI_API_KEY", "CONCURRENT_COMFY_TOKEN", "CONCURRENT_COMFY_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[密钥隐藏]")
    text = re.sub(r"sk-[\w-]+|Bearer\s+\S+", "[密钥隐藏]", text)
    return text[:1500]


class CallError(RuntimeError):
    def __init__(self, message, retryable=True, delay=0):
        super().__init__(message)
        self.retryable = retryable
        self.delay = delay


def validate_apiyi_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API易地址必须是无用户名/密码/查询参数的HTTPS地址。")
    return url.rstrip("/")


def apiyi_call(raw, options):
    key = os.environ.get("APIYI_API_KEY", "").strip()
    if not key:
        raise CallError("未配置 APIYI_API_KEY，切换官方备用。", retryable=False)
    payload = {"model": options["primary_model"], "prompt": options["prompt"],
               "quality": options["quality"], "background": "auto", "response_format": options.get("response_format", "b64_json")}
    with Image.open(io.BytesIO(raw)) as image:
        payload["size"] = f"{image.width}x{image.height}"
    try:
        files = [("image", ("tile.png", raw, "image/png"))]
        files.extend(("image", (f"reference_{i+1}.png", reference, "image/png"))
                     for i, reference in enumerate(options.get("_references", [])))
        response = requests.post(validate_apiyi_url(options["primary_url"]) + "/images/edits",
                                 headers={"Authorization": "Bearer " + key}, data=payload,
                                 files=files,
                                 timeout=(30, options["request_timeout"]), allow_redirects=False)
    except requests.RequestException as exc:
        raise CallError("API易连接中断；原结果无法查询，重试可能重复计费。" + type(exc).__name__) from None
    if response.status_code != 200:
        retryable = response.status_code in (408, 429, 500, 502, 503, 504)
        retry_after = response.headers.get("Retry-After", "")
        raise CallError(f"API易 HTTP {response.status_code}: {redact(response.text[:1500])}", retryable,
                        min(120, float(retry_after)) if retry_after.isdigit() else 0)
    try:
        result = response.json()
    except ValueError:
        raise CallError("API易返回非JSON；重试可能重复计费。") from None
    if not isinstance(result, dict) or not isinstance(result.get("data"), list) or len(result["data"]) != 1:
        raise CallError("API易没有返回恰好一张图片；重试可能重复计费。")
    item = result["data"][0]
    if not isinstance(item, dict) or not (item.get("b64_json") or item.get("url")):
        raise CallError("API易返回空图片结果，按失败重试。")
    return result


def download_result(body, timeout):
    item = body["data"][0]
    if item.get("b64_json"):
        value = item["b64_json"]
        return base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
    url = item.get("url", "")
    if urlsplit(url).scheme != "https":
        raise ValueError("图片结果既无base64也无HTTPS链接。")
    # Never send API credentials to the image CDN.
    response = requests.get(url, timeout=(30, timeout))
    response.raise_for_status()
    return response.content


def normalize_result(raw):
    with Image.open(io.BytesIO(raw)) as original:
        image = ImageOps.exif_transpose(original)
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        output = io.BytesIO()
        image.save(output, format="PNG")
    return output.getvalue()


def process_item(directory, index, official_call, sleep=time.sleep):
    directory = Path(directory)
    manifest = read_json(directory / "job.json")
    options = manifest["options"]
    status_path = directory / f"item-{index:06d}.json"
    item = read_json(status_path)
    if item["state"] in ("succeeded", "failed"):
        return
    raw = (directory / f"input-{index:06d}.png").read_bytes()
    if sha(raw) != manifest["inputs"][index]["sha256"]:
        item.update(state="failed", error="输入校验失败，未提交。")
        write_json(status_path, item)
        return
    references = []
    for group_index, group in enumerate(manifest.get("reference_groups", [])):
        reference = (directory / f"reference-{group_index:03d}-{index:06d}.png").read_bytes()
        if len(group) != len(manifest["inputs"]) or group[index]["index"] != index or sha(reference) != group[index]["sha256"]:
            item.update(state="failed", error="参考图编号/文件校验失败，未提交。")
            write_json(status_path, item)
            return
        references.append(reference)
    if references:
        options = dict(options, _references=references)
    result_path = directory / f"result-{index:06d}.png"
    # Recover the atomic result/response that was saved just before a process crash.
    if item["state"] == "running":
        if result_path.exists():
            normalized = normalize_result(result_path.read_bytes())
            item.update(state="succeeded", result_sha256=sha(normalized), error=None)
            write_json(status_path, item)
            return
        response_path = directory / item["attempts"][-1]["response_file"]
        if response_path.exists():
            item.update(state="response", response_file=response_path.name)
        else:
            item["attempts"][-1]["error"] = "进程中断，调用结果未知；原接口不提供事后取回。"
            item["state"] = "pending"
        write_json(status_path, item)

    providers = ["apiyi", "official"] if options["route"] == "API易→官方备用" else ["official"]
    while item["state"] == "pending":
        provider = None
        for candidate in providers:
            attempts = [a for a in item["attempts"] if a["provider"] == candidate]
            if len(attempts) < options["retries"] + 1 and not any(a.get("terminal") for a in attempts):
                provider = candidate
                break
        if provider is None:
            item["state"] = "failed"
            write_json(status_path, item)
            return
        number = len(item["attempts"]) + 1
        attempt = {"provider": provider, "number": number, "started": time.time(),
                   "response_file": f"response-{index:06d}-{item.get('cycle', 0):03d}-{number:03d}.json"}
        item["attempts"].append(attempt)
        item.update(state="running", error=None)
        write_json(status_path, item)  # Must commit before any paid request.
        try:
            if provider == "apiyi":
                body = apiyi_call(raw, options)
                write_json(directory / attempt["response_file"], body)
                item.update(state="response", response_file=attempt["response_file"])
            else:
                call_options = dict(options, _response_file=str(directory / attempt["response_file"]),
                                    _native_temp=str(directory / "official-temp"))
                generated = normalize_result(official_call(raw, call_options, index))
                atomic_write(result_path, generated)
                item.update(state="succeeded", result_sha256=sha(generated))
            attempt["finished"] = time.time()
            write_json(status_path, item)
        except Exception as exc:
            # This is a job boundary: preserve partial successes and a sanitized error per tile.
            attempt.update(error=redact(exc), finished=time.time(),
                           terminal=isinstance(exc, CallError) and not exc.retryable)
            if (directory / attempt["response_file"]).exists():
                item.update(state="response", response_file=attempt["response_file"], error=attempt["error"])
                write_json(status_path, item)
                break
            item.update(state="pending", error=attempt["error"])
            write_json(status_path, item)
            if not attempt["terminal"]:
                count = sum(a["provider"] == provider for a in item["attempts"])
                delay = min(60, options["retry_delay"] * 2 ** (count - 1))
                sleep(max(delay, exc.delay if isinstance(exc, CallError) else 0))

    if item["state"] == "response":
        # Download failures retain the original response; restarting retrieval never regenerates this tile.
        for download_attempt in range(options.get("download_retries", 12) + 1):
            try:
                saved = read_json(directory / item["response_file"])
                if item["attempts"][-1]["provider"] == "official":
                    call_options = dict(options, _saved_response=saved, _native_temp=str(directory / "official-temp"))
                    raw_result = official_call(raw, call_options, index)
                else:
                    raw_result = download_result(saved, options["request_timeout"])
                generated = normalize_result(raw_result)
                atomic_write(result_path, generated)
                item.update(state="succeeded", result_sha256=sha(generated), error=None)
                write_json(status_path, item)
                return
            except Exception as exc:
                item["error"] = "结果已保存，下载/解码未完成：" + redact(exc)
                write_json(status_path, item)
                expired = isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code in (404, 410)
                invalid = isinstance(exc, (ValueError, KeyError, TypeError)) or (isinstance(exc, OSError) and not isinstance(exc, requests.RequestException))
                if expired or invalid:
                    item.update(state="pending", error="已保存结果不可用，按剩余额度重新生成：" + redact(exc))
                    item["attempts"][-1]["error"] = item["error"]
                    write_json(status_path, item)
                    return process_item(directory, index, official_call, sleep)
                if download_attempt == options.get("download_retries", 12):
                    return
                sleep(min(60, options["retry_delay"] * 2 ** download_attempt))
