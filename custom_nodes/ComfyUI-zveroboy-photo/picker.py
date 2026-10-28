import asyncio
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import json
import sys

from aiohttp import web
from server import PromptServer


picker_lock = asyncio.Lock()


def select_paths(kind):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(Path(__file__).with_name("picker.ps1")), "-Kind", kind],
        capture_output=True, encoding="utf-8-sig", errors="replace", timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("无法打开本机选择窗口，请改为填写完整路径。")
    return json.loads(result.stdout)


@PromptServer.instance.routes.post("/zveroboy-photo/picker")
async def choose_photo_path(request):
    if request.remote not in ("127.0.0.1", "::1"):
        return web.json_response({"error": "本机选择窗口仅支持在此电脑上打开的ComfyUI。"}, status=403)
    origin = request.headers.get("Origin")
    if (origin and urlsplit(origin).netloc != request.host) or request.content_type != "application/json":
        return web.json_response({"error": "请选择节点上的文件选择按钮。"}, status=403)
    data = await request.json()
    if not isinstance(data, dict) or data.get("kind") not in ("file", "folder"):
        return web.json_response({"error": "不支持的选择类型。"}, status=400)
    if sys.platform != "win32":
        return web.json_response({"error": "本机选择按钮需要Windows；其他系统可填写照片或文件夹路径。"}, status=400)
    if picker_lock.locked():
        return web.json_response({"error": "已有选择窗口打开，请先完成或取消该窗口。"}, status=409)
    async with picker_lock:
        try:
            return web.json_response(await asyncio.to_thread(select_paths, data["kind"]))
        except (subprocess.TimeoutExpired, OSError, RuntimeError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=500)
