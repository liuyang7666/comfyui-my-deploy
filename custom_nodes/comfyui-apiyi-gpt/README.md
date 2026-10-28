# API易兼容入口

仅保留 `nodes.py` 和 `config.json` 供并发模块读取配置，不注册节点。实际 API易节点由 MY 包提供。完整旧版源码在 `archives/standalone/`，不要与 MY 同时启用。
