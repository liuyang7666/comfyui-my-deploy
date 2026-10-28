# 我的 ComfyUI 节点与工作流部署包

用于在新机器恢复个人 ComfyUI 节点和已保存的工作流。中文说明，公开仓库，快照日期：**2026-10-28**。

包含 **8 个部署目录、17 份工作流/子图**，以及独立旧版源码、第三方版本清单和少量本地修复快照。以当前安装目录为来源；没有修改正在使用的 ComfyUI。

## 快速部署

1. 先安装适合新机器的 ComfyUI 和 PyTorch/CUDA 环境。
2. 下载本仓库，或执行：

```powershell
git clone https://github.com/liuyang7666/comfyui-my-deploy.git
cd comfyui-my-deploy
```

3. 关闭目标 ComfyUI，使用**该实例的 Python**运行部署脚本。把示例路径改成自己的安装目录：

```powershell
# 先预览，不写入目标目录
python scripts/deploy.py --comfyui "D:/ComfyUI" --dry-run

# 复制节点、保存的流程和子图，覆盖前自动备份
python scripts/deploy.py --comfyui "D:/ComfyUI"

# 若缺少附加依赖，再用 ComfyUI 实际使用的 Python 执行
python -m pip install -r requirements-local.txt
```

Windows 也可使用 `部署.ps1 -ComfyUI "D:/ComfyUI" -Python "D:/ComfyUI/.venv/Scripts/python.exe"`。便携版通常使用自己的 `python_embeded/python.exe`；Desktop 请以实例配置中的解释器为准。Python 脚本同时适用于 Linux，Windows 原生选图功能除外。

4. 按[第三方依赖](docs/第三方依赖.md)安装工作流需要的插件，并按[模型与密钥](docs/模型与密钥.md)准备权重、ExifTool 和新机器密钥。
5. 重启 ComfyUI 并刷新前端，在工作流菜单的「我的部署包」中打开流程。重新选择输入图片/目录，先运行一张测试图。

## 目录

| 目录 | 内容 |
|---|---|
| `custom_nodes/` | 默认部署的本地节点；包括不注册节点的 API易兼容入口 |
| `workflows/saved/` | 已保存的常用流程 |
| `workflows/examples/` | 当前节点示例，含尚未视觉验收的实验流程 |
| `subgraphs/` | 可复用子图，复制到用户子图库 |
| `workflows/archive/` | 旧版流程，仅归档，默认不部署 |
| `archives/standalone/` | 已整合进 MY 的独立旧版源码，仅供对照 |
| `overlays/` | 已安装第三方局部修复，仅供同版本核对合入 |
| `manifests/` | 来源 SHA256、环境、节点版本、工作流依赖与文件校验 |
| `scripts/` | 部署、回滚、完整性和运行时检查 |

## 更新和回滚

更新仓库后重新运行部署脚本。目标同名目录先整体移入 `ComfyUI/deploy-backups/时间戳/`，避免旧文件残留。其他第三方节点和模型保留原位置。

部署 MY 时，已有的独立 `comfyui-qwen-api`、`image25-tile` 目录会一并移入备份，避免重复注册。旧 `comfyui-apiyi-gpt` 会备份后替换为配置兼容入口；API易实际节点仍由 MY 提供。

```powershell
python scripts/deploy.py --comfyui "D:/ComfyUI" --restore "D:/ComfyUI/deploy-backups/实际时间戳"
```

回滚会把部署后的文件另存于备份目录，不直接删除。请在停止 ComfyUI 后操作。

## 检查与说明

```powershell
python scripts/validate.py
python scripts/check_runtime.py --url http://127.0.0.1:8188
```

详细阅读[节点清单](docs/节点清单.md)、[工作流目录](docs/工作流目录.md)、[已知限制](docs/已知限制.md)和[来源与许可](THIRD_PARTY_NOTICES.md)。本次只检查打包与部署结构，未重新验证付费 API、照片生成或成片质量。尚未保存的桌面标签页不在仓库内。

**不包含 API 密钥、账号凭据、照片、模型权重、数据库或生成历史。** 发布前会检查上传文件；后续更新也请保持这一边界。
