# API易GPT

## API易多模态聊天（识图）

新增节点 `APIYiVisionChat`，分类 `MY/API易`。使用 API易的 OpenAI 兼容 `/chat/completions` 接口，支持提示词、可选系统提示、单张或批次 IMAGE 输入，并输出回答和输入/输出 token 用量。

模型列表包含七款视觉模型：

- `gemini-3.8-flash`：默认推荐，日常图像问答均衡。
- `gemini-3.5-flash-lite`：响应快，适合大量低延迟识图。
- `gpt-6.1-sol`：复杂图像分析，能力较强，单价高于轻量模型。
- `gpt-6.1-terra`：视觉推理与性价比兼顾；API易侧可用型号以账号实时模型列表为准。
- `gpt-6-luna`：OpenAI 当前 Luna API ID，成本低，适合概述；细小文字 OCR 可能漏字。
- `deepseek-v4-flash-vision-exp`：图像输入 token 较少，适合控制成本。
- `qwen3.8-flash`：中文识图性价比高。

连接 IMAGE 后，批次中的每张图会作为同一条用户消息的独立 `image_url` 发送。图像统一转为 JPEG，长边缩放至 2048 像素，再 Base64 内嵌；单次最多 16 张。用量输出会显示模型、图像数、输入/输出 token；若图片请求的输入 token 明显偏少，会提示检查模型是否收到图片。该节点只做单次调用，不自动重试；连接中断或超时后请先查 API易调用记录再重新运行，避免重复计费。

密钥继续使用当前用户环境变量 `APIYI_API_KEY`，地址沿用 `APIYI_BASE_URL` 覆盖配置，默认 `https://api.apiyi.com/v1`。

接口参考：[API易图像理解（识图）API](https://docs.apiyi.com/api-capabilities/vision-understanding)、[文本生成（对话补全）](https://docs.apiyi.com/api-capabilities/text-generation)。

**API易GPT（官方备用）**（搜索 `APIYiGPTFallback`，分类仍为 `API易/图像`）：每张先调用 API易 VIP，失败后分别等待 5/10/20 秒重试 **3 次**，即最多 4 次 VIP 请求；仍失败时使用同一 API易地址与密钥，调用对应**非 VIP 模型 1 次**；再次失败才执行本机官方 `OpenAIGPTImageNodeV2`。API易两阶段均关闭内部429重试，最多4次VIP加1次非VIP。普通 `API易GPT` 保持原来的重试策略。

非VIP和ComfyUI官方阶段的模型映射相同：默认 VIP 和 flare-vip 对应 `gpt-image-2.5-flare`，sunburst-vip 对应 `gpt-image-2.5-sunburst`，2-vip 对应 `gpt-image-2`。提示词、质量、背景、请求尺寸、参考图与蒙版内容及顺序均沿用本次输入。非VIP按[API易官转格式](https://docs.apiyi.com/api-capabilities/gpt-image-2/overview)移除不支持的 `response_format`，多图使用 `image[]` 文件字段。ComfyUI官方备用需要登录账户（或设置 `COMFY_API_KEY`），使用其账户积分；自定义尺寸按官方校验，不自动改尺寸。官方自身的网络重试沿用原实现。

数量大于 1 时逐张独立重试和切换，已完成的图片保留；取消任务会停止后续重试及切换。API易密钥缺失时直接使用ComfyUI官方。结果 JSON 的 `request` 保留原VIP输入，`apiyi_model` / `apiyi_tier` 记录实际成功的API易模型与线路，`attempts` 逐次记录模型、线路与结果；面板显示VIP→非VIP→官方路线。非VIP按自己的 request_id 查询API易实扣，不使用VIP的$0.03估算；ComfyUI官方结果不查询API易账单。断线后原请求可能已经扣费，重试与切换可能增加费用，API易显示的单次实扣不含无法查询的失败请求。已生成但下载失败时保留现有记录，不因此重新生图。

安装更新后重启 ComfyUI 后端并刷新页面。可加载 `API易GPT-重试3次-官方备用.json` 示例工作流。

v1.0.1 修复官方备用在 ComfyUI 异步调度中报事件循环冲突的问题；现有工作流和参数可继续使用。

v1.1 两个节点均增加可选 **蒙版（MASK）** 接口：白色为编辑区、黑色为保留区，灰度过渡保留。蒙版须与第一张输入图像同尺寸，接入单张蒙版；不接蒙版时保持原用法。上传压缩缩小目标图时，蒙版按相同尺寸同步缩放，不膨胀、不额外羽化；作为独立透明 PNG 的 `mask` 文件上传。蒙版在重试与官方切换中沿用，不会当作另一张参考图。**官方节点的蒙版编辑仅支持单张输入图像**，请勿同时接入多张参考图。API易 VIP 仍可能改动蒙版外区域，接入蒙版不等于精确像素保护，也不自动将生成图按蒙版贴回原图。

ComfyUI 自定义节点，分类 `API易/图像`，搜索 `API易GPT`。不接图为文生图，接入图像为图片编辑/多图融合。默认 `gpt-image-2.5-vip`，可切换 flare-vip、sunburst-vip、2-vip。

## 使用

1. 设置当前用户环境变量 `APIYI_API_KEY`（可运行 `配置密钥.ps1`）。完全退出并重新打开绘世启动器和 ComfyUI，使其继承环境变量。
2. 添加 `API易GPT`，填写提示词、模型、尺寸、背景、质量、数量、种子。结果在节点中预览，同时自动保存为 `ComfyUI/output/APIYI-GPT/*.png`，相邻 JSON 记录参数、耗时和 token usage，不含密钥。可以右键预览图保存，也可连接其他 IMAGE 节点。
3. 图片编辑：连接一个或多个 IMAGE 输入。每个批次中的图片都按顺序作为参考图，不会给每张参考图单独发起编辑。输出为 IMAGE 列表，以保留不同返回尺寸和透明通道。
4. 费用与余额：设置 `APIYI_ACCOUNT_TOKEN` 为 `nap_` 查询令牌，同时勾选「个人资料·读取」和「用量与个人记录·读取」。节点完成每次调用后按 request_id 查实际消费日志，再查账户余额，面板显示每次实扣、余额和本批合计，JSON 同步保存 `billing`。查询失败或权限不足只显示状态，不影响已生成图片。余额是查询时的账户余额，不是单个模型 key 的额度；命中缓存时显示的是该次结果原有账单。

尺寸支持 `自动 + 官方 30 档 + 自定义`。选「自定义」后，填写下方「自定义宽度 / 自定义高度」（默认 1024×1024），单位为像素，必须是正整数。建议宽高均为 16 的倍数，例如 1024×1024、1920×1088、2048×1152。节点按输入原样传给 API，不自动对齐或缩放生成结果；结果面板同时显示请求尺寸与实际尺寸，有差异时标注「接口调整」，JSON 保存 `size_adjusted`。

VIP 文档实测：非 16 倍数可能被上游对齐，例如 1920×1080 → 1920×1088；过小会被抬高，例如 512×512 → 816×816。这些是示例，不是完整的上下限规则；文档未承诺任意自定义尺寸的硬性最大值或比例范围。要稳定锁定像素尺寸，优先用 30 档预设（已覆盖到 3840×2160 等 4K 尺寸）。预设/自动模式不会读取自定义宽高，旧工作流可继续使用。

支持上传遮罩，但 VIP 的 mask 可能整图改动；精确遮罩需求应使用具备该能力的官方接口并验收。背景透明保留 alpha，但上游不保证每次兑现。

`config.json` 配置地址、默认模型、模型列表、超时、返回格式、饱和重试次数。`APIYI_BASE_URL` 可覆盖地址。默认显式发送 `response_format=b64_json`；可改为 `url`，服务端立即下载保存。解析同时兼容纯 base64、带 data 前缀和 URL；下载不携带密钥。API key 只从环境变量读取。

## 调用与费用

- timeout 最低 360 秒，不自动重试超时、断连、5xx，避免同步接口重复扣费。只有明确的 429 上游饱和才进行最多两次退避重试；参数拒绝直接显示错误。直连 API易，无新增反向代理、网关或 Serverless 层；部署在已有中间层之后时，其请求/读取超时也需至少 360 秒。
- 数量 1–8 对应独立的 1–8 次请求；不向 API 发送 `n`、`aspect_ratio`、`seed`、`mask`。种子仅用于 ComfyUI 缓存失效与重新运行，不保证图片复现。每张完成后立即落盘，后续失败时保留已完成结果。
- VIP 文档标价 $0.03/张，所有尺寸/质量同价。JSON 中 `price_usd_estimate` 是标价估算；`billing.cost_usd` 来自 request_id 精确查询的消费日志实扣，`billing.balance_usd` 来自账户余额接口。原始日志可能包含密钥，因此只保存必要的费用字段。
- 2.5 质量：低/中/高/超高/最高；2-vip：低/中/高。2.5 的高约等于 2-vip 的中，2.5 的最高约等于 2-vip 的高。
- 上传格式 PNG/JPEG/WebP；超过 1.5 MiB 的图长边最多 2048，质量 90、同格式重编码；必要时继续等比缩小。多图目标合计不超过 6 MiB、单图不超过 10 MiB。压缩异常回退该张原图继续，并记录警告，因此异常回退可能超出建议预算。ComfyUI IMAGE 张量不保留原文件编码格式，进入此节点时先编码 PNG；压缩层保留其输入格式及 alpha。

原始官方 Markdown 已保存到 `docs/`。来源：[概览](https://docs.apiyi.com/api-capabilities/gpt-image-2-vip/overview.md)、[文生图](https://docs.apiyi.com/api-capabilities/gpt-image-2-vip/text-to-image.md)、[图片编辑](https://docs.apiyi.com/api-capabilities/gpt-image-2-vip/image-edit.md)。
