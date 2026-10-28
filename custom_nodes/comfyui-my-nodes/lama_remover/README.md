# MY LaMa 路人移除节点

基于 AILab-AI/ComfyUI-RMBG v3.2.0 的 `AILab_LamaRemover`，集成到 MY 节点包并使用独立 ID `MYLamaRemover`，避免与原 RMBG 插件节点冲突。

节点名：**移除路人（LaMa，遮罩驱动）**。需要连接待处理图像和遮罩；白色遮罩区域表示移除，黑色区域保留。节点本身不检测或自动选择路人。

继续使用当前 ComfyUI 的 `models/RMBG/Lama/big-lama.pt` 模型文件；文件缺失时从 `1038lab/Lama` 下载到该目录。没有将大模型复制进 MY 包。模型通过 Python 二进制文件流加载，以兼容包含中文字符的 Windows 安装路径。

上游来源：<https://github.com/1038lab/ComfyUI-RMBG>。本机集成来源为 v3.2.0 的 `py/AILab_LamaRemover.py`；该安装目录没有独立 Git 元数据，因此不标注具体上游 commit。集成时保留了此本机源码中的 Windows 中文路径文件流加载修复。来源文件 SHA-256：`9ada0c8eeff4a05fff3695378b9cf8698049317878e23cc890af6c07eb75dcd8`。本目录保留上游 GPL-3.0 许可证。
