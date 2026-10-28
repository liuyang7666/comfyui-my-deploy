# 来源与许可说明

本仓库是个人 ComfyUI 部署快照，包含自行编写、修改及整合的代码，不将全部文件声明为单一原创项目。

- 像素漂移对齐基于 [Mozer/ComfyUI-PixelDriftFix](https://github.com/Mozer/ComfyUI-PixelDriftFix)，来源提交 `62b79b86016e3914e62e258624646afa6cd8d394`；保留上游 README。已保留目录未附独立许可证文件，不能据本仓库推定第三方代码具有额外开源授权。
- LaMa 适配来自 [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) 3.2.0；GPL-3.0 许可在 `custom_nodes/comfyui-my-nodes/lama_remover/LICENSE`。
- 照片节点基于 [thezveroboy/ComfyUI-zveroboy-photo](https://github.com/thezveroboy/ComfyUI-zveroboy-photo)，上游提交 `62d7de1428c271fe8a0c33867d910c1aef6d1fde`。
- TBG-SAM3 补丁基于 [Ltamann/ComfyUI-TBG-SAM3](https://github.com/Ltamann/ComfyUI-TBG-SAM3) 1.1.1；保留对应目录中的上游许可文件。
- 其余第三方依赖仅登记来源与版本，不将整个依赖项目重新上传。

各文件保留原有来源与许可约束；未附许可证的部分不自动成为 MIT/Apache/GPL。模型权重另按原发布方条件取得。源码文件的原始及打包 SHA256 见 `manifests/source-provenance.json`。
