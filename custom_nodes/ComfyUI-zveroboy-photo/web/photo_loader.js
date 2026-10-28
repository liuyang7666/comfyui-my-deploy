import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "zveroboy.photo.filePicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PhotoLoadRaw") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            let choosing = false;
            const choose = async (kind) => {
                if (choosing) return;
                choosing = true;
                try {
                    const response = await api.fetchApi("/zveroboy-photo/picker", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ kind }),
                    });
                    const selection = await response.json();
                    if (!response.ok) throw new Error(selection.error || "无法打开选择窗口。");
                    if (!selection.paths?.length) return;
                    const pathWidget = this.widgets.find(widget => widget.name === "raw_file");
                    if (!pathWidget) throw new Error("未找到照片路径输入框，请刷新页面后重新添加节点。");
                    pathWidget.value = selection.paths.length === 1 ? selection.paths[0] : JSON.stringify(selection.paths);
                    pathWidget.callback?.(pathWidget.value);
                    this.setDirtyCanvas(true, true);
                    app.graph?.change();
                } catch (error) {
                    app.extensionManager.toast.add({ severity: "error", summary: "照片选择失败", detail: error.message, life: 7000 });
                } finally {
                    choosing = false;
                }
            };
            this.addWidget("button", "Choose photos", null, () => choose("file"), { serialize: false });
            this.addWidget("button", "Choose folder", null, () => choose("folder"), { serialize: false });
            const preview = ComfyWidgets.STRING(this, "exif_preview", ["STRING", { multiline: true }], app).widget;
            preview.value = "运行后显示所选照片的EXIF信息。";
            preview.options.serialize = false;
            preview.inputEl.readOnly = true;
            this.setSize(this.computeSize());
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            if (Array.isArray(message.exif_text)) {
                const preview = this.widgets.find(widget => widget.name === "exif_preview");
                if (preview) preview.value = message.exif_text.join("\n\n────────────\n\n");
                this.setDirtyCanvas(true, true);
            }
            return result;
        };
    },
});
