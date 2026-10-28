import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const pendingOrder = new WeakSet();

function orderMaskAfterImages(node) {
    if (pendingOrder.has(node)) return;
    pendingOrder.add(node);
    queueMicrotask(() => {
        pendingOrder.delete(node);
        const inputs = node.inputs ?? [];
        const maskIndex = inputs.findIndex(input => input.name === "mask");
        const imageIndex = inputs.findLastIndex(input => input.type === "IMAGE" && input.name.startsWith("model.images."));
        if (maskIndex < 0 || imageIndex < 0) return;
        const target = imageIndex + (maskIndex < imageIndex ? 0 : 1);
        if (target === maskIndex) return;
        const mask = inputs[maskIndex];
        const linkFor = input => input.link == null ? undefined : node.graph?.links.get(input.link);
        const maskLink = linkFor(mask);
        // Free one slot before shifting: current LiteGraph rejects occupied link targets.
        if (maskLink) maskLink.target_slot = inputs.length;
        const step = maskIndex < target ? 1 : -1;
        for (let i = maskIndex + step; i !== target + step; i += step) {
            const link = linkFor(inputs[i]);
            if (link) link.target_slot = i - step;
        }
        if (maskLink) maskLink.target_slot = target;
        inputs.splice(maskIndex, 1);
        inputs.splice(target, 0, mask);
        node.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "APIYiGPT.Billing",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["APIYiGPT", "APIYiGPTFallback"].includes(nodeData.name)) return;

        for (const name of ["addInput", "removeInput", "onConfigure"]) {
            const original = nodeType.prototype[name];
            nodeType.prototype[name] = function () {
                const result = original?.apply(this, arguments);
                orderMaskAfterImages(this);
                return result;
            };
        }

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const widget = ComfyWidgets.STRING(this, "尺寸、费用与余额", [
                "STRING", { multiline: true, default: "运行后显示实际尺寸、费用和余额（USD）" },
            ], app).widget;
            widget.inputEl.readOnly = true;
            widget.options = { ...widget.options, serialize: false };
            widget.serializeValue = async () => undefined;
            this.apiyiBillingWidget = widget;
            orderMaskAfterImages(this);
            return result;
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            if (this.apiyiBillingWidget && Array.isArray(message.apiyi_billing)) {
                this.apiyiBillingWidget.value = [
                    ...(message.apiyi_dimensions ?? []), ...message.apiyi_billing,
                ].join("\n\n");
                app.graph.setDirtyCanvas(true, false);
            }
            return result;
        };
    },
});
