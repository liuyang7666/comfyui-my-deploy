import { app } from "../../scripts/app.js";

// Preserve old saved workflows after removing the manual scheduling fields.
app.registerExtension({
    name: "ConcurrentImages.WorkflowMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ConcurrentImageSubmit") return;
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = configured?.apply(this, arguments);
            if (info.widgets_values?.length >= 10) {
                const oldNames = ["prompt", "route", "primary_model", "fallback_model", "quality", "concurrency", "retries", "request_timeout", "seed", "primary_url"];
                const previous = Object.fromEntries(oldNames.map((name, i) => [name, info.widgets_values[i]]));
                for (const widget of this.widgets ?? []) {
                    if (Object.hasOwn(previous, widget.name)) widget.value = previous[widget.name];
                }
            }
            for (let i = (this.inputs?.length ?? 0) - 1; i >= 0; i--) {
                if (["concurrency", "request_timeout", "primary_url"].includes(this.inputs[i].name)) this.removeInput(i);
            }
            return result;
        };
    },
});
