import { app } from "../../scripts/app.js";

const previousNames = new Set([
    "PixelDriftFix",
    "Pixel Drift Fix",
    "像素漂移修复",
    "像素漂移修复（自动对齐）",
]);

app.registerExtension({
    name: "MY.AutoAlign",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PixelDriftFix") return;
        for (const hook of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                if (previousNames.has(this.title)) this.title = "自动对齐";
                return result;
            };
        }
    },
});
