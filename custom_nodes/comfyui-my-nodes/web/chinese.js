import { app } from "/scripts/app.js";

// Only labels change. Names and serialized workflow inputs stay intact.
const definitions = {
    MYSkinBlemish: {
        inputs: { min_spot_size: "最小瑕疵直径（像素）", ignore_smallest_percent: "忽略最小区域比例（%）", protect_hair: "自动保护头发", image: "原图", sensitivity: "检测灵敏度", max_spot_size: "最大瑕疵尺寸（像素）", skin_range: "肤色范围", strength: "修复强度", dark_amount: "暗斑／黑头／痦子", red_amount: "红点／红痘", bright_amount: "白头／亮点", mask: "处理范围蒙版", protect_mask: "保护蒙版" },
        outputs: { image: "修复图像", spot_mask: "瑕疵蒙版", skin_mask: "皮肤蒙版" },
    },
    PixelDriftFix: {
        inputs: { source_image: "原图（对齐基准）", edited_image: "编辑后图像" },
        outputs: { fixed_image: "对齐后图像" },
    },
    QwenAPI: { inputs: { model: "模型", prompt: "提示词", images: "参考图像" } },
    QwenImageAPI: { inputs: { seed: "种子" } },
    Image25TilePlan: { inputs: { image: "图像", max_long_edge: "分块长边上限", overlap: "重叠比例偏好" } },
};

function labelNode(node, labels) {
    for (const input of node.inputs ?? []) {
        if (labels.inputs?.[input.name]) input.label = labels.inputs[input.name];
    }
    for (const output of node.outputs ?? []) {
        if (labels.outputs?.[output.name]) output.label = labels.outputs[output.name];
    }
    for (const widget of node.widgets ?? []) {
        if (labels.inputs?.[widget.name]) widget.label = labels.inputs[widget.name];
    }
}

app.registerExtension({
    name: "MY.ChineseLabels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const labels = definitions[nodeData.name];
        if (!labels) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                labelNode(this, labels);
                queueMicrotask(() => labelNode(this, labels));
                return result;
            };
        }
    },
});




