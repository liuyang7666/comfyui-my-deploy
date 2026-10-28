import { app } from "/scripts/app.js";

const EMPTY_TITLE = "请先点击添加";

function entriesOf(node) {
    return JSON.parse(node.widgets.find(widget => widget.name === "entries_json").value);
}

function refreshChoices(node) {
    const entries = entriesOf(node);
    const choice = node.widgets.find(widget => widget.name === "selected");
    const titles = entries.map(entry => entry.title);
    choice.options.values = titles.length ? titles : [EMPTY_TITLE];
    if (!titles.includes(choice.value)) choice.value = titles[0] ?? EMPTY_TITLE;
    node.setDirtyCanvas(true, true);
}

function saveEntry(node, previousTitle, title, text) {
    title = title.trim();
    if (!title) throw new Error("请输入标题。");
    const entries = entriesOf(node);
    if (entries.some(entry => entry.title === title && entry.title !== previousTitle)) {
        throw new Error("这个标题已存在，请换一个标题。");
    }
    const index = previousTitle == null ? -1 : entries.findIndex(entry => entry.title === previousTitle);
    if (previousTitle != null && index < 0) throw new Error("选项已变动，请重新打开修改窗口。");
    node.graph?.beforeChange(node);
    if (index < 0) entries.push({ title, text });
    else entries[index] = { title, text };
    node.widgets.find(widget => widget.name === "entries_json").value = JSON.stringify(entries);
    node.widgets.find(widget => widget.name === "selected").value = title;
    refreshChoices(node);
    node.graph?.afterChange(node);
}

function openEditor(node, editing) {
    const choice = node.widgets.find(widget => widget.name === "selected");
    const entry = editing ? entriesOf(node).find(item => item.title === choice.value) : null;
    node.myDropdownEditor?.close();

    const dialog = document.createElement("dialog");
    dialog.className = "my-dropdown-editor";
    dialog.setAttribute("aria-label", entry ? "修改选项" : "添加选项");
    const form = document.createElement("form");
    const heading = document.createElement("h2");
    heading.textContent = entry ? "修改选项" : "添加选项";

    const titleLabel = document.createElement("label");
    titleLabel.textContent = "标题";
    const titleInput = document.createElement("input");
    titleInput.type = "text";
    titleInput.name = "title";
    titleInput.placeholder = "例如：白色";
    titleInput.value = entry?.title ?? "";
    titleInput.autocomplete = "off";
    titleLabel.append(titleInput);

    const textLabel = document.createElement("label");
    textLabel.textContent = "文本";
    const textInput = document.createElement("textarea");
    textInput.name = "text";
    textInput.placeholder = "输入这个标题对应的完整文本，支持多行长文本。";
    textInput.value = entry?.text ?? "";
    textInput.spellcheck = false;
    textLabel.append(textInput);

    const error = document.createElement("p");
    error.className = "my-dropdown-error";
    error.setAttribute("role", "alert");
    const actions = document.createElement("div");
    actions.className = "my-dropdown-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "取消";
    cancel.addEventListener("click", () => dialog.close());
    const save = document.createElement("button");
    save.type = "submit";
    save.textContent = "保存";
    save.className = "my-dropdown-save";
    actions.append(cancel, save);
    form.append(heading, titleLabel, textLabel, error, actions);
    form.addEventListener("submit", event => {
        event.preventDefault();
        try {
            saveEntry(node, entry?.title ?? null, titleInput.value, textInput.value);
            dialog.close();
        } catch (failure) {
            error.textContent = failure.message;
        }
    });
    form.addEventListener("keydown", event => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            form.requestSubmit();
        }
    });
    dialog.addEventListener("close", () => {
        dialog.remove();
        if (node.myDropdownEditor === dialog) delete node.myDropdownEditor;
    });
    dialog.append(form);
    document.body.append(dialog);
    node.myDropdownEditor = dialog;
    dialog.showModal();
    titleInput.focus();
}

app.registerExtension({
    name: "MY.DropdownSelect",
    setup() {
        const style = document.createElement("style");
        style.textContent = `
            .my-dropdown-editor {
                box-sizing: border-box; width: min(720px, 92vw); max-height: 90vh;
                padding: 24px; border: 1px solid var(--border-color, #555); border-radius: 12px;
                background: var(--comfy-menu-bg, #252525); color: var(--fg-color, #eee);
                font: 14px/1.5 system-ui, sans-serif; box-shadow: 0 16px 64px #0008;
            }
            .my-dropdown-editor::backdrop { background: #0009; }
            .my-dropdown-editor form { display: flex; flex-direction: column; gap: 16px; }
            .my-dropdown-editor h2 { margin: 0; font-size: 19px; }
            .my-dropdown-editor label { display: flex; flex-direction: column; gap: 6px; }
            .my-dropdown-editor input, .my-dropdown-editor textarea {
                box-sizing: border-box; width: 100%; padding: 10px 12px;
                border: 1px solid var(--border-color, #555); border-radius: 6px;
                background: var(--comfy-input-bg, #171717); color: inherit; font: inherit;
            }
            .my-dropdown-editor textarea { height: min(360px, 42vh); min-height: 150px; resize: vertical; }
            .my-dropdown-editor input:focus, .my-dropdown-editor textarea:focus {
                outline: 2px solid #6d9de7; outline-offset: 1px;
            }
            .my-dropdown-error { min-height: 21px; margin: 0; color: #ff9292; }
            .my-dropdown-actions { display: flex; justify-content: flex-end; gap: 10px; }
            .my-dropdown-actions button {
                padding: 8px 24px; border: 1px solid #666; border-radius: 6px;
                background: var(--comfy-input-bg, #333); color: inherit; font: inherit; cursor: pointer;
            }
            .my-dropdown-actions .my-dropdown-save { background: #356ac3; border-color: #5688db; color: white; }
        `;
        document.head.append(style);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MYDropdownSelect") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const data = this.widgets.find(widget => widget.name === "entries_json");
            data.type = "converted-widget";
            data.hidden = true;
            data.computeSize = () => [0, -4];
            data.draw = () => {};
            const choice = this.widgets.find(widget => widget.name === "selected");
            choice.type = "combo";
            choice.label = "标题";
            choice.options = { ...choice.options, values: [] };
            this.addWidget("button", "添加", null, () => openEditor(this, false), { serialize: false });
            this.addWidget("button", "修改", null, () => openEditor(this, true), { serialize: false });
            refreshChoices(this);
            this.setSize([Math.max(300, this.size[0]), this.computeSize()[1]]);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            refreshChoices(this);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.myDropdownEditor?.close();
            return removed?.apply(this, arguments);
        };
    },
});
