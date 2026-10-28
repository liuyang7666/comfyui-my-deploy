"""A workflow-local dropdown that returns the full text behind a short title."""

import json

from comfy_api.latest import io as IO


def parse_entries(entries_json):
    try:
        entries = json.loads(entries_json)
    except json.JSONDecodeError as exc:
        raise ValueError("下拉选项数据损坏，请重新保存选项。") from exc
    if not isinstance(entries, list):
        raise ValueError("下拉选项必须是标题和文本组成的列表。")
    titles = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("title"), str) or not isinstance(entry.get("text"), str):
            raise ValueError("每个下拉选项都必须包含标题和文本。")
        title = entry["title"]
        if not title.strip():
            raise ValueError("下拉选项的标题不能为空。")
        if title in titles:
            raise ValueError(f"下拉选项标题重复：{title}")
        titles.add(title)
    return entries


class DropdownSelect(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MYDropdownSelect",
            display_name="下拉选择",
            category="MY/文本",
            description="下拉只显示标题，输出对应的完整长文本。点击添加或修改，在弹窗中编辑标题和文本；选项随工作流保存。",
            search_aliases=["下拉", "文本选择", "标题", "dropdown", "preset text"],
            inputs=[
                IO.String.Input("selected", display_name="标题", default="", socketless=True, dynamic_prompts=False),
                IO.String.Input("entries_json", default="[]", socketless=True, dynamic_prompts=False),
            ],
            outputs=[IO.String.Output("text", display_name="文本")],
        )

    @classmethod
    def execute(cls, selected, entries_json):
        entries = parse_entries(entries_json)
        if not entries:
            return IO.NodeOutput("")
        for entry in entries:
            if entry["title"] == selected:
                return IO.NodeOutput(entry["text"])
        raise ValueError(f"未找到下拉标题“{selected}”，请重新选择。")
