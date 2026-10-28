"""只读取 ComfyUI 的已注册节点，检查流程依赖；不执行工作流或调用付费接口。"""
import argparse
import json
from pathlib import Path
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--include-archive', action='store_true')
    args = parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip('/') + '/object_info', timeout=30) as response:
        info = json.load(response)
    entries = json.loads((ROOT / 'manifests/workflows.json').read_text(encoding='utf-8'))
    missing = {}
    unresolved = {}
    for entry in entries:
        if '/archive/' in entry['path'] and not args.include_archive:
            continue
        needed = set(entry['node_dependencies'])
        absent = sorted(needed - set(info))
        if absent:
            missing[entry['path']] = absent
        still_unresolved = sorted(set(entry['unresolved_node_types']) - set(info))
        if still_unresolved:
            unresolved[entry['path']] = still_unresolved
    print(json.dumps({'registered_node_count': len(info), 'missing_known_nodes': missing, 'unresolved_at_snapshot': unresolved, 'note': '仅核对节点类型，不验证模型文件、端口服务以外的网络或照片效果。'}, ensure_ascii=False, indent=2))
    return 1 if missing or unresolved else 0


if __name__ == '__main__':
    sys.exit(main())
