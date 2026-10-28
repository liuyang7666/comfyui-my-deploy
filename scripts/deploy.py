"""将本仓库的节点和已保存工作流安装到指定 ComfyUI，覆盖前自动备份。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(comfyui, *, dry_run=False, install_deps=False, nodes=None):
    comfyui = Path(comfyui).expanduser().resolve()
    if not (comfyui / 'main.py').is_file() or not (comfyui / 'custom_nodes').is_dir():
        raise ValueError('请指定包含 main.py 和 custom_nodes 的 ComfyUI 根目录。')
    if comfyui == ROOT or ROOT in comfyui.parents or comfyui in ROOT.parents:
        raise ValueError('ComfyUI 目录不能与部署仓库重叠。')
    manifest = json.loads((ROOT / 'manifests/snapshot.json').read_text(encoding='utf-8'))
    available = {x['directory'] for x in manifest['local_nodes']}
    selected = set(nodes or available)
    if selected - available:
        raise ValueError('未知节点包：' + ', '.join(sorted(selected - available)))
    operations = []
    if 'comfyui-my-nodes' in selected:
        for legacy in ('comfyui-qwen-api', 'image25-tile'):
            target = comfyui / 'custom_nodes' / legacy
            if target.exists():
                operations.append((None, target))
    for name in sorted(selected):
        operations.append((ROOT / 'custom_nodes' / name, comfyui / 'custom_nodes' / name))
    for category in ('saved', 'examples'):
        source = ROOT / 'workflows' / category
        if source.exists():
            for file in sorted(source.rglob('*.json')):
                operations.append((file, comfyui / 'user/default/workflows/我的部署包' / category / file.relative_to(source)))
    source = ROOT / 'subgraphs'
    for file in sorted(source.rglob('*.json')):
        operations.append((file, comfyui / 'user/default/subgraphs' / file.relative_to(source)))
    for source, target in operations:
        if source is not None and (source.is_symlink() or any(x.is_symlink() for x in source.rglob('*'))):
            raise ValueError('部署源不能包含符号链接。')
        target.resolve().relative_to(comfyui)
        label = '停用旧重复目录：' if source is None else '安装：'
        print(('预览 ' if dry_run else '') + label + str(target.relative_to(comfyui)))
    if dry_run:
        print(f'预览完成，共 {len(operations)} 项；没有写入目标目录。')
        return None
    backup = comfyui / 'deploy-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.resolve().relative_to(comfyui)
    backup.mkdir(parents=True, exist_ok=False)
    journal = []
    for source, target in operations:
        relative = str(target.relative_to(comfyui))
        entry = {'target': relative, 'existed': target.exists()}
        if target.exists():
            previous = backup / relative
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(previous))
        journal.append(entry)
        (backup / 'restore.json').write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding='utf-8')
        if source is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(source, target)
    if install_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-local.txt')], check=True)
    print('部署完成；回滚记录：' + str(backup))
    print('请检查第三方节点和模型清单，重新配置密钥，然后重启 ComfyUI 并刷新页面。')
    return backup


def restore(comfyui, backup, *, dry_run=False):
    comfyui = Path(comfyui).resolve()
    backup = Path(backup).resolve()
    backup.relative_to((comfyui / 'deploy-backups').resolve())
    journal = json.loads((backup / 'restore.json').read_text(encoding='utf-8'))
    if (backup / 'rolled-back.json').exists():
        raise ValueError('该备份已经回滚，请勿重复执行。')
    displaced = backup / 'replaced-on-rollback'
    for entry in journal:
        (comfyui / entry['target']).resolve().relative_to(comfyui)
        previous = (backup / entry['target']).resolve()
        previous.relative_to(backup)
        if entry['existed'] and not previous.exists():
            raise ValueError('回滚备份不完整：' + entry['target'])
    for entry in reversed(journal):
        target = comfyui / entry['target']
        target.resolve().relative_to(comfyui)
        previous = backup / entry['target']
        print('回滚：' + entry['target'])
        if dry_run:
            continue
        if target.exists():
            saved = displaced / entry['target']
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(saved))
        if entry['existed']:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(previous), str(target))
    if not dry_run:
        (backup / 'rolled-back.json').write_text('{}', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfyui', required=True, help='目标 ComfyUI 根目录')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不写入目标目录')
    parser.add_argument('--install-deps', action='store_true', help='用当前 Python 安装本地节点附加依赖')
    parser.add_argument('--nodes', nargs='+', help='只安装指定节点包；工作流仍会复制')
    parser.add_argument('--restore', help='使用 deploy-backups 内的备份目录回滚')
    args = parser.parse_args()
    if args.restore:
        restore(args.comfyui, args.restore, dry_run=args.dry_run)
    else:
        run(args.comfyui, dry_run=args.dry_run, install_deps=args.install_deps, nodes=args.nodes)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('部署失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
