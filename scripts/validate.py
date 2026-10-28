"""离线检查源码语法、JSON、敏感文件、令牌和发布清单完整性。"""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,})')
FORBIDDEN = {'.dpapi', '.pt', '.pth', '.safetensors', '.ckpt', '.onnx', '.gguf', '.exe', '.dll', '.db', '.pem', '.key', '.png', '.jpg', '.jpeg', '.webp', '.cr3'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(t in {'.git', '__pycache__'} for t in p.relative_to(ROOT).parts) and p.suffix != '.pyc')


def secrets_in_json(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if re.fullmatch(r'(api[_-]?key|access[_-]?token|auth[_-]?token|account[_-]?token|password|secret)', key, re.I) and isinstance(child, str) and child:
                raise ValueError('JSON 包含非空密钥字段，需本地复核；不输出字段值。')
            secrets_in_json(child)
    elif isinstance(value, list):
        for child in value:
            secrets_in_json(child)


def validate():
    all_files = files()
    for path in all_files:
        if path.suffix.lower() in FORBIDDEN or path.name.startswith(('.env', 'credentials')):
            raise ValueError('不应上传的文件：' + str(path.relative_to(ROOT)))
        text = path.read_text(encoding='utf-8-sig')
        if TOKEN.search(text):
            raise ValueError('疑似令牌；请本地检查：' + str(path.relative_to(ROOT)))
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        elif path.suffix == '.json':
            secrets_in_json(json.loads(text))
    manifest = json.loads((ROOT / 'manifests/file-sha256.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_files if p.relative_to(ROOT).as_posix() != 'manifests/file-sha256.json'}
    if manifest != actual:
        mismatched = sorted(set(manifest) ^ set(actual) | {p for p in set(manifest) & set(actual) if manifest[p] != actual[p]})
        raise ValueError('文件完整性不匹配：' + ', '.join(mismatched))
    print(f'通过：{len(all_files)} 个文件；Python/JSON 可解析，发布文件校验一致，未发现所检查的敏感文件或令牌格式。')
    return len(all_files)


if __name__ == '__main__':
    try:
        validate()
    except (ValueError, OSError, SyntaxError) as exc:
        print('检查失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
