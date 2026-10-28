"""Use the installed Darkroom floating-point colour implementation unchanged."""
import hashlib
import importlib.util
import importlib
import sys
import types
from pathlib import Path


def load_color_core(root=None):
    if root is None:
        import folder_paths
        candidates = [Path(p) for p in folder_paths.get_folder_paths('custom_nodes')]
        roots = [p / name for p in candidates for name in ('comfyui-darkroom', 'ComfyUI-Darkroom')]
        root = next((p for p in roots if (p / 'utils/colorspace.py').is_file()), None)
        if root is None:
            raise RuntimeError('请先安装 ComfyUI-Darkroom；本节点需要其浮点 colorspace.py。')
    path = Path(root) / 'utils/colorspace.py'
    name = '_darkroom_reference_colorspace_' + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name], {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load_gpu_core(color):
    root = Path(color.__file__).resolve().parents[1]
    name = '_darkroom_reference_gpu_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
        utils = types.ModuleType(name + '.utils')
        utils.__path__ = [str(root / 'utils')]
        sys.modules[utils.__name__] = utils
    return importlib.import_module(name + '.utils.gpu_color')
