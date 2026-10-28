from .pixel_drift_fix_node import PixelDriftFixNode

# Keep the node ID so existing workflow connections survive the display rename.
NODE_CLASS_MAPPINGS = {"PixelDriftFix": PixelDriftFixNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelDriftFix": "自动对齐"}
