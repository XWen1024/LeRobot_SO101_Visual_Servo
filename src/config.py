"""
config.py - Loads and provides access to config.yaml settings.
"""
import yaml
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
_config = None


def load_config() -> dict:
    global _config
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    global _config
    if _config is None:
        load_config()
    return _config


def save_config():
    """Persist the entire in-memory config back to config.yaml."""
    global _config
    if _config is None:
        return
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_camera_index(index: int):
    """Persist the selected camera index back to config.yaml."""
    cfg = get_config()
    cfg["camera"]["default_index"] = index
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
