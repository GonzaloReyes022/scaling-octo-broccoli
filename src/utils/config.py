"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/utils to project root
    current = Path(__file__).resolve()
    # Go up: config.py -> utils -> src -> project_root
    return current.parent.parent.parent


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    config_path = get_project_root() / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.environ.get(env_var, config)
    return config


def save_config(config: Dict[str, Any], config_name: str) -> Path:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_name: Name for the config file

    Returns:
        Path to saved config file
    """
    config_dir = get_project_root() / "configs"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / f"{config_name}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path
