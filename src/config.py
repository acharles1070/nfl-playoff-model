"""Project configuration utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load the project's YAML configuration."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def project_path(*parts: str) -> Path:
    """Return an absolute path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def configured_path(name: str) -> Path:
    """Return an absolute path for a path defined in config.yaml."""
    relative = CONFIG["paths"][name]
    return project_path(relative)