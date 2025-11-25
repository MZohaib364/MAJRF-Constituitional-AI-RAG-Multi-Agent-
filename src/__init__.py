"""
Core package for the Constitutional Compliance Checker.

This module exposes helper functions to initialize configuration and
agent pipelines, and loads environment variables from .env files.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    _ROOT = Path(__file__).resolve().parent.parent
    for _env_file in (".env", "groq.env"):
        path = _ROOT / _env_file
        if path.exists():
            load_dotenv(dotenv_path=path, override=True)
except Exception:  # noqa: BLE001
    pass

from .config import ProjectConfig, load_project_config  # noqa: F401

