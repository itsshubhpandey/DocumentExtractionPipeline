"""Project environment loading with deterministic precedence."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_environment(
    dotenv_path: str | Path = PROJECT_ENV_FILE,
    *,
    override: bool = True,
) -> bool:
    """Load the project ``.env`` file, overriding inherited shell values."""

    return load_dotenv(
        dotenv_path=Path(dotenv_path),
        override=override,
    )
