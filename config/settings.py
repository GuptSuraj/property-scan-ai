"""Validated environment settings, loaded only when explicitly requested."""

from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from property_scanner.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Environment overrides .env; explicit arguments override both."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    model_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    input_dir: Path = Path("inputs")

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("APP_ENV must not be empty")
        return value.strip()

    @field_validator("model_dir", "output_dir", "input_dir", mode="before")
    @classmethod
    def validate_directory_value(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("directory paths must not be empty")
        return value

    @field_validator("model_dir", "output_dir", "input_dir")
    @classmethod
    def normalize_directory(cls, value: Path) -> Path:
        path = value.expanduser().resolve()
        if path.exists() and not path.is_dir():
            raise ValueError("expected a directory path")
        return path


def load_settings() -> Settings:
    """Translate configuration errors into concise application errors."""
    try:
        return Settings()
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"Invalid settings: {details}") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(f"Unable to load settings: {exc}") from exc
