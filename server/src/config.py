import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env")

_default_model_config = SettingsConfigDict(
    env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
)


class Config(BaseSettings):
    model_config = _default_model_config

    app_port: int = Field(alias="APP_PORT")
    allowed_origins: str = Field(alias="ALLOWED_ORIGINS")


@lru_cache()
def get_config() -> Config:
    """Function for getting all settings."""
    return Config()  # type: ignore


cfg = get_config()
