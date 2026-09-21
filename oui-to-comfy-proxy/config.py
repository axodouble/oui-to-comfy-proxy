import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    comfyui_url: str = "http://127.0.0.1:8188"
    model_name: str = "qwen-image-2.1"
    workflow_dir: str = "./workflows"
    timeout: float = 600.0
    poll_interval: float = 0.5
    proxy_api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000


def load_settings(env: dict | None = None) -> Settings:
    env = env if env is not None else os.environ
    return Settings(
        comfyui_url=env.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/"),
        model_name=env.get("MODEL_NAME", "qwen-image-2.1"),
        workflow_dir=env.get("WORKFLOW_DIR", "./workflows"),
        timeout=float(env.get("TIMEOUT", "600")),
        poll_interval=float(env.get("POLL_INTERVAL", "0.5")),
        proxy_api_key=env.get("PROXY_API_KEY") or None,
        host=env.get("HOST", "0.0.0.0"),
        port=int(env.get("PORT", "8000")),
    )
