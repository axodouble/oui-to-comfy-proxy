from oui_to_comfy_proxy.config import load_settings


def test_defaults():
    s = load_settings({})
    assert s.comfyui_url == "http://127.0.0.1:8188"
    assert s.model_name == "qwen-image-2.1"
    assert s.workflow_dir == "./workflows"
    assert s.timeout == 600.0
    assert s.poll_interval == 0.5
    assert s.proxy_api_key is None
    assert s.host == "0.0.0.0"
    assert s.port == 8000


def test_overrides():
    s = load_settings(
        {
            "COMFYUI_URL": "http://gpu:8188/",
            "MODEL_NAME": "qwen",
            "TIMEOUT": "120",
            "POLL_INTERVAL": "1",
            "PROXY_API_KEY": "k",
            "PORT": "9000",
        }
    )
    assert s.comfyui_url == "http://gpu:8188"
    assert s.model_name == "qwen"
    assert s.timeout == 120.0
    assert s.poll_interval == 1.0
    assert s.proxy_api_key == "k"
    assert s.port == 9000
