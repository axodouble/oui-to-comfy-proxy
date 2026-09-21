import base64
import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from imgproxy.comfyui import ComfyUIClient
from imgproxy.config import load_settings
from imgproxy.main import create_app

WORKFLOWS = Path(__file__).parent.parent / "workflows"
PNG = b"\x89PNG\r\n\x1a\nfake"


def make_comfyui_state():
    return {"uploads": [], "prompts": [], "history": {}}


def comfyui_handler(state):
    def handler(request):
        path = request.url.path
        if path == "/upload/image":
            name = re.search(rb'filename="([^"]+)"', request.content).group(1).decode()
            state["uploads"].append(name)
            return httpx.Response(
                200, json={"name": name, "subfolder": "", "type": "input"}
            )
        if path == "/prompt":
            payload = json.loads(request.content)
            prompt_id = f"p{len(state['prompts'])}"
            state["prompts"].append(payload["prompt"])
            save_id = next(
                i
                for i, n in payload["prompt"].items()
                if n["class_type"] == "SaveImageAdvanced"
            )
            state["history"][prompt_id] = {
                "prompt": [payload["prompt"], {}, 0, 0],
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    save_id: {
                        "images": [
                            {
                                "filename": f"out_{prompt_id}.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
            return httpx.Response(200, json={"prompt_id": prompt_id, "number": 1})
        if path.startswith("/history/"):
            return httpx.Response(200, json=state["history"])
        if path == "/view":
            return httpx.Response(200, content=PNG)
        return httpx.Response(404, text="not found")

    return handler


@pytest.fixture
def make_app():
    def _make(key=None, extra_env=None):
        state = make_comfyui_state()
        client = ComfyUIClient(
            "http://comfy.test", transport=httpx.MockTransport(comfyui_handler(state))
        )
        env_vars = {"WORKFLOW_DIR": str(WORKFLOWS), "POLL_INTERVAL": "0.01"}
        if key:
            env_vars["PROXY_API_KEY"] = key
        if extra_env:
            env_vars.update(extra_env)
        return TestClient(
            create_app(settings=load_settings(env_vars), comfyui_client=client)
        ), state

    return _make


def _latent_dims(graph):
    lat = next(n for n in graph.values() if n["class_type"] == "EmptyLatentImage")
    return lat["inputs"]["width"], lat["inputs"]["height"]


def test_generation_end_to_end(make_app):
    c, state = make_app()
    with c:
        r = c.post(
            "/v1/images/generations",
            json={
                "model": "qwen-image-2.1",
                "prompt": "a cat",
                "n": 1,
                "size": "auto",
                "response_format": "b64_json",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["created"] > 0
        assert len(body["data"]) == 1
        assert base64.b64decode(body["data"][0]["b64_json"]) == PNG
    assert _latent_dims(state["prompts"][0]) == (1024, 1024)


def test_generation_size_rounding(make_app):
    c, state = make_app()
    with c:
        r = c.post("/v1/images/generations", json={"prompt": "x", "size": "1000x1000"})
        assert r.status_code == 200
    assert _latent_dims(state["prompts"][0]) == (992, 992)


def test_generation_n2_two_runs_distinct_seeds(make_app):
    c, state = make_app()
    with c:
        r = c.post("/v1/images/generations", json={"prompt": "x", "n": 2})
        assert r.status_code == 200
        assert len(r.json()["data"]) == 2
    seeds = [
        next(n["inputs"]["seed"] for n in g.values() if n["class_type"] == "KSampler")
        for g in state["prompts"]
    ]
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_generation_validation_errors(make_app):
    c, _ = make_app()
    with c:
        assert c.post("/v1/images/generations", json={}).status_code == 400
        assert (
            c.post(
                "/v1/images/generations", json={"prompt": "x", "size": "big"}
            ).status_code
            == 400
        )
        assert (
            c.post("/v1/images/generations", json={"prompt": "x", "n": 9}).status_code
            == 400
        )


def test_auth(make_app):
    c, _ = make_app(key="sekret")
    with c:
        r = c.post("/v1/images/generations", json={"prompt": "x"})
        assert r.status_code == 401
        r = c.post(
            "/v1/images/generations",
            json={"prompt": "x"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401
        r = c.post(
            "/v1/images/generations",
            json={"prompt": "x"},
            headers={"Authorization": "Bearer sekret"},
        )
        assert r.status_code == 200


def test_comfyui_failure_maps_502(make_app):
    state = make_comfyui_state()
    client = ComfyUIClient(
        "http://comfy.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    c = TestClient(
        create_app(
            settings=load_settings({"WORKFLOW_DIR": str(WORKFLOWS)}),
            comfyui_client=client,
        )
    )
    with c:
        assert c.post("/v1/images/generations", json={"prompt": "x"}).status_code == 502


def test_comfyui_error_status_maps_502():
    def handler(request):
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p0", "number": 1})
        if request.url.path.startswith("/history/"):
            return httpx.Response(
                200,
                json={
                    "p0": {
                        "status": {
                            "status_str": "error",
                            "messages": [
                                ["execution_error", {"exception_message": "bad node"}]
                            ],
                        },
                        "outputs": {},
                    }
                },
            )
        return httpx.Response(404, text="not found")

    client = ComfyUIClient("http://comfy.test", transport=httpx.MockTransport(handler))
    c = TestClient(
        create_app(
            settings=load_settings({"WORKFLOW_DIR": str(WORKFLOWS)}),
            comfyui_client=client,
        )
    )
    with c:
        r = c.post("/v1/images/generations", json={"prompt": "x"})
        assert r.status_code == 502
        assert "bad node" in r.json()["detail"]


def test_timeout_maps_504():
    def handler(request):
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p0", "number": 1})
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={})
        return httpx.Response(404, text="not found")

    client = ComfyUIClient("http://comfy.test", transport=httpx.MockTransport(handler))
    c = TestClient(
        create_app(
            settings=load_settings(
                {
                    "WORKFLOW_DIR": str(WORKFLOWS),
                    "TIMEOUT": "0.05",
                    "POLL_INTERVAL": "0.01",
                }
            ),
            comfyui_client=client,
        )
    )
    with c:
        assert c.post("/v1/images/generations", json={"prompt": "x"}).status_code == 504


def test_health(make_app):
    c, _ = make_app()
    with c:
        assert c.get("/health").json() == {"status": "ok"}
