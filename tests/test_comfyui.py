import re

import httpx
import pytest

from imgproxy.comfyui import ComfyUIClient, ComfyUIError, ComfyUITimeout


async def test_upload_image():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["filename"] = (
            re.search(rb'filename="([^"]+)"', request.content).group(1).decode()
        )
        seen["overwrite"] = (
            re.search(rb'name="overwrite"\r\n\r\ntrue\r\n', request.content) is not None
        )
        return httpx.Response(
            200, json={"name": "server_name.png", "subfolder": "", "type": "input"}
        )

    client = ComfyUIClient("http://comfy.test", transport=httpx.MockTransport(handler))
    name = await client.upload_image(b"bytes", "a.png", "image/png")
    assert name == "server_name.png"
    assert seen["path"] == "/upload/image"
    assert seen["filename"] == "a.png"
    assert seen["overwrite"]


async def test_upload_image_failure_raises():
    client = ComfyUIClient(
        "http://comfy.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(ComfyUIError):
        await client.upload_image(b"bytes", "a.png", "image/png")


async def test_queue_returns_prompt_id():
    client = ComfyUIClient(
        "http://comfy.test",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"prompt_id": "p1", "number": 3})
        ),
    )
    assert await client.queue({"1": {}}) == "p1"


async def test_wait_polls_until_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={
                "p1": {
                    "status": {"status_str": "success"},
                    "outputs": {"9": {"images": []}},
                }
            },
        )

    client = ComfyUIClient("http://comfy.test", transport=httpx.MockTransport(handler))
    entry = await client.wait("p1", timeout=5, poll_interval=0.01)
    assert entry["status"]["status_str"] == "success"
    assert calls["n"] == 2


async def test_wait_error_status_raises():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "p1": {
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

    client = ComfyUIClient("http://comfy.test", transport=httpx.MockTransport(handler))
    with pytest.raises(ComfyUIError, match="bad node"):
        await client.wait("p1", timeout=5, poll_interval=0.01)


async def test_wait_timeout_raises():
    client = ComfyUIClient(
        "http://comfy.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(ComfyUITimeout):
        await client.wait("p1", timeout=0.05, poll_interval=0.01)


async def test_fetch_image():
    client = ComfyUIClient(
        "http://comfy.test",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=b"PNGDATA")
        ),
    )
    data = await client.fetch_image("out.png", "")
    assert data == b"PNGDATA"
