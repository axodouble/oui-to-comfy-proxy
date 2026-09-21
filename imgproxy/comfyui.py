import asyncio
import uuid

import httpx


class ComfyUIError(Exception):
    pass


class ComfyUITimeout(Exception):
    pass


class ComfyUIClient:
    def __init__(self, base_url: str, request_timeout: float = 30.0, transport=None):
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=request_timeout, transport=transport
        )
        self._client_id = uuid.uuid4().hex

    async def aclose(self):
        await self._client.aclose()

    async def upload_image(self, data: bytes, filename: str, content_type: str) -> str:
        r = await self._client.post(
            "/upload/image",
            files={"image": (filename, data, content_type)},
            data={"overwrite": "true", "subfolder": ""},
        )
        if r.status_code != 200:
            raise ComfyUIError(f"upload failed: {r.status_code} {r.text[:200]}")
        name = r.json().get("name")
        if not name:
            raise ComfyUIError(f"upload response missing name: {r.text[:200]}")
        return name

    async def queue(self, graph: dict) -> str:
        r = await self._client.post(
            "/prompt", json={"prompt": graph, "client_id": self._client_id}
        )
        if r.status_code != 200:
            raise ComfyUIError(f"queue failed: {r.status_code} {r.text[:200]}")
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"queue response missing prompt_id: {r.text[:200]}")
        return prompt_id

    async def wait(
        self, prompt_id: str, timeout: float, poll_interval: float = 0.5
    ) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            r = await self._client.get(f"/history/{prompt_id}")
            if r.status_code != 200:
                raise ComfyUIError(f"history failed: {r.status_code} {r.text[:200]}")
            entry = r.json().get(prompt_id)
            if entry:
                status = (entry.get("status") or {}).get("status_str")
                if status == "success":
                    return entry
                if status == "error":
                    raise ComfyUIError(
                        f"execution failed: {str((entry.get('status') or {}).get('messages'))[:500]}"
                    )
            if loop.time() >= deadline:
                raise ComfyUITimeout(
                    f"timed out after {timeout}s waiting for {prompt_id}"
                )
            if deadline - loop.time() <= poll_interval:
                await asyncio.sleep(max(0, deadline - loop.time()))
            else:
                await asyncio.sleep(poll_interval)

    async def fetch_image(
        self, filename: str, subfolder: str, image_type: str = "output"
    ) -> bytes:
        r = await self._client.get(
            "/view",
            params={"filename": filename, "subfolder": subfolder, "type": image_type},
        )
        if r.status_code != 200:
            raise ComfyUIError(f"view failed: {r.status_code} {r.text[:200]}")
        return r.content
