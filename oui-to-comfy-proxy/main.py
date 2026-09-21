import asyncio
import base64
import hmac
import mimetypes
import random
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .comfyui import ComfyUIError, ComfyUITimeout, ComfyUIClient
from .config import Settings, load_settings
from .openai_api import RequestError, image_response, parse_size, validate_n
from .templates import (
    MAX_IMAGES,
    Templates,
    build_edit_graph,
    build_generation_graph,
    find_save_node_id,
    load_templates,
)

IMAGE_PART_RE = re.compile(r"^image(\[\d*\])?$")


@dataclass
class AppContext:
    settings: Settings
    templates: Templates
    comfyui: ComfyUIClient


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RequestError("n must be an integer")


async def collect_outputs(ctx: AppContext, entry: dict) -> list[str]:
    save_id = find_save_node_id(entry["prompt"][0])
    saved = (entry.get("outputs", {}).get(save_id) or {}).get("images") or []
    if not saved:
        raise ComfyUIError("no saved images in output")
    data = await ctx.comfyui.fetch_image(
        saved[0]["filename"],
        saved[0].get("subfolder", ""),
        saved[0].get("type", "output"),
    )
    return [base64.b64encode(data).decode()]


def create_app(
    settings: Settings | None = None, comfyui_client: ComfyUIClient | None = None
) -> FastAPI:
    settings = settings or load_settings()
    workflow_dir = Path(settings.workflow_dir)
    if not workflow_dir.is_absolute():
        workflow_dir = Path(__file__).resolve().parent.parent / workflow_dir
    templates = load_templates(workflow_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = comfyui_client or ComfyUIClient(settings.comfyui_url)
        app.state.ctx = AppContext(settings, templates, client)
        try:
            yield
        finally:
            if comfyui_client is None:
                await client.aclose()

    app = FastAPI(title="oui-to-comfy-proxy", lifespan=lifespan)

    @app.exception_handler(RequestError)
    async def request_error_handler(request: Request, exc: RequestError):
        return JSONResponse(status_code=400, content={"detail": exc.detail})

    @app.exception_handler(ComfyUIError)
    async def comfyui_error_handler(request: Request, exc: ComfyUIError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ComfyUITimeout)
    async def comfyui_timeout_handler(request: Request, exc: ComfyUITimeout):
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    def get_ctx(request: Request) -> AppContext:
        return request.app.state.ctx

    def require_key(
        authorization: str | None = Header(default=None),
        ctx: AppContext = Depends(get_ctx),
    ):
        key = ctx.settings.proxy_api_key
        if not key:
            return
        if not authorization or not hmac.compare_digest(authorization, f"Bearer {key}"):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(ctx: AppContext = Depends(get_ctx), _=Depends(require_key)):
        return {
            "object": "list",
            "data": [
                {
                    "id": ctx.settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "oui-to-comfy-proxy",
                }
            ],
        }

    @app.post("/v1/images/generations")
    async def generate(
        request: Request, ctx: AppContext = Depends(get_ctx), _=Depends(require_key)
    ):
        body = await request.json()
        prompt = body.get("prompt")
        if not prompt or not str(prompt).strip():
            raise RequestError("prompt is required")
        n = validate_n(_as_int(body.get("n")))
        width, height = parse_size(body.get("size"))
        b64_list: list[str] = []
        for _ in range(n):
            graph = build_generation_graph(
                ctx.templates.generation,
                str(prompt),
                width,
                height,
                random.getrandbits(32),
            )
            prompt_id = await ctx.comfyui.queue(graph)
            entry = await ctx.comfyui.wait(
                prompt_id, ctx.settings.timeout, ctx.settings.poll_interval
            )
            b64_list.extend(await collect_outputs(ctx, entry))
        return image_response(b64_list)

    @app.post("/v1/images/edits")
    async def edit(
        request: Request, ctx: AppContext = Depends(get_ctx), _=Depends(require_key)
    ):
        form = await request.form()
        prompt = form.get("prompt")
        if not prompt or not str(prompt).strip():
            raise RequestError("prompt is required")
        n = validate_n(_as_int(form.get("n")))
        files = [
            v
            for k, v in form.multi_items()
            if not isinstance(v, str) and IMAGE_PART_RE.match(k)
        ]
        if not files:
            raise RequestError("at least one image is required")
        if len(files) > MAX_IMAGES:
            raise RequestError(f"at most {MAX_IMAGES} images are supported")
        batch = uuid.uuid4().hex[:8]

        async def upload(i: int, f) -> str:
            data = await f.read()
            ext = mimetypes.guess_extension(f.content_type or "") or ".png"
            return await ctx.comfyui.upload_image(
                data,
                f"oui-to-comfy-proxy_{batch}_{i}{ext}",
                f.content_type or "application/octet-stream",
            )

        filenames = list(
            await asyncio.gather(*[upload(i, f) for i, f in enumerate(files, start=1)])
        )
        b64_list = []
        for _ in range(n):
            graph = build_edit_graph(
                ctx.templates.edit, str(prompt), "", filenames, random.getrandbits(32)
            )
            prompt_id = await ctx.comfyui.queue(graph)
            entry = await ctx.comfyui.wait(
                prompt_id, ctx.settings.timeout, ctx.settings.poll_interval
            )
            b64_list.extend(await collect_outputs(ctx, entry))
        return image_response(b64_list)

    return app


app = create_app()
