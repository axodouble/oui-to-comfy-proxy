# oui-to-comfy-proxy — OpenAI-compatible image proxy for Qwen Image 2.1 via ComfyUI

Date: 2026-09-21
Status: Approved design, pending spec review

## 1. Purpose

A small OpenAI-compatible HTTP service that lets Open WebUI drive the
Qwen Image 2.1 model running in ComfyUI:

- `POST /v1/images/generations` — text-to-image
- `POST /v1/images/edits` — image-to-image edit with **any number of
  reference images (1–16)**, mapped automatically onto the ComfyUI
  workflow's `images.image_1..image_N` inputs

The proxy builds ComfyUI prompt graphs from two JSON workflow templates
supplied by the user, mutating only the dynamic parts (prompt, seed,
size, input images). All model wiring (GGUF unet, qwen3vl CLIP, VAE,
cache node, switch, save format) lives in the template files and can be
changed without touching code.

## 2. Verified upstream contracts

### 2.1 Open WebUI → OpenAI (from `backend/open_webui/routers/images.py`)

**Generation** — `POST {base}/images/generations`, JSON:

```json
{
  "model": "<model name>",
  "prompt": "<prompt>",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "...": "merged IMAGES_OPENAI_API_PARAMS (ignored)"
}
```

`size` is the configured `IMAGE_SIZE` (e.g. `"auto"`, a `"WxH"` string)
or absent. The client expects the OpenAI response:
`{"created": ..., "data": [{"url": ...} | {"b64_json": ...}]}`.

**Edit** — `POST {base}/images/edits`, `multipart/form-data`:

- form fields: `model`, `prompt`, and optionally `n`, `size`,
  `background`, `response_format`
- files: a single image in field `image`; multiple images as repeated
  `image[]` fields. Files carry the decoded image bytes (Open WebUI
  converts data-URLs to file items before sending) with a filename and
  content type.

Response shape: same as generation.

### 2.2 ComfyUI API (standard, confirmed present in the pinned clone)

- `POST /upload/image` — multipart: file field `image`, plus
  `overwrite` and `subfolder` fields → `{"name", "subfolder", "type"}`
- `POST /prompt` — `{"prompt": <graph>, "client_id": "<uuid>"}` →
  `{"prompt_id", "number"}`
- `GET /history/{prompt_id}` — `{<prompt_id>: {"status": {...}, "outputs": {...}}}`
- `GET /view?filename=...&subfolder=...&type=output` — image bytes

### 2.3 Node contract (`comfy_extras/nodes_qwen.py`, `TextEncodeQwenImage21`)

Inputs: `clip`, `prompt`, `negative_prompt`, `vae` (optional),
`resolution` (int, default 1024; 0 = keep each reference at its native
size), and an **Autogrow** `images` input: template name `image`,
generated names `image_1`..`image_16`, min 0. So exactly 16 reference
image slots exist.

Outputs: `0` = positive conditioning, `1` = negative conditioning,
`2` = empty latent sized to the first reference image (or `resolution`
when there are none).

In the edit workflow the KSampler consumes the encoder latent (via
`ComfySwitchNode`, `switch=false`), so **edit output size = first
reference image's size**, rounded to a multiple of 32.

## 3. Architecture

Single Python (FastAPI) service. Python 3.11+, managed with `uv`.
Dependencies: `fastapi`, `uvicorn[standard]`, `httpx`, `python-multipart`.
Dev: `pytest`, `pytest-asyncio` (or anyio), `httpx` (test transport).

```
oui-to-comfy-proxy/
  pyproject.toml
  .python-version
  README.md
  workflows/
    imagegen.json      # text-to-image template (from examples/imagegen.json)
    imageedit.json     # edit template (from examples/imageedit.json, seed fixed)
  oui-to-comfy-proxy/
    __init__.py
    config.py          # env-based settings
    comfyui.py         # async ComfyUI client
    templates.py       # template loading + per-request graph surgery
    openai_api.py      # OpenAI request/response models + mapping helpers
    main.py            # FastAPI app, routes, auth
  tests/
    test_templates.py
    test_api.py
```

### 3.1 Modules

- **config** — plain env reading, one `Settings` dataclass.
- **templates** — `load_templates(workflow_dir)` at startup;
  `build_generation_graph(prompt, width, height, seed)` and
  `build_edit_graph(prompt, negative_prompt, image_filenames, seed)`
  return fresh graph dicts (templates deep-copied per request).
- **comfyui** — `ComfyUIClient` over `httpx.AsyncClient`:
  `upload_image(bytes, filename, content_type)`,
  `queue(graph) -> prompt_id`, `wait(prompt_id) -> history entry`,
  `fetch_image(filename, subfolder) -> bytes`.
- **openai_api** — pydantic models for the OpenAI request/response
  shapes and small mappers (size parsing, data assembly).
- **main** — wires the above, defines routes and auth.

## 4. Endpoints

### 4.1 `POST /v1/images/generations`

Request: JSON body per §2.1. Handling:

1. `prompt` required. `n` default 1, integer ≥1, **capped at 8**
   (400 if larger).
2. `size`: `"WxH"` with positive ints → each dimension rounded to the
   nearest multiple of 32, clamped to [32, 4096]. `"auto"`, empty, or
   absent → 1024x1024. Unparseable non-auto string → 400.
3. Per run i in 1..n:
   - graph = `build_generation_graph(prompt, w, h, random_seed)`,
     seed = random 32-bit unsigned int, fresh each run.
   - queue → wait → fetch the `SaveImageAdvanced` output image.
4. Response 200:

```json
{"created": 1750000000, "data": [{"b64_json": "<png b64>", ...one per run}]}
```

`response_format` and unknown fields are accepted and ignored; the
response is always `b64_json`.

### 4.2 `POST /v1/images/edits`

Request: multipart per §2.2. Handling:

1. Collect image parts in multipart order: part name `image`, `image[]`,
   or `image[...]`. Zero images → 400. More than 16 → 400 (node
   autogrow limit).
2. `prompt` required. `n` default 1, capped at 8. `size` and
   `background` accepted and ignored (edit output size = first
   reference image's size, the user's validated setup). **Image order
   matters: image 1 determines output dimensions.**
3. Upload all images to ComfyUI in parallel via `POST /upload/image`,
   each renamed `oui-to-comfy-proxy_{8-hex}_{i}{ext}` (ext from content type,
   default `.png`) to avoid cross-request filename collisions;
   `overwrite=true`, no subfolder.
4. Per run i in 1..n: graph = `build_edit_graph(prompt, "", filenames,
   random_seed)`; queue → wait → fetch output. Runs reuse the uploaded
   filenames; only the seed changes.
5. Response 200: same shape as generation.

### 4.3 `GET /v1/models`

`{"object": "list", "data": [{"id": "<MODEL_NAME>", "object": "model",
"created": 0, "owned_by": "oui-to-comfy-proxy"}]}`

### 4.4 `GET /health`

200 `{"status": "ok"}`. Not auth-protected.

### 4.5 Auth

If `PROXY_API_KEY` is set, every `/v1/*` route requires
`Authorization: Bearer <key>` (constant-time compare); mismatch/missing
→ 401. If unset, the proxy is open (local-network assumption).

## 5. Graph surgery

Nodes are located by `class_type`, never by node ID — templates may be
re-numbered freely.

**Generation** (template `imagegen.json`):

- `TextEncodeQwenImage21`: set `prompt` (and `negative_prompt` =
  template value, default `""`); set `resolution` = max(width, height)
  (cosmetic when no reference images are connected, kept consistent).
- `EmptyLatentImage`: set `width`, `height`.
- `KSampler`: set `seed`.

**Edit** (template `imageedit.json`):

1. Find `TextEncodeQwenImage21`. Remove every existing input whose key
   matches `images.image_(\d+)`; record each linked source node ID.
2. Delete recorded source nodes that are `LoadImage` and referenced by
   no other node (safety: if still referenced, leave it in place).
3. For i in 1..N add node `oui-to-comfy-proxy_load_{i}` =
   `{"inputs": {"image": "<uploaded filename>"}, "class_type": "LoadImage"}`
   and wire `images.image_{i}` = `[oui-to-comfy-proxy_load_{i}, 0]`.
4. Set `prompt`, `negative_prompt` (`""`), and `KSampler.seed`.
5. Leave `ComfySwitchNode` (`switch=false`), `ResolutionSelector`,
   `EmptyLatentImage`, `QwenImage21Cache`, loaders, and
   `SaveImageAdvanced` untouched.

**Output collection** (both flows): from the history entry, find the
output of the `SaveImageAdvanced` node: `outputs[node_id]["images"][0]`
→ `{filename, subfolder, type}` → `GET /view`.

**Template fix (one-time, part of this work):** in
`workflows/imageedit.json`, `KSampler.seed` is the string
`"qwen3vl_8b_fp8_scaled.safetensors"` — replace with `0`. The proxy
always overwrites the seed, but the template should be valid on its own.

## 6. ComfyUI client behavior

- One `httpx.AsyncClient`, base URL `COMFYUI_URL`, per-call timeout 30s.
- `queue`: `POST /prompt` with a per-process random `client_id`.
- `wait`: poll `GET /history/{prompt_id}` every `POLL_INTERVAL`
  seconds (default 0.5) until the entry appears and
  `status.status_str` is terminal (`success`/`error`), or the
  per-request `TIMEOUT` elapses → timeout error.
- Errors from ComfyUI (non-2xx, execution status `error`, missing
  output) map to 502 with the upstream message; timeout maps to 504.
- ComfyUI unreachable on the first call of a request → 502.

## 7. Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `MODEL_NAME` | `qwen-image-2.1` | Model ID served at `/v1/models` |
| `WORKFLOW_DIR` | `./workflows` | Directory with the two template JSONs |
| `TIMEOUT` | `600` | Max seconds to wait for a ComfyUI run |
| `POLL_INTERVAL` | `0.5` | Seconds between `/history` polls |
| `PROXY_API_KEY` | *(unset = open)* | Required Bearer key when set |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address for the run script |

Run: `uv run uvicorn oui-to-comfy-proxy.main:app --host $HOST --port $PORT`.

## 8. Error summary

| Condition | Status |
|---|---|
| Missing prompt | 400 |
| Unparseable `size` | 400 |
| Edit: 0 images or >16 images | 400 |
| `n` > 8 | 400 |
| Bad/missing Bearer key (when `PROXY_API_KEY` set) | 401 |
| ComfyUI unreachable or run failed | 502 |
| Run exceeded `TIMEOUT` | 504 |

## 9. Testing

**Unit (`tests/test_templates.py`)** — pure graph surgery, no network:

- generation: prompt/seed/width/height/resolution set correctly;
- edit N=1, 2, 5: exactly N `LoadImage` nodes, wired to
  `image_1..N`, all placeholder loads and their links removed;
- edit N=16 works;
- prompt/negative/seed set; unrelated nodes (loaders, switch, save,
  cache) byte-identical to template;
- templates are not mutated across calls (deep-copy check).

**Integration (`tests/test_api.py`)** — FastAPI `TestClient` with a
stubbed ComfyUI (httpx mock transport / small in-test ASGI stub):

- full generation flow: JSON in → b64_json out, size rounding
  (e.g. `1000x1000` → `992x992`, since 1000/32 = 31.25 → 31*32),
  `auto` default, `n=2` → two runs, two seeds;
- full edit flow: single `image` part; repeated `image[]` parts (2 and
  5 images); part order preserved → upload order → `image_1..N` order;
  `size`/`background` ignored;
- 0 images → 400; 17 images → 400; n>8 → 400;
- auth: 401 without/with wrong key when key configured;
- ComfyUI 500 → 502; history error status → 502; timeout → 504.

Run with `uv run pytest`.

## 10. Open WebUI setup (README content)

- Image generation: engine `openai`, base URL
  `http://<proxy-host>:8000/v1`, API key = `PROXY_API_KEY` (or any
  string if the proxy is open), model `qwen-image-2.1`, size as desired.
- Image editing: engine `openai`, same base URL/key, model
  `qwen-image-2.1`, leave `IMAGE_EDIT_SIZE` on auto (size is ignored
  anyway; output matches the first attached image).
- Note in README: attach images in the order you expect them referenced;
  the first image sets the output size.

## 11. Out of scope

- SSE/streaming, `url` response format, background param semantics,
  multiple models, workflow auto-discovery, Docker packaging,
  modifying the ComfyUI clone or Open WebUI.
