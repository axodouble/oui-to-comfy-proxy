# oui-to-comfy-proxy

OpenAI-compatible proxy that serves Qwen Image 2.1 (running in ComfyUI) to
Open WebUI. It translates `/v1/images/generations` and `/v1/images/edits`
into ComfyUI prompt graphs built from the templates in `workflows/`,
mapping any number of edit input images (1-16) onto the workflow's
`images.image_1..image_N` slots automatically.

## Run

    uv sync
    export COMFYUI_URL=http://127.0.0.1:8188
    uv run uvicorn oui-to-comfy-proxy.main:app --host 0.0.0.0 --port 8000

## Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `MODEL_NAME` | `qwen-image-2.1` | Model ID served at `/v1/models` |
| `WORKFLOW_DIR` | `./workflows` | Template directory (relative paths resolve from the repo root) |
| `TIMEOUT` | `600` | Max seconds to wait for a ComfyUI run |
| `POLL_INTERVAL` | `0.5` | Seconds between `/history` polls |
| `PROXY_API_KEY` | *(unset = open)* | Required Bearer key on `/v1/*` when set |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |

## Open WebUI setup

Image generation: engine `openai`, base URL `http://<proxy-host>:8000/v1`,
API key = your `PROXY_API_KEY` (any string if the proxy is open), model
`qwen-image-2.1`, size as desired.

Image editing: engine `openai`, same base URL and key, model
`qwen-image-2.1`. Leave `IMAGE_EDIT_SIZE` on auto - the edit output size
always follows the first attached image (rounded to a multiple of 32), so
attach images in the order you expect them to be referenced.

## Workflows

Templates live in `workflows/`. Nodes are matched by `class_type`, so you
can renumber or rename node IDs freely. The proxy overwrites the prompt,
seed, size (generation), and input images (editing) on every request;
everything else (model loaders, cache, switch, save settings) is used as-is.
To change the model or sampler, edit the template JSON.

## Tests

    uv run pytest
