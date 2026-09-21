import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_IMAGES = 16
IMAGE_SLOT_RE = re.compile(r"^images\.image_(\d+)$")


class TemplateError(Exception):
    pass


@dataclass(frozen=True)
class Templates:
    generation: dict
    edit: dict


def _find_node(graph: dict, class_type: str) -> str:
    for node_id, node in graph.items():
        if node.get("class_type") == class_type:
            return node_id
    raise TemplateError(f"no {class_type} node in template")


def load_templates(workflow_dir: str | Path) -> Templates:
    d = Path(workflow_dir)
    generation = json.loads((d / "imagegen.json").read_text())
    edit = json.loads((d / "imageedit.json").read_text())
    for node_type in ("TextEncodeQwenImage21", "KSampler", "SaveImageAdvanced"):
        _find_node(generation, node_type)
        _find_node(edit, node_type)
    _find_node(generation, "EmptyLatentImage")
    return Templates(generation, edit)


def find_save_node_id(graph: dict) -> str:
    return _find_node(graph, "SaveImageAdvanced")


def build_generation_graph(
    template: dict, prompt: str, width: int, height: int, seed: int
) -> dict:
    graph = copy.deepcopy(template)
    text = graph[_find_node(graph, "TextEncodeQwenImage21")]["inputs"]
    text["prompt"] = prompt
    text["resolution"] = max(width, height)
    latent = graph[_find_node(graph, "EmptyLatentImage")]["inputs"]
    latent["width"] = width
    latent["height"] = height
    graph[_find_node(graph, "KSampler")]["inputs"]["seed"] = seed
    return graph


def _is_referenced(node_id: str, graph: dict) -> bool:
    """True if any node links to node_id via a [node_id, slot] input value."""
    for node in graph.values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and value and value[0] == node_id:
                return True
    return False


def build_edit_graph(
    template: dict, prompt: str, negative_prompt: str, image_filenames: list[str], seed: int
) -> dict:
    if not 1 <= len(image_filenames) <= MAX_IMAGES:
        raise ValueError(f"expected 1..{MAX_IMAGES} images, got {len(image_filenames)}")
    graph = copy.deepcopy(template)
    text_id = _find_node(graph, "TextEncodeQwenImage21")
    inputs = graph[text_id]["inputs"]
    inputs["prompt"] = prompt
    inputs["negative_prompt"] = negative_prompt

    drop = set()
    for key in list(inputs):
        if not IMAGE_SLOT_RE.match(key):
            continue
        value = inputs[key]
        if isinstance(value, list) and value:
            drop.add(value[0])
        del inputs[key]

    for node_id in drop:
        node = graph.get(node_id)
        if node is None or node.get("class_type") != "LoadImage":
            continue
        if _is_referenced(node_id, graph):
            continue
        del graph[node_id]

    for i, filename in enumerate(image_filenames, start=1):
        node_id = f"oui-to-comfy-proxy_load_{i}"
        graph[node_id] = {
            "inputs": {"image": filename},
            "class_type": "LoadImage",
            "_meta": {"title": f"Proxy Input {i}"},
        }
        inputs[f"images.image_{i}"] = [node_id, 0]

    graph[_find_node(graph, "KSampler")]["inputs"]["seed"] = seed
    return graph
