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
