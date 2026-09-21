from pathlib import Path

from imgproxy.templates import load_templates

WORKFLOWS = Path(__file__).parent.parent / "workflows"


def test_templates_load():
    t = load_templates(WORKFLOWS)
    for graph, expected in (
        (
            t.generation,
            [
                "TextEncodeQwenImage21",
                "KSampler",
                "EmptyLatentImage",
                "SaveImageAdvanced",
            ],
        ),
        (t.edit, ["TextEncodeQwenImage21", "KSampler", "SaveImageAdvanced"]),
    ):
        class_types = {n["class_type"] for n in graph.values()}
        for node_type in expected:
            assert node_type in class_types


def test_edit_template_seed_is_int():
    t = load_templates(WORKFLOWS)
    ks = next(n for n in t.edit.values() if n["class_type"] == "KSampler")
    assert ks["inputs"]["seed"] == 0
