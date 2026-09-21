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

import copy

from imgproxy.templates import build_generation_graph


def _id_by_type(graph, class_type):
    return next(i for i, n in graph.items() if n["class_type"] == class_type)


def test_generation_sets_prompt_size_and_seed():
    t = load_templates(WORKFLOWS)
    g = build_generation_graph(t.generation, "a cat", 1024, 768, 42)
    text_id = _id_by_type(g, "TextEncodeQwenImage21")
    assert g[text_id]["inputs"]["prompt"] == "a cat"
    assert g[text_id]["inputs"]["resolution"] == 1024
    lat_id = _id_by_type(g, "EmptyLatentImage")
    assert g[lat_id]["inputs"]["width"] == 1024
    assert g[lat_id]["inputs"]["height"] == 768
    ks_id = _id_by_type(g, "KSampler")
    assert g[ks_id]["inputs"]["seed"] == 42


def test_generation_does_not_mutate_template():
    t = load_templates(WORKFLOWS)
    before = copy.deepcopy(t.generation)
    build_generation_graph(t.generation, "x", 512, 512, 7)
    assert t.generation == before
