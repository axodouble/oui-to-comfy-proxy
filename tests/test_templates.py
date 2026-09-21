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


import pytest

from imgproxy.templates import MAX_IMAGES, build_edit_graph


def _image_nodes(g):
    return {i: n for i, n in g.items() if n["class_type"] == "LoadImage"}


def _slots(g):
    text_id = _id_by_type(g, "TextEncodeQwenImage21")
    return {
        k: v for k, v in g[text_id]["inputs"].items() if k.startswith("images.image_")
    }


def test_edit_maps_n_images_to_n_slots():
    t = load_templates(WORKFLOWS)
    for n in (1, 2, 5):
        names = [f"img_{i}.png" for i in range(n)]
        g = build_edit_graph(t.edit, "p", "n", names, 1)
        assert len(_image_nodes(g)) == n
        assert sorted(_slots(g)) == [f"images.image_{i}" for i in range(1, n + 1)]
        for i, name in enumerate(names, start=1):
            src_id = _slots(g)[f"images.image_{i}"][0]
            assert _image_nodes(g)[src_id]["inputs"]["image"] == name


def test_edit_removes_placeholder_loads():
    t = load_templates(WORKFLOWS)
    g = build_edit_graph(t.edit, "p", "n", ["a.png"], 1)
    for node in _image_nodes(g).values():
        assert node["inputs"]["image"] == "a.png"


def test_edit_max_images():
    t = load_templates(WORKFLOWS)
    g = build_edit_graph(t.edit, "p", "n", [f"i{i}.png" for i in range(MAX_IMAGES)], 1)
    assert len(_image_nodes(g)) == MAX_IMAGES


def test_edit_rejects_out_of_range():
    t = load_templates(WORKFLOWS)
    with pytest.raises(ValueError):
        build_edit_graph(t.edit, "p", "n", [], 1)
    with pytest.raises(ValueError):
        build_edit_graph(t.edit, "p", "n", ["a.png"] * (MAX_IMAGES + 1), 1)


def test_edit_sets_prompt_and_seed():
    t = load_templates(WORKFLOWS)
    g = build_edit_graph(t.edit, "new prompt", "new neg", ["a.png"], 99)
    text_id = _id_by_type(g, "TextEncodeQwenImage21")
    assert g[text_id]["inputs"]["prompt"] == "new prompt"
    assert g[text_id]["inputs"]["negative_prompt"] == "new neg"
    assert g[_id_by_type(g, "KSampler")]["inputs"]["seed"] == 99


def test_edit_static_nodes_untouched():
    t = load_templates(WORKFLOWS)
    g = build_edit_graph(t.edit, "p", "n", ["a.png"], 1)
    for class_type in (
        "UnetLoaderGGUF",
        "CLIPLoader",
        "VAELoader",
        "QwenImage21Cache",
        "ComfySwitchNode",
        "ResolutionSelector",
        "EmptyLatentImage",
        "SaveImageAdvanced",
        "UnloadAllModels",
    ):
        before = {i: n for i, n in t.edit.items() if n["class_type"] == class_type}
        after = {i: n for i, n in g.items() if n["class_type"] == class_type}
        assert before == after


def test_edit_does_not_mutate_template():
    t = load_templates(WORKFLOWS)
    before = copy.deepcopy(t.edit)
    build_edit_graph(t.edit, "x", "y", ["a.png"], 7)
    assert t.edit == before
