import json
import os
import pytest
from backend.pipeline.workflow_builder import build_workflow, fill_prompts


@pytest.fixture
def base_workflow():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "mock_base_workflow.json")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def sample_template():
    return {
        "id": "char_action_fight",
        "positive_prompt": "(masterpiece:1.2), {character_name}, battle stance",
        "negative_prompt": "blurry, low quality",
        "workflow_params": {
            "steps": 30,
            "cfg_scale": 8.0,
            "width": 1024,
            "height": 768,
            "sampler": "dpmpp_2m_karras",
            "lora_strength": 0.75
        }
    }


def test_fill_prompts_replaces_placeholders(base_workflow, sample_template):
    character_name = "艾琳"
    filled = fill_prompts(base_workflow, sample_template, character_name)

    workflow_str = json.dumps(filled, ensure_ascii=False)
    assert "{PROMPT}" not in workflow_str
    assert "{NEGATIVE}" not in workflow_str
    assert "{character_name}" not in workflow_str
    assert "艾琳" in workflow_str
    assert "(masterpiece:1.2)" in workflow_str


def test_build_workflow_sets_ksampler_params(base_workflow, sample_template):
    result = build_workflow(sample_template, base_workflow, "艾琳")

    ksampler = None
    for node in result["nodes"].values():
        if node["type"] == "KSampler":
            ksampler = node
            break

    assert ksampler is not None
    assert ksampler["inputs"]["steps"] == 30
    assert ksampler["inputs"]["cfg"] == 8.0
    assert ksampler["inputs"]["sampler_name"] == "dpmpp_2m_karras"


def test_build_workflow_sets_lora_strength(base_workflow, sample_template):
    result = build_workflow(sample_template, base_workflow, "艾琳")

    lora = None
    for node in result["nodes"].values():
        if node["type"] == "LoraLoader":
            lora = node
            break

    assert lora is not None
    assert lora["inputs"]["strength_model"] == 0.75
    assert lora["inputs"]["strength_clip"] == 0.75


def test_build_workflow_does_not_mutate_input(base_workflow, sample_template):
    original = json.dumps(base_workflow)
    build_workflow(sample_template, base_workflow, "艾琳")
    assert json.dumps(base_workflow) == original
