import json
import copy


def fill_prompts(workflow: dict, template: dict, character_name: str) -> dict:
    """Replace prompt placeholders in workflow JSON with template values."""
    workflow_str = json.dumps(workflow, ensure_ascii=False)
    positive = template["positive_prompt"].replace("{character_name}", character_name)
    negative = template["negative_prompt"].replace("{character_name}", character_name)
    workflow_str = workflow_str.replace("{PROMPT}", positive)
    workflow_str = workflow_str.replace("{NEGATIVE}", negative)
    return json.loads(workflow_str)


def build_workflow(template: dict, base_workflow: dict, character_name: str) -> dict:
    """Build a complete ComfyUI workflow by filling a base skeleton with template params.

    Does not mutate the input base_workflow.
    """
    workflow = fill_prompts(copy.deepcopy(base_workflow), template, character_name)
    params = template.get("workflow_params", {})

    for node in workflow.get("nodes", {}).values():
        node_type = node.get("type", "")

        if node_type == "KSampler":
            if "steps" in params:
                node["inputs"]["steps"] = params["steps"]
            if "cfg_scale" in params:
                node["inputs"]["cfg"] = params["cfg_scale"]
            if "sampler" in params:
                node["inputs"]["sampler_name"] = params["sampler"]

        elif node_type == "LoraLoader":
            if "lora_strength" in params:
                node["inputs"]["strength_model"] = params["lora_strength"]
                node["inputs"]["strength_clip"] = params["lora_strength"]

        elif node_type == "EmptyLatentImage":
            if "width" in params:
                node["inputs"]["width"] = params["width"]
            if "height" in params:
                node["inputs"]["height"] = params["height"]

    return workflow
