import json
import os


def load_templates(templates_dir: str) -> list[dict]:
    """Load all JSON template files from the templates directory."""
    templates = []
    if not os.path.isdir(templates_dir):
        return templates

    for filename in sorted(os.listdir(templates_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(templates_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                template = json.load(f)
                templates.append(template)

    return templates


def calculate_score(params: dict, template: dict) -> float:
    """Score a template against extracted params by tag intersection.

    Score = matched_tags / total_template_tags.
    """
    param_keywords = set()
    for key in ("pose", "mood", "angle", "category"):
        value = params.get(key)
        if value:
            param_keywords.add(value)

    template_tags = set(template.get("tags", []))
    if not template_tags:
        return 0.0

    intersection = param_keywords & template_tags
    return len(intersection) / len(template_tags)


def match_template(params: dict, templates_dir: str) -> tuple[dict | None, float]:
    """Find the best matching template for the given params.

    Returns (best_template, score). If no templates loaded, returns (None, 0.0).
    Score ranges from 0.0 (no match) to 1.0 (perfect match).
    """
    templates = load_templates(templates_dir)
    if not templates:
        return None, 0.0

    best_template = None
    best_score = 0.0

    for template in templates:
        score = calculate_score(params, template)
        if score > best_score:
            best_score = score
            best_template = template

    return best_template, best_score
