import pytest
from backend.pipeline.template_matcher import load_templates, match_template, calculate_score


def test_load_templates_loads_all_json_files(templates_dir):
    templates = load_templates(templates_dir)
    assert len(templates) == 3
    ids = {t["id"] for t in templates}
    assert "char_stand_front" in ids
    assert "char_action_fight" in ids
    assert "char_expression" in ids


def test_calculate_score_exact_match():
    params = {"pose": "站立", "mood": "中性", "angle": "正面"}
    template = {"tags": ["站立", "正面", "中性", "全身"]}

    score = calculate_score(params, template)
    assert score == 3 / 4  # 3 tags matched out of 4


def test_calculate_score_no_match():
    params = {"pose": "战斗", "mood": "张力"}
    template = {"tags": ["站立", "正面", "中性", "全身"]}

    score = calculate_score(params, template)
    assert score == 0.0


def test_calculate_score_handles_empty_tags():
    params = {"pose": "战斗"}
    template = {"tags": []}

    score = calculate_score(params, template)
    assert score == 0.0


def test_match_template_returns_best_match(templates_dir):
    params = {"pose": "战斗", "mood": "张力", "angle": "正面", "category": "角色"}

    best, score = match_template(params, templates_dir)
    assert best["id"] == "char_action_fight"
    assert score > 0


def test_match_template_returns_low_score_when_no_good_match(templates_dir):
    params = {"pose": "飞行", "mood": "愉快", "angle": "仰视", "category": "角色"}

    best, score = match_template(params, templates_dir)
    assert score < 0.3
