import pytest
import json
from unittest.mock import patch, MagicMock
from backend.pipeline.intent import recognize_intent


def make_mock_response(content_dict):
    """Helper to create a mock DashScope response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.output = MagicMock()
    mock.output.choices = [
        MagicMock(message=MagicMock(content=json.dumps(content_dict)))
    ]
    return mock


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_returns_structured_data(mock_gen):
    mock_gen.call.return_value = make_mock_response({
        "category": "角色",
        "intent": "action_pose",
        "character": "艾琳",
        "pose": "战斗姿态",
        "mood": "张力",
        "angle": "正面",
        "extra_description": None
    })

    result = recognize_intent("生成一张艾琳的战斗姿态")

    assert result["category"] == "角色"
    assert result["intent"] == "action_pose"
    assert result["character"] == "艾琳"
    assert result["pose"] == "战斗姿态"


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_fills_missing_fields_with_none(mock_gen):
    mock_gen.call.return_value = make_mock_response({
        "category": "场景",
        "intent": "scene_generation",
        "character": None,
        "pose": None,
        "mood": None,
        "angle": "鸟瞰",
        "extra_description": "古城"
    })

    result = recognize_intent("生成古代城市鸟瞰图")

    assert result["character"] is None
    assert result["angle"] == "鸟瞰"


@patch("backend.pipeline.intent.Generation")
def test_recognize_intent_handles_api_error(mock_gen):
    mock_gen.call.side_effect = Exception("API error")

    with pytest.raises(Exception, match="Intent recognition failed"):
        recognize_intent("测试prompt")
