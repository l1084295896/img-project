import pytest
import json
from unittest.mock import patch, MagicMock
from backend.pipeline.intent import rewrite_prompt


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
def test_rewrite_prompt_returns_positive_and_negative(mock_gen):
    mock_gen.call.return_value = make_mock_response({
        "positive": "guofeng, Chinese style, masterpiece, best quality, a warrior woman standing",
        "negative": "blurry, low quality, bad anatomy, extra limbs"
    })

    result = rewrite_prompt("一名女战士站立")

    assert "positive" in result
    assert "negative" in result
    assert "guofeng" in result["positive"]
    assert "blurry" in result["negative"]


@patch("backend.pipeline.intent.Generation")
def test_rewrite_prompt_strips_whitespace(mock_gen):
    mock_gen.call.return_value = make_mock_response({
        "positive": "  guofeng, Chinese style, a landscape  ",
        "negative": "  blurry, low quality  "
    })

    result = rewrite_prompt("山水风景")

    assert result["positive"] == "guofeng, Chinese style, a landscape"
    assert result["negative"] == "blurry, low quality"


@patch("backend.pipeline.intent.Generation")
def test_rewrite_prompt_handles_non_200_status(mock_gen):
    mock = make_mock_response({})
    mock.status_code = 403
    mock_gen.call.return_value = mock

    with pytest.raises(Exception, match="Prompt rewrite failed"):
        rewrite_prompt("测试prompt")


@patch("backend.pipeline.intent.Generation")
def test_rewrite_prompt_handles_invalid_json(mock_gen):
    mock = MagicMock()
    mock.status_code = 200
    mock.output = MagicMock()
    mock.output.choices = [
        MagicMock(message=MagicMock(content="这不是JSON"))
    ]
    mock_gen.call.return_value = mock

    with pytest.raises(Exception, match="Prompt rewrite failed"):
        rewrite_prompt("测试prompt")


@patch("backend.pipeline.intent.Generation")
def test_rewrite_prompt_handles_api_error(mock_gen):
    mock_gen.call.side_effect = Exception("API error")

    with pytest.raises(Exception, match="Prompt rewrite failed"):
        rewrite_prompt("测试prompt")


def test_rewrite_prompt_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty string"):
        rewrite_prompt("")
    with pytest.raises(ValueError, match="non-empty string"):
        rewrite_prompt("   ")
