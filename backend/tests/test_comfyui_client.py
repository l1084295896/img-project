import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
from backend.pipeline.comfyui_client import ComfyUIClient


@pytest.fixture
def client():
    return ComfyUIClient(base_url="http://localhost:8188", timeout=10)


@pytest.fixture
def sample_workflow():
    return {"nodes": {"1": {"type": "KSampler"}}, "links": []}


@pytest.mark.asyncio
async def test_submit_workflow_returns_prompt_id(client, sample_workflow):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prompt_id": "abc123"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        prompt_id = await client.submit_workflow(sample_workflow)

    assert prompt_id == "abc123"


@pytest.mark.asyncio
async def test_submit_workflow_raises_on_error(client, sample_workflow):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(httpx.HTTPStatusError):
            await client.submit_workflow(sample_workflow)


@pytest.mark.asyncio
async def test_wait_for_result_returns_image_bytes(client):
    mock_history = MagicMock()
    mock_history.status_code = 200
    mock_history.raise_for_status = MagicMock()
    mock_history.json.return_value = {
        "abc123": {
            "outputs": {
                "7": {
                    "images": [
                        {"filename": "output_00001.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }

    mock_image = MagicMock()
    mock_image.status_code = 200
    mock_image.raise_for_status = MagicMock()
    mock_image.content = b"fake_image_bytes"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [mock_history, mock_image]
        result = await client.wait_for_result("abc123")

    assert result == b"fake_image_bytes"


@pytest.mark.asyncio
async def test_wait_for_result_retries_until_ready(client):
    """First two polls return empty history, third returns the image."""
    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.raise_for_status = MagicMock()
    mock_empty.json.return_value = {}

    mock_ready = MagicMock()
    mock_ready.status_code = 200
    mock_ready.raise_for_status = MagicMock()
    mock_ready.json.return_value = {
        "abc123": {
            "outputs": {
                "7": {
                    "images": [
                        {"filename": "out.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }

    mock_image = MagicMock()
    mock_image.status_code = 200
    mock_image.raise_for_status = MagicMock()
    mock_image.content = b"delayed_result"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [mock_empty, mock_empty, mock_ready, mock_image]
        result = await client.wait_for_result("abc123")

    assert result == b"delayed_result"


@pytest.mark.asyncio
async def test_wait_for_result_raises_timeout(client):
    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.raise_for_status = MagicMock()
    mock_empty.json.return_value = {}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_empty
        with pytest.raises(TimeoutError, match="timed out"):
            await client.wait_for_result("abc123")
