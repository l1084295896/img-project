import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app, tasks


@pytest.fixture
def client():
    tasks.clear()
    return TestClient(app)


def test_generate_returns_task_id(client):
    with patch("backend.main.asyncio.create_task"):
        resp = client.post("/api/generate", json={"prompt": "生成艾琳的战斗姿态"})

    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert len(data["task_id"]) == 8


def test_get_result_returns_404_for_unknown_task(client):
    resp = client.get("/api/result/nonexist")
    assert resp.status_code == 404


def test_get_result_returns_processing_status(client):
    tasks["test123"] = {
        "status": "processing",
        "steps": [
            {"name": "intent", "status": "done"},
            {"name": "template", "status": "in_progress"}
        ],
        "image": None,
        "info": None,
        "error": None,
    }

    resp = client.get("/api/result/test123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert len(data["steps"]) == 2


def test_get_result_returns_done_with_image(client):
    tasks["done1"] = {
        "status": "done",
        "steps": [
            {"name": "intent", "status": "done"},
            {"name": "template", "status": "done"},
            {"name": "workflow", "status": "done"},
            {"name": "generate", "status": "done"},
        ],
        "image": "iVBORw0KGgo=",
        "info": {"character": "艾琳", "template": "char_action_fight", "elapsed": 12.3},
        "error": None,
    }

    resp = client.get("/api/result/done1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["image"] == "iVBORw0KGgo="
    assert data["info"]["character"] == "艾琳"


def test_frontend_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
